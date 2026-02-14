#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Few-shot learning comparison EVALUATION: Load pre-trained models and test on holdout test folds.
This evaluates already-trained models on the test data they haven't seen.

Command-line version with argparse
"""

"""python 9lyp_fewshot_evaluation.py \\
  --cross_attn_base_model /n/netscratch/shakhnovich_lab/Lab/jonathanfeldman/AF3Complex/ProSST_Max_loss_all_splits/60_percent/fold_0_full_model.pth \\
  --simple_base_model /n/netscratch/shakhnovich_lab/Lab/jonathanfeldman/AF3Complex/prosst_no_scores/fold_0_model.pth \\
  --cache_path /n/netscratch/shakhnovich_lab/Lab/jonathanfeldman/ProSST_PPI-main/bloom_antibodies/9lyp_score_cache.npz \\
  --npz_dir /n/netscratch/shakhnovich_lab/Lab/jonathanfeldman/ProSST_PPI-main/bloom_antibodies/9lyp_merged \\
  --models_dir /n/netscratch/shakhnovich_lab/Lab/jonathanfeldman/ProSST_PPI-main/bloom_antibodies/9lyp_shot_comparison_output/ \\
  --output_dir /n/netscratch/shakhnovich_lab/Lab/jonathanfeldman/ProSST_PPI-main/bloom_antibodies/9lyp_test_evaluation/ \\
  --antibody_name 9LYP \\
  --train_ratios 0.0 0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 \\
  --random_seed 42
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import json
import argparse


# ============================================================================
# Cross-Attention Model Architecture
# ============================================================================

class CrossEmbeddingAttention(nn.Module):
    def __init__(self, input_dim=3840, reduced_dim=512, num_scores=5, embed_dim=768):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_scores = num_scores
        self.reduced_dim = reduced_dim
        
        self.embedding_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, reduced_dim),
                nn.ReLU(),
                nn.Dropout(0.4)
            ) for _ in range(num_scores)
        ])
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=reduced_dim,
            num_heads=8,
            dropout=0.3,
            batch_first=True
        )
        
        self.attention_norm = nn.LayerNorm(reduced_dim)
        
        self.final_attention = nn.Sequential(
            nn.Linear(num_scores, 32),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(32, num_scores)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        embeddings = x.view(batch_size, self.num_scores, self.embed_dim)
        
        projected = []
        for i, proj in enumerate(self.embedding_projections):
            proj_emb = proj(embeddings[:, i, :])
            projected.append(proj_emb)
        
        projected = torch.stack(projected, dim=1)
        attended, _ = self.cross_attention(projected, projected, projected)
        attended = self.attention_norm(attended + projected)
        
        attended_t = attended.transpose(1, 2)
        attn_logits = self.final_attention(attended_t)
        attn_weights = F.softmax(attn_logits, dim=-1)
        
        output = (attn_weights * attended_t).sum(dim=-1)
        return output


class DDGPredictorCrossAttn(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=768, dropout_rate=0.35, num_hidden=3):
        super().__init__()
        
        layers = []
        in_dim = input_dim
        
        for i in range(num_hidden):
            if i == 0:
                out_dim = hidden_dim
            elif i == 1:
                out_dim = hidden_dim // 2
            else:
                out_dim = hidden_dim // 4
            
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = out_dim
        
        self.mlp = nn.Sequential(*layers)
        
        self.head = nn.Sequential(
            nn.Dropout(dropout_rate * 0.8),
            nn.Linear(in_dim, 1)
        )
    
    def forward(self, xf, xr):
        f = self.head(self.mlp(xf))
        r = self.head(self.mlp(xr))
        return (f - r)/2


class FullModelCrossAttention(nn.Module):
    def __init__(self, input_dim=3840, reduced_dim=512, hidden_dim=768, dropout=0.35, num_hidden=3):
        super().__init__()
        
        self.pooling = CrossEmbeddingAttention(
            input_dim=input_dim, 
            reduced_dim=reduced_dim,
            num_scores=5,
            embed_dim=768
        )
        
        self.ddg_predictor = DDGPredictorCrossAttn(
            input_dim=reduced_dim,
            hidden_dim=hidden_dim,
            dropout_rate=dropout,
            num_hidden=num_hidden
        )
        
        self.ilddt_predictor = DDGPredictorCrossAttn(
            input_dim=reduced_dim,
            hidden_dim=hidden_dim,
            dropout_rate=dropout,
            num_hidden=num_hidden
        )
    
    def forward(self, xf, xr):
        xf_pooled = self.pooling(xf)
        xr_pooled = self.pooling(xr)
        
        ddg_pred = self.ddg_predictor(xf_pooled, xr_pooled).squeeze(-1)
        ilddt_pred = self.ilddt_predictor(xf_pooled, xr_pooled).squeeze(-1)
        
        return ddg_pred, ilddt_pred


# ============================================================================
# Simple Model Architecture
# ============================================================================

class DDGPredictorSimple(nn.Module):
    """
    Simple MLP predictor for 768-dim embeddings (no scores, no cross-attention).
    """
    def __init__(self, input_dim=768, hidden_dim=1024, dropout_rate=0.2, num_hidden=4):
        super().__init__()
        layers = []
        in_dim = input_dim
        for i in range(num_hidden):
            out_dim = hidden_dim // (2**i)
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout_rate)]
            in_dim = out_dim
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, 1)

    def forward(self, xf, xr):
        f = self.head(self.mlp(xf))
        r = self.head(self.mlp(xr))
        return (f - r) / 2


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_cross_attention_data(cache_path):
    """Load pre-computed cache data for cross-attention model."""
    print(f"\n{'='*60}")
    print("LOADING CROSS-ATTENTION DATA (from cache)")
    print(f"{'='*60}")
    
    data = np.load(cache_path)
    Xf = data['Xf'].astype(np.float32)
    Xr = data['Xr'].astype(np.float32)
    y = data['y'].astype(np.float32)
    ids = data['ids']
    
    print(f"✓ Loaded {len(y)} total samples")
    print(f"  Xf shape: {Xf.shape}")
    print(f"  Xr shape: {Xr.shape}")
    
    return Xf, Xr, y, ids


def load_simple_data(combined_npz_dir):
    """
    Load npz files and create simple 768-dim embeddings.
    Max pool each embedding to get 768-dim vectors.
    """
    print(f"\n{'='*60}")
    print("LOADING SIMPLE DATA (from npz files)")
    print(f"{'='*60}")
    
    npz_files = [f for f in os.listdir(combined_npz_dir) if f.endswith('.npz')]
    print(f"\nFound {len(npz_files)} npz files")
    
    Xf_list, Xr_list, y_list, ids_list = [], [], [], []
    
    processed = 0
    skipped = 0
    
    for fname in tqdm(npz_files, desc='Processing'):
        try:
            data = np.load(os.path.join(combined_npz_dir, fname))
        except Exception as e:
            print(f"Error loading {fname}: {e}")
            skipped += 1
            continue
        
        # Check for required keys
        if 'Xf' not in data or 'Xr' not in data or 'ddG' not in data or 'index' not in data:
            skipped += 1
            continue
        
        Xf_emb = data['Xf']  # Shape: (L, 768)
        Xr_emb = data['Xr']  # Shape: (L, 768)
        ddg = data['ddG']
        mut_id = str(data['index'])
        
        # Verify 2D embeddings
        if Xf_emb.ndim != 2 or Xr_emb.ndim != 2:
            skipped += 1
            continue
        
        # Max pool to get 768-dim vectors
        Xf_pooled = np.max(Xf_emb, axis=0)  # Shape: (768,)
        Xr_pooled = np.max(Xr_emb, axis=0)  # Shape: (768,)
        
        Xf_list.append(Xf_pooled)
        Xr_list.append(Xr_pooled)
        y_list.append(float(ddg))
        ids_list.append(mut_id)
        processed += 1
    
    print(f"\n✓ Processed: {processed}")
    print(f"✗ Skipped: {skipped}")
    
    if not Xf_list:
        raise ValueError("No data loaded! Check your npz files.")
    
    Xf = np.vstack(Xf_list).astype(np.float32)
    Xr = np.vstack(Xr_list).astype(np.float32)
    y = np.array(y_list, dtype=np.float32)
    ids = np.array(ids_list)
    
    print(f"\nFinal data shapes:")
    print(f"  Xf: {Xf.shape}")
    print(f"  Xr: {Xr.shape}")
    print(f"  y: {y.shape}")
    
    return Xf, Xr, y, ids


# ============================================================================
# Evaluation Functions
# ============================================================================

def evaluate_cross_attention(model, Xf, Xr, y, device, batch_size=32):
    """Evaluate cross-attention model and return predictions and metrics."""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for i in range(0, len(Xf), batch_size):
            batch_xf = torch.tensor(Xf[i:i+batch_size]).to(device)
            batch_xr = torch.tensor(Xr[i:i+batch_size]).to(device)
            
            pred_ddg, _ = model(batch_xf, batch_xr)
            predictions.extend(pred_ddg.cpu().numpy())
    
    predictions = np.array(predictions)
    
    # Calculate metrics
    pearson_r, pearson_p = pearsonr(y, predictions)
    spearman_r, spearman_p = spearmanr(y, predictions)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    mae = mean_absolute_error(y, predictions)
    ss_res = np.sum((y - predictions) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    metrics = {
        'pearson_r': float(pearson_r),
        'pearson_p': float(pearson_p),
        'spearman_r': float(spearman_r),
        'spearman_p': float(spearman_p),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'n_samples': len(y)
    }
    
    return predictions, metrics


def evaluate_simple(model, Xf, Xr, y, device, batch_size=32):
    """Evaluate simple model and return predictions and metrics."""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for i in range(0, len(Xf), batch_size):
            batch_xf = torch.tensor(Xf[i:i+batch_size]).to(device)
            batch_xr = torch.tensor(Xr[i:i+batch_size]).to(device)
            
            pred_ddg = model(batch_xf, batch_xr).squeeze()
            
            # Handle scalar predictions
            if pred_ddg.ndim == 0:
                predictions.append(pred_ddg.cpu().numpy().item())
            else:
                predictions.extend(pred_ddg.cpu().numpy())
    
    predictions = np.array(predictions)
    
    # Calculate metrics
    pearson_r, pearson_p = pearsonr(y, predictions)
    spearman_r, spearman_p = spearmanr(y, predictions)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    mae = mean_absolute_error(y, predictions)
    ss_res = np.sum((y - predictions) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    metrics = {
        'pearson_r': float(pearson_r),
        'pearson_p': float(pearson_p),
        'spearman_r': float(spearman_r),
        'spearman_p': float(spearman_p),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'n_samples': len(y)
    }
    
    return predictions, metrics


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_line_comparison(cross_attn_results, simple_results, output_dir, antibody_name):
    """
    Create line plots showing how both models perform across training sizes.
    """
    train_percentages = sorted(cross_attn_results.keys())
    
    # Extract metrics for both models
    ca_pearson = [cross_attn_results[pct]['metrics']['pearson_r'] for pct in train_percentages]
    ca_spearman = [cross_attn_results[pct]['metrics']['spearman_r'] for pct in train_percentages]
    ca_rmse = [cross_attn_results[pct]['metrics']['rmse'] for pct in train_percentages]
    ca_r2 = [cross_attn_results[pct]['metrics']['r2'] for pct in train_percentages]
    
    simple_pearson = [simple_results[pct]['metrics']['pearson_r'] for pct in train_percentages]
    simple_spearman = [simple_results[pct]['metrics']['spearman_r'] for pct in train_percentages]
    simple_rmse = [simple_results[pct]['metrics']['rmse'] for pct in train_percentages]
    simple_r2 = [simple_results[pct]['metrics']['r2'] for pct in train_percentages]
    
    # Create figure with four subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Pearson correlation plot
    ax = axes[0, 0]
    ax.plot(train_percentages, ca_pearson, 'o-', linewidth=3, markersize=10, 
            color='#2E86AB', label='Cross-Attention', alpha=0.8)
    ax.plot(train_percentages, simple_pearson, 's--', linewidth=3, markersize=10, 
            color='#A23B72', label='Simple', alpha=0.8)
    ax.set_xlabel('Training Set Size (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Pearson Correlation (r)', fontsize=14, fontweight='bold')
    ax.set_title(f'Pearson Correlation vs Training Size\n({antibody_name} - Test Fold)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(train_percentages)
    ax.legend(fontsize=12, loc='best', framealpha=0.9)
    ax.tick_params(labelsize=12)
    
    # Spearman correlation plot
    ax = axes[0, 1]
    ax.plot(train_percentages, ca_spearman, 'o-', linewidth=3, markersize=10, 
            color='#2E86AB', label='Cross-Attention', alpha=0.8)
    ax.plot(train_percentages, simple_spearman, 's--', linewidth=3, markersize=10, 
            color='#A23B72', label='Simple', alpha=0.8)
    ax.set_xlabel('Training Set Size (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Spearman Correlation (ρ)', fontsize=14, fontweight='bold')
    ax.set_title(f'Spearman Correlation vs Training Size\n({antibody_name} - Test Fold)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(train_percentages)
    ax.legend(fontsize=12, loc='best', framealpha=0.9)
    ax.tick_params(labelsize=12)
    
    # RMSE plot
    ax = axes[1, 0]
    ax.plot(train_percentages, ca_rmse, 'o-', linewidth=3, markersize=10, 
            color='#2E86AB', label='Cross-Attention', alpha=0.8)
    ax.plot(train_percentages, simple_rmse, 's--', linewidth=3, markersize=10, 
            color='#A23B72', label='Simple', alpha=0.8)
    ax.set_xlabel('Training Set Size (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('RMSE (kcal/mol)', fontsize=14, fontweight='bold')
    ax.set_title(f'RMSE vs Training Size\n({antibody_name} - Test Fold)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(train_percentages)
    ax.legend(fontsize=12, loc='best', framealpha=0.9)
    ax.tick_params(labelsize=12)
    
    # R² plot
    ax = axes[1, 1]
    ax.plot(train_percentages, ca_r2, 'o-', linewidth=3, markersize=10, 
            color='#2E86AB', label='Cross-Attention', alpha=0.8)
    ax.plot(train_percentages, simple_r2, 's--', linewidth=3, markersize=10, 
            color='#A23B72', label='Simple', alpha=0.8)
    ax.set_xlabel('Training Set Size (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('R² Score', fontsize=14, fontweight='bold')
    ax.set_title(f'R² vs Training Size\n({antibody_name} - Test Fold)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(train_percentages)
    ax.legend(fontsize=12, loc='best', framealpha=0.9)
    ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f'{antibody_name.lower()}_fewshot_test_metrics.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.savefig(plot_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"\n✓ Saved line comparison plot to {plot_path}")
    plt.close()


# ============================================================================
# Argument Parser
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Few-shot learning evaluation: Load and test pre-trained models on holdout test folds'
    )
    
    # Required arguments
    parser.add_argument('--cross_attn_base_model', type=str, required=True,
                       help='Path to cross-attention base model (.pth)')
    parser.add_argument('--simple_base_model', type=str, required=True,
                       help='Path to simple base model (.pth)')
    parser.add_argument('--cache_path', type=str, required=True,
                       help='Path to cross-attention cache (.npz)')
    parser.add_argument('--npz_dir', type=str, required=True,
                       help='Directory containing simple model npz files')
    parser.add_argument('--models_dir', type=str, required=True,
                       help='Directory containing pre-trained models from training')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for results and plots')
    
    # Optional arguments
    parser.add_argument('--antibody_name', type=str, default='Antibody',
                       help='Name of antibody for plot titles (default: Antibody)')
    parser.add_argument('--train_ratios', type=float, nargs='+',
                       default=[0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
                       help='Training set ratios to evaluate (default: 0.0 to 0.8 at 0.1 intervals)')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducible splits (default: 42)')
    
    return parser.parse_args()


# ============================================================================
# Main Function
# ============================================================================

def main():
    args = parse_args()
    
    print("\n" + "="*60)
    print("FEW-SHOT LEARNING EVALUATION")
    print("Test Pre-Trained Models on Holdout Test Folds")
    print("Cross-Attention vs Simple Model")
    print("="*60)
    
    print("\nConfiguration:")
    print(f"  Cross-Attn base:  {args.cross_attn_base_model}")
    print(f"  Simple base:      {args.simple_base_model}")
    print(f"  Cache:            {args.cache_path}")
    print(f"  NPZ dir:          {args.npz_dir}")
    print(f"  Models dir:       {args.models_dir}")
    print(f"  Output dir:       {args.output_dir}")
    print(f"  Antibody name:    {args.antibody_name}")
    print(f"  Train ratios:     {[f'{r*100:.0f}%' for r in args.train_ratios]}")
    print(f"  Random seed:      {args.random_seed}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Device: {device}")
    
    # Set random seed for reproducible splits
    np.random.seed(args.random_seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data for both models
    Xf_ca, Xr_ca, y_ca, ids_ca = load_cross_attention_data(args.cache_path)
    Xf_simple, Xr_simple, y_simple, ids_simple = load_simple_data(args.npz_dir)
    
    # Find common IDs to ensure identical splits
    print(f"\n{'='*60}")
    print("FINDING COMMON SAMPLES")
    print(f"{'='*60}")
    common_ids = set(ids_ca) & set(ids_simple)
    print(f"Cross-Attention samples: {len(ids_ca)}")
    print(f"Simple samples:          {len(ids_simple)}")
    print(f"Common samples:          {len(common_ids)}")
    
    # Filter to common IDs and align order
    common_ids_list = sorted(list(common_ids))
    
    # Create index maps
    ca_idx_map = {id_: i for i, id_ in enumerate(ids_ca)}
    simple_idx_map = {id_: i for i, id_ in enumerate(ids_simple)}
    
    ca_indices = [ca_idx_map[id_] for id_ in common_ids_list]
    simple_indices = [simple_idx_map[id_] for id_ in common_ids_list]
    
    # Filter and align data
    Xf_ca = Xf_ca[ca_indices]
    Xr_ca = Xr_ca[ca_indices]
    y_ca = y_ca[ca_indices]
    ids_ca = np.array(common_ids_list)
    
    Xf_simple = Xf_simple[simple_indices]
    Xr_simple = Xr_simple[simple_indices]
    y_simple = y_simple[simple_indices]
    ids_simple = np.array(common_ids_list)
    
    print(f"✓ Aligned {len(common_ids_list)} samples for both models")
    
    # Verify alignment
    assert np.array_equal(ids_ca, ids_simple), "IDs must match!"
    assert np.allclose(y_ca, y_simple), "Labels must match!"
    
    # Store results
    cross_attn_results = {}
    simple_results = {}
    
    # Iterate over training ratios and LOAD pre-trained models
    for train_ratio in args.train_ratios:
        print(f"\n{'='*60}")
        print(f"EVALUATING {train_ratio*100:.0f}% MODELS ON TEST FOLD")
        print(f"{'='*60}")
        
        pct = int(train_ratio * 100)
        
        if train_ratio == 0.0:
            # Use base models on all data
            print("\nLoading base models (0% = no fine-tuning)")
            
            # Cross-Attention
            ca_model = FullModelCrossAttention().to(device)
            ca_checkpoint = torch.load(args.cross_attn_base_model, map_location=device, weights_only=False)
            if isinstance(ca_checkpoint, dict) and 'model_state_dict' in ca_checkpoint:
                ca_model.load_state_dict(ca_checkpoint['model_state_dict'])
            else:
                ca_model.load_state_dict(ca_checkpoint)
            ca_model.eval()
            
            # Simple
            simple_model = DDGPredictorSimple(
                input_dim=768,
                hidden_dim=1024,
                dropout_rate=0.2,
                num_hidden=4
            ).to(device)
            simple_model.load_state_dict(torch.load(args.simple_base_model, map_location=device))
            simple_model.eval()
            
            # Evaluate on all data
            Xf_eval_ca, Xr_eval_ca, y_eval_ca = Xf_ca, Xr_ca, y_ca
            Xf_eval_simple, Xr_eval_simple, y_eval_simple = Xf_simple, Xr_simple, y_simple
            ids_eval = ids_ca
            
        else:
            # Reproduce the train/test split to get TEST indices
            indices = np.arange(len(y_ca))
            train_idx, test_idx = train_test_split(
                indices,
                test_size=1.0 - train_ratio,
                random_state=args.random_seed,
                shuffle=True
            )
            
            # Load pre-trained models for this percentage
            ca_model_path = os.path.join(args.models_dir, f'cross_attn_model_{pct}pct.pth')
            simple_model_path = os.path.join(args.models_dir, f'simple_model_{pct}pct.pth')
            
            print(f"\nLoading pre-trained models:")
            print(f"  Cross-Attn: {ca_model_path}")
            print(f"  Simple:     {simple_model_path}")
            
            # Load cross-attention model
            ca_model = FullModelCrossAttention().to(device)
            ca_model.load_state_dict(torch.load(ca_model_path, map_location=device))
            ca_model.eval()
            
            # Load simple model
            simple_model = DDGPredictorSimple(
                input_dim=768,
                hidden_dim=1024,
                dropout_rate=0.2,
                num_hidden=4
            ).to(device)
            simple_model.load_state_dict(torch.load(simple_model_path, map_location=device))
            simple_model.eval()
            
            # Get TEST fold data (holdout data the model has never seen)
            Xf_eval_ca = Xf_ca[test_idx]
            Xr_eval_ca = Xr_ca[test_idx]
            y_eval_ca = y_ca[test_idx]
            
            Xf_eval_simple = Xf_simple[test_idx]
            Xr_eval_simple = Xr_simple[test_idx]
            y_eval_simple = y_simple[test_idx]
            
            ids_eval = ids_ca[test_idx]
            
            print(f"Test fold size: {len(test_idx)} samples ({len(test_idx)/len(y_ca)*100:.1f}%)")
            print(f"Train fold size: {len(train_idx)} samples ({len(train_idx)/len(y_ca)*100:.1f}%)")
        
        # Evaluate both models on test fold
        print(f"\n--- Evaluating Models on Test Fold ---")
        
        ca_predictions, ca_metrics = evaluate_cross_attention(
            ca_model, Xf_eval_ca, Xr_eval_ca, y_eval_ca, device
        )
        
        simple_predictions, simple_metrics = evaluate_simple(
            simple_model, Xf_eval_simple, Xr_eval_simple, y_eval_simple, device
        )
        
        print(f"\nCross-Attention performance:")
        print(f"  Pearson r:  {ca_metrics['pearson_r']:.4f}")
        print(f"  Spearman ρ: {ca_metrics['spearman_r']:.4f}")
        print(f"  RMSE:       {ca_metrics['rmse']:.4f}")
        print(f"  R²:         {ca_metrics['r2']:.4f}")
        
        print(f"\nSimple model performance:")
        print(f"  Pearson r:  {simple_metrics['pearson_r']:.4f}")
        print(f"  Spearman ρ: {simple_metrics['spearman_r']:.4f}")
        print(f"  RMSE:       {simple_metrics['rmse']:.4f}")
        print(f"  R²:         {simple_metrics['r2']:.4f}")
        
        # Store results
        cross_attn_results[pct] = {
            'predictions': ca_predictions,
            'true_values': y_eval_ca,
            'ids': ids_eval,
            'metrics': ca_metrics,
            'train_ratio': train_ratio,
            'n_eval': len(y_eval_ca)
        }
        
        simple_results[pct] = {
            'predictions': simple_predictions,
            'true_values': y_eval_simple,
            'ids': ids_eval,
            'metrics': simple_metrics,
            'train_ratio': train_ratio,
            'n_eval': len(y_eval_simple)
        }
    
    # Create line comparison plot
    print(f"\n{'='*60}")
    print("CREATING LINE COMPARISON PLOT")
    print(f"{'='*60}")
    
    plot_line_comparison(cross_attn_results, simple_results, args.output_dir, args.antibody_name)
    
    # Save results
    print(f"\n{'='*60}")
    print("SAVING RESULTS")
    print(f"{'='*60}")
    
    # Summary CSV
    summary_data = []
    for pct in sorted(cross_attn_results.keys()):
        ca_result = cross_attn_results[pct]
        simple_result = simple_results[pct]
        
        summary_data.append({
            'train_pct': pct,
            'n_eval': ca_result['n_eval'],
            'ca_pearson_r': ca_result['metrics']['pearson_r'],
            'ca_spearman_r': ca_result['metrics']['spearman_r'],
            'ca_rmse': ca_result['metrics']['rmse'],
            'ca_r2': ca_result['metrics']['r2'],
            'simple_pearson_r': simple_result['metrics']['pearson_r'],
            'simple_spearman_r': simple_result['metrics']['spearman_r'],
            'simple_rmse': simple_result['metrics']['rmse'],
            'simple_r2': simple_result['metrics']['r2']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(args.output_dir, f'{args.antibody_name.lower()}_test_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"✓ Saved comparison summary to {summary_path}")
    
    # Save detailed predictions
    for pct in sorted(cross_attn_results.keys()):
        ca_result = cross_attn_results[pct]
        simple_result = simple_results[pct]
        
        df = pd.DataFrame({
            'id': ca_result['ids'],
            'true_ddG': ca_result['true_values'],
            'ca_pred_ddG': ca_result['predictions'],
            'simple_pred_ddG': simple_result['predictions'],
            'ca_error': ca_result['predictions'] - ca_result['true_values'],
            'simple_error': simple_result['predictions'] - simple_result['true_values']
        })
        
        pred_path = os.path.join(args.output_dir, 
                                 f'{args.antibody_name.lower()}_predictions_{int(pct)}pct.csv')
        df.to_csv(pred_path, index=False)
        print(f"✓ Saved predictions for {int(pct)}% to {pred_path}")
    
    # Save complete results as JSON
    json_results = {
        'experiment': f'{args.antibody_name} Few-Shot Evaluation',
        'description': 'Test pre-trained models on holdout test folds',
        'antibody': args.antibody_name,
        'cross_attention': {},
        'simple': {}
    }
    
    for pct in sorted(cross_attn_results.keys()):
        ca_result = cross_attn_results[pct]
        simple_result = simple_results[pct]
        
        json_results['cross_attention'][f'{int(pct)}pct'] = {
            'train_ratio': ca_result['train_ratio'],
            'n_eval': ca_result['n_eval'],
            'metrics': ca_result['metrics']
        }
        
        json_results['simple'][f'{int(pct)}pct'] = {
            'train_ratio': simple_result['train_ratio'],
            'n_eval': simple_result['n_eval'],
            'metrics': simple_result['metrics']
        }
    
    json_path = os.path.join(args.output_dir, f'{args.antibody_name.lower()}_test_results.json')
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"✓ Saved all results to {json_path}")
    
    # Print summary table
    print(f"\n{'='*60}")
    print(f"SUMMARY TABLE ({args.antibody_name.upper()} TEST EVALUATION)")
    print(f"{'='*60}")
    print(summary_df.to_string(index=False))
    
    print("\n" + "="*60)
    print("✓ ANALYSIS COMPLETE!")
    print("="*60)
    print(f"\nResults saved to: {args.output_dir}")
    
    return cross_attn_results, simple_results, summary_df


if __name__ == '__main__':
    main()