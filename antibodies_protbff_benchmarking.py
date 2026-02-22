#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate ProSST and ESM2 cross-attention models across multiple training percentages.
Recreates train/test splits to evaluate on proper test sets.

Usage:
  
  For 9LYP:
  
    python antibodies_protbff_benchmarking.py \
        --prosst_model_dir model_benchmarking\9lyp_comparison_output \
        --prosst_cache_path model_benchmarking/score_caches/9lyp_score_cache.npz \
        --esm2_model_dir /n/netscratch/shakhnovich_lab/Lab/jonathanfeldman/ProSST_PPI-main/bloom_antibodies/9lyp_esm2_cross_attn_output/ \
        --esm2_cache_path model_benchmarking/score_caches/9lyp_esm_score_cache.npz \
        --output_dir . \
        --random_seed 42
        
  For 7W9I:
  
     python antibodies_protbff_benchmarking.py \
        --prosst_model_dir model_benchmarking/7w9i_comparison_output \
        --prosst_cache_path model_benchmarking/score_caches/ace2_score_cache.npz \
        --esm2_model_dir /n/netscratch/shakhnovich_lab/Lab/jonathanfeldman/ProSST_PPI-main/few_shot_esm2_cross_attn_output/ \
        --esm2_cache_path model_benchmarking/score_caches/ace2_esm_score_cache.npz \
        --output_dir . \
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
import os
import argparse
import json
import glob


# ============================================================================
# Model Architecture
# ============================================================================

class CrossEmbeddingAttention(nn.Module):
    def __init__(self, input_dim=3840, reduced_dim=512, num_scores=5, embed_dim=768):
        """
        Cross-attention mechanism for protein embeddings.
        
        Args:
            input_dim: Total input dimension (num_scores × embed_dim)
            reduced_dim: Reduced dimension for attention
            num_scores: Number of score embeddings (default: 5)
            embed_dim: Dimension of each embedding (768 for ProSST, 1280 for ESM2)
        """
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


class DDGPredictor(nn.Module):
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
        return (f - r) / 2


class FullModelCrossAttention(nn.Module):
    def __init__(self, input_dim=3840, reduced_dim=512, hidden_dim=768, dropout=0.35, num_hidden=3, embed_dim=768):
        """
        Full cross-attention model for ddG prediction.
        
        Args:
            input_dim: Total input dimension (should be num_scores × embed_dim)
            reduced_dim: Reduced dimension for attention
            hidden_dim: Hidden dimension for predictor
            dropout: Dropout rate
            num_hidden: Number of hidden layers
            embed_dim: Dimension of each embedding (768 for ProSST, 1280 for ESM2)
        """
        super().__init__()
        
        self.pooling = CrossEmbeddingAttention(
            input_dim=input_dim, 
            reduced_dim=reduced_dim,
            num_scores=5,
            embed_dim=embed_dim
        )
        
        self.ddg_predictor = DDGPredictor(
            input_dim=reduced_dim,
            hidden_dim=hidden_dim,
            dropout_rate=dropout,
            num_hidden=num_hidden
        )
        
        self.ilddt_predictor = DDGPredictor(
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
# Data Loading
# ============================================================================

def load_data(cache_path, model_name):
    """Load pre-computed cache data."""
    print(f"\n{'='*70}")
    print(f"LOADING {model_name.upper()} DATA")
    print(f"{'='*70}")
    
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cache file not found: {cache_path}")
    
    print(f"Loading data from: {cache_path}")
    data = np.load(cache_path)
    
    Xf = data['Xf'].astype(np.float32)
    Xr = data['Xr'].astype(np.float32)
    y = data['y'].astype(np.float32)
    ids = data['ids']
    
    print(f"✓ Loaded {len(y)} samples")
    print(f"  Xf shape: {Xf.shape}")
    print(f"  Xr shape: {Xr.shape}")
    
    # Determine embedding dimension from data shape
    input_dim = Xf.shape[1]
    if input_dim == 3840:
        embed_dim = 768
        print(f"  Detected: ProSST embeddings (768-dim)")
    elif input_dim == 6400:
        embed_dim = 1280
        print(f"  Detected: ESM2 embeddings (1280-dim)")
    else:
        raise ValueError(f"Unexpected input dimension: {input_dim}. Expected 3840 or 6400.")
    
    return Xf, Xr, y, ids, input_dim, embed_dim


# ============================================================================
# Train/Test Split Recreation
# ============================================================================

def recreate_split(n_samples, train_ratio, random_seed):
    """
    Recreate the exact train/test split used during training.
    
    Args:
        n_samples: Total number of samples
        train_ratio: Ratio of training data (0.0 to 1.0)
        random_seed: Random seed for reproducibility
    
    Returns:
        train_idx, test_idx: Arrays of indices
    """
    if train_ratio == 0.0:
        # No training, all data is test
        return np.array([]), np.arange(n_samples)
    
    indices = np.arange(n_samples)
    train_idx, test_idx = train_test_split(
        indices,
        test_size=1.0 - train_ratio,
        random_state=random_seed,
        shuffle=True
    )
    
    return train_idx, test_idx


# ============================================================================
# Model Discovery and Loading
# ============================================================================

def find_model_files(model_dir):
    """Find model files in directory. Tries multiple naming patterns."""
    
    # Try specific pattern first
    pattern1 = os.path.join(model_dir, 'cross_attn_model_*pct.pth')
    model_files = glob.glob(pattern1)
    
    # If not found, try more general patterns
    if not model_files:
        print(f"No cross_attn_model_*pct.pth files found, trying alternative patterns...")
        
        # Try esm2_model pattern
        pattern2 = os.path.join(model_dir, 'esm2_model_*pct.pth')
        model_files = glob.glob(pattern2)
        
        # Try any *_model_*pct.pth pattern
        if not model_files:
            pattern3 = os.path.join(model_dir, '*_model_*pct.pth')
            model_files = glob.glob(pattern3)
        
        # Try any *pct.pth pattern as last resort
        if not model_files:
            pattern4 = os.path.join(model_dir, '*pct.pth')
            model_files = glob.glob(pattern4)
    
    if not model_files:
        raise FileNotFoundError(f"No *pct.pth model files found in {model_dir}")
    
    print(f"Found {len(model_files)} model files in {model_dir}")
    
    # Extract percentages and sort
    model_dict = {}
    for filepath in model_files:
        filename = os.path.basename(filepath)
        
        # Try to extract percentage from filename
        import re
        match = re.search(r'(\d+)pct\.pth$', filename)
        
        if match:
            pct = int(match.group(1))
            model_dict[pct] = filepath
            print(f"  {pct}%: {filename}")
        else:
            print(f"Warning: Could not parse percentage from {filename}")
            continue
    
    if not model_dict:
        raise ValueError(f"Found .pth files but could not parse percentages from filenames in {model_dir}")
    
    return model_dict


def load_model(model_path, input_dim, embed_dim, device):
    """Load model with appropriate architecture."""
    # Initialize model with correct architecture
    model = FullModelCrossAttention(
        input_dim=input_dim,
        reduced_dim=512,
        hidden_dim=768,
        dropout=0.35,
        num_hidden=3,
        embed_dim=embed_dim
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    return model


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_model(model, Xf, Xr, y, device, batch_size=32):
    """Evaluate model and return predictions and metrics."""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for i in range(0, len(Xf), batch_size):
            batch_xf = torch.tensor(Xf[i:i+batch_size], dtype=torch.float32).to(device)
            batch_xr = torch.tensor(Xr[i:i+batch_size], dtype=torch.float32).to(device)
            
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


# ============================================================================
# Plotting
# ============================================================================

def plot_metrics_comparison(prosst_results, esm2_results, output_dir):
    """Plot comparison of metrics across training percentages."""
    percentages = sorted(set(prosst_results.keys()) | set(esm2_results.keys()))
    
    # Extract metrics
    prosst_pearson = [prosst_results.get(pct, {}).get('metrics', {}).get('pearson_r', np.nan) for pct in percentages]
    prosst_spearman = [prosst_results.get(pct, {}).get('metrics', {}).get('spearman_r', np.nan) for pct in percentages]
    prosst_rmse = [prosst_results.get(pct, {}).get('metrics', {}).get('rmse', np.nan) for pct in percentages]
    
    esm2_pearson = [esm2_results.get(pct, {}).get('metrics', {}).get('pearson_r', np.nan) for pct in percentages]
    esm2_spearman = [esm2_results.get(pct, {}).get('metrics', {}).get('spearman_r', np.nan) for pct in percentages]
    esm2_rmse = [esm2_results.get(pct, {}).get('metrics', {}).get('rmse', np.nan) for pct in percentages]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Pearson correlation
    axes[0].plot(percentages, prosst_pearson, 'o-', linewidth=2, markersize=8, 
                 color='blue', label='ProSST')
    axes[0].plot(percentages, esm2_pearson, 's-', linewidth=2, markersize=8, 
                 color='green', label='ESM2')
    axes[0].set_xlabel('Training Set Size (%)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Pearson Correlation (r)', fontsize=12, fontweight='bold')
    axes[0].set_title('Pearson Correlation vs Training Size', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=11)
    axes[0].set_xticks(percentages)
    
    # Spearman correlation
    axes[1].plot(percentages, prosst_spearman, 'o-', linewidth=2, markersize=8, 
                 color='blue', label='ProSST')
    axes[1].plot(percentages, esm2_spearman, 's-', linewidth=2, markersize=8, 
                 color='green', label='ESM2')
    axes[1].set_xlabel('Training Set Size (%)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Spearman Correlation (ρ)', fontsize=12, fontweight='bold')
    axes[1].set_title('Spearman Correlation vs Training Size', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=11)
    axes[1].set_xticks(percentages)
    
    # RMSE
    axes[2].plot(percentages, prosst_rmse, 'o-', linewidth=2, markersize=8, 
                 color='blue', label='ProSST')
    axes[2].plot(percentages, esm2_rmse, 's-', linewidth=2, markersize=8, 
                 color='green', label='ESM2')
    axes[2].set_xlabel('Training Set Size (%)', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('RMSE (kcal/mol)', fontsize=12, fontweight='bold')
    axes[2].set_title('RMSE vs Training Size', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=11)
    axes[2].set_xticks(percentages)
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'prosst_vs_esm2_metrics.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.savefig(plot_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"✓ Saved metrics comparison plot to {plot_path}")
    plt.close()


def plot_scatter_grid(prosst_results, esm2_results, output_dir):
    """Create grid of scatter plots for each percentage."""
    percentages = sorted(set(prosst_results.keys()) | set(esm2_results.keys()))
    n_pcts = len(percentages)
    
    fig, axes = plt.subplots(2, n_pcts, figsize=(6*n_pcts, 12))
    
    for idx, pct in enumerate(percentages):
        # ProSST scatter plot
        if pct in prosst_results:
            prosst_pred = prosst_results[pct]['predictions']
            prosst_true = prosst_results[pct]['true_values']
            prosst_metrics = prosst_results[pct]['metrics']
            
            ax = axes[0, idx]
            ax.scatter(prosst_true, prosst_pred, alpha=0.5, s=20, c='blue')
            
            min_val = min(min(prosst_true), min(prosst_pred))
            max_val = max(max(prosst_true), max(prosst_pred))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
            
            ax.set_xlabel('True ΔΔG (kcal/mol)', fontsize=10)
            ax.set_ylabel('Predicted ΔΔG (kcal/mol)', fontsize=10)
            ax.set_title(f'ProSST: {pct}% Training', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            legend_text = f"Pearson: {prosst_metrics['pearson_r']:.3f}\n"
            legend_text += f"Spearman: {prosst_metrics['spearman_r']:.3f}\n"
            legend_text += f"RMSE: {prosst_metrics['rmse']:.3f}"
            ax.text(0.05, 0.95, legend_text, transform=ax.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                    fontsize=8)
        
        # ESM2 scatter plot
        if pct in esm2_results:
            esm2_pred = esm2_results[pct]['predictions']
            esm2_true = esm2_results[pct]['true_values']
            esm2_metrics = esm2_results[pct]['metrics']
            
            ax = axes[1, idx]
            ax.scatter(esm2_true, esm2_pred, alpha=0.5, s=20, c='green')
            
            min_val = min(min(esm2_true), min(esm2_pred))
            max_val = max(max(esm2_true), max(esm2_pred))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
            
            ax.set_xlabel('True ΔΔG (kcal/mol)', fontsize=10)
            ax.set_ylabel('Predicted ΔΔG (kcal/mol)', fontsize=10)
            ax.set_title(f'ESM2: {pct}% Training', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            legend_text = f"Pearson: {esm2_metrics['pearson_r']:.3f}\n"
            legend_text += f"Spearman: {esm2_metrics['spearman_r']:.3f}\n"
            legend_text += f"RMSE: {esm2_metrics['rmse']:.3f}"
            ax.text(0.05, 0.95, legend_text, transform=ax.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                    fontsize=8)
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'prosst_vs_esm2_scatter_grid.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.savefig(plot_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"✓ Saved scatter grid to {plot_path}")
    plt.close()


# ============================================================================
# Main Evaluation
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ProSST and ESM2 models across training percentages"
    )
    
    # ProSST arguments
    parser.add_argument('--prosst_model_dir', type=str, required=True,
                       help='Directory containing ProSST models (cross_attn_model_*pct.pth files)')
    parser.add_argument('--prosst_cache_path', type=str, required=True,
                       help='Path to ProSST data cache (.npz file)')
    
    # ESM2 arguments
    parser.add_argument('--esm2_model_dir', type=str, required=True,
                       help='Directory containing ESM2 models (cross_attn_model_*pct.pth files)')
    parser.add_argument('--esm2_cache_path', type=str, required=True,
                       help='Path to ESM2 data cache (.npz file)')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Directory for output files (default: evaluation_results)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation (default: 32)')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducing train/test splits (default: 42)')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("CROSS-ATTENTION MODEL EVALUATION")
    print("ProSST vs ESM2 - Few-Shot Learning Comparison")
    print("="*70)
    print(f"Random seed: {args.random_seed} (for reproducing train/test splits)")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # ========================================================================
    # Load ProSST data and models
    # ========================================================================
    
    # Load data once
    prosst_Xf, prosst_Xr, prosst_y, prosst_ids, prosst_input_dim, prosst_embed_dim = load_data(
        args.prosst_cache_path, "ProSST"
    )
    
    # Find all ProSST models
    print(f"\n{'='*70}")
    print("FINDING PROSST MODELS")
    print(f"{'='*70}")
    prosst_models = find_model_files(args.prosst_model_dir)
    print(f"✓ Found {len(prosst_models)} ProSST models at percentages: {sorted(prosst_models.keys())}")
    
    # Evaluate all ProSST models
    prosst_results = {}
    
    for pct in sorted(prosst_models.keys()):
        print(f"\n{'='*70}")
        print(f"EVALUATING PROSST MODEL: {pct}% Training")
        print(f"{'='*70}")
        
        # Recreate train/test split
        train_ratio = pct / 100.0
        train_idx, test_idx = recreate_split(len(prosst_y), train_ratio, args.random_seed)
        
        print(f"Recreated split: {len(train_idx)} train, {len(test_idx)} test")
        
        # Extract test data
        Xf_test = prosst_Xf[test_idx]
        Xr_test = prosst_Xr[test_idx]
        y_test = prosst_y[test_idx]
        
        # Load and evaluate model
        model_path = prosst_models[pct]
        print(f"Loading: {os.path.basename(model_path)}")
        
        model = load_model(model_path, prosst_input_dim, prosst_embed_dim, device)
        predictions, metrics = evaluate_model(model, Xf_test, Xr_test, y_test, device, args.batch_size)
        
        print(f"  Pearson:  {metrics['pearson_r']:.4f}")
        print(f"  Spearman: {metrics['spearman_r']:.4f}")
        print(f"  RMSE:     {metrics['rmse']:.4f}")
        print(f"  n_test:   {metrics['n_samples']}")
        
        prosst_results[pct] = {
            'predictions': predictions,
            'true_values': y_test,
            'metrics': metrics,
            'n_train': len(train_idx),
            'n_test': len(test_idx)
        }
    
    # ========================================================================
    # Load ESM2 data and models
    # ========================================================================
    
    # Load data once
    esm2_Xf, esm2_Xr, esm2_y, esm2_ids, esm2_input_dim, esm2_embed_dim = load_data(
        args.esm2_cache_path, "ESM2"
    )
    
    # Find all ESM2 models
    print(f"\n{'='*70}")
    print("FINDING ESM2 MODELS")
    print(f"{'='*70}")
    esm2_models = find_model_files(args.esm2_model_dir)
    print(f"✓ Found {len(esm2_models)} ESM2 models at percentages: {sorted(esm2_models.keys())}")
    
    # Evaluate all ESM2 models
    esm2_results = {}
    
    for pct in sorted(esm2_models.keys()):
        print(f"\n{'='*70}")
        print(f"EVALUATING ESM2 MODEL: {pct}% Training")
        print(f"{'='*70}")
        
        # Recreate train/test split
        train_ratio = pct / 100.0
        train_idx, test_idx = recreate_split(len(esm2_y), train_ratio, args.random_seed)
        
        print(f"Recreated split: {len(train_idx)} train, {len(test_idx)} test")
        
        # Extract test data
        Xf_test = esm2_Xf[test_idx]
        Xr_test = esm2_Xr[test_idx]
        y_test = esm2_y[test_idx]
        
        # Load and evaluate model
        model_path = esm2_models[pct]
        print(f"Loading: {os.path.basename(model_path)}")
        
        model = load_model(model_path, esm2_input_dim, esm2_embed_dim, device)
        predictions, metrics = evaluate_model(model, Xf_test, Xr_test, y_test, device, args.batch_size)
        
        print(f"  Pearson:  {metrics['pearson_r']:.4f}")
        print(f"  Spearman: {metrics['spearman_r']:.4f}")
        print(f"  RMSE:     {metrics['rmse']:.4f}")
        print(f"  n_test:   {metrics['n_samples']}")
        
        esm2_results[pct] = {
            'predictions': predictions,
            'true_values': y_test,
            'metrics': metrics,
            'n_train': len(train_idx),
            'n_test': len(test_idx)
        }
    
    # ========================================================================
    # Save results
    # ========================================================================
    
    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print(f"{'='*70}")
    
    # Save summary CSV
    summary_data = []
    all_pcts = sorted(set(prosst_results.keys()) | set(esm2_results.keys()))
    
    for pct in all_pcts:
        row = {'training_pct': pct}
        
        if pct in prosst_results:
            pm = prosst_results[pct]['metrics']
            row['prosst_n_train'] = prosst_results[pct]['n_train']
            row['prosst_n_test'] = prosst_results[pct]['n_test']
            row['prosst_pearson'] = pm['pearson_r']
            row['prosst_spearman'] = pm['spearman_r']
            row['prosst_rmse'] = pm['rmse']
            row['prosst_mae'] = pm['mae']
            row['prosst_r2'] = pm['r2']
        
        if pct in esm2_results:
            em = esm2_results[pct]['metrics']
            row['esm2_n_train'] = esm2_results[pct]['n_train']
            row['esm2_n_test'] = esm2_results[pct]['n_test']
            row['esm2_pearson'] = em['pearson_r']
            row['esm2_spearman'] = em['spearman_r']
            row['esm2_rmse'] = em['rmse']
            row['esm2_mae'] = em['mae']
            row['esm2_r2'] = em['r2']
        
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(args.output_dir, 'prosst_vs_esm2_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"✓ Saved summary to {summary_path}")
    
    # Save complete metrics as JSON
    all_metrics = {
        'prosst': {f'{pct}pct': {
            'metrics': prosst_results[pct]['metrics'],
            'n_train': prosst_results[pct]['n_train'],
            'n_test': prosst_results[pct]['n_test']
        } for pct in prosst_results},
        'esm2': {f'{pct}pct': {
            'metrics': esm2_results[pct]['metrics'],
            'n_train': esm2_results[pct]['n_train'],
            'n_test': esm2_results[pct]['n_test']
        } for pct in esm2_results}
    }
    
    json_path = os.path.join(args.output_dir, 'all_metrics.json')
    with open(json_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"✓ Saved all metrics to {json_path}")
    
    # ========================================================================
    # Create plots
    # ========================================================================
    
    print(f"\n{'='*70}")
    print("CREATING PLOTS")
    print(f"{'='*70}")
    
    plot_metrics_comparison(prosst_results, esm2_results, args.output_dir)
    plot_scatter_grid(prosst_results, esm2_results, args.output_dir)
    
    # ========================================================================
    # Print summary
    # ========================================================================
    
    print(f"\n{'='*70}")
    print("EVALUATION COMPLETE - SUMMARY TABLE")
    print(f"{'='*70}")
    print(summary_df.to_string(index=False))
    
    print(f"\n{'='*70}")
    print("FILES SAVED")
    print(f"{'='*70}")
    print(f"Results saved to: {args.output_dir}")
    print("  - prosst_vs_esm2_summary.csv")
    print("  - prosst_vs_esm2_metrics.png/pdf")
    print("  - prosst_vs_esm2_scatter_grid.png/pdf")
    print("  - all_metrics.json")
    print()


if __name__ == '__main__':
    main()