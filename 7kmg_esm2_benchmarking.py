#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Few-shot learning comparison: Cross-attention model vs Simple model.
Fine-tune both models on varying percentages: 0% to 80% at 10% intervals.
Tests on the remaining data and compares performance.

Command-line version with argparse
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import json
import copy
import argparse


# ============================================================================
# Dataset Class (Shared)
# ============================================================================

class DdgDataset(Dataset):
    def __init__(self, xf, xr, y):
        self.xf = torch.tensor(xf, dtype=torch.float32)
        self.xr = torch.tensor(xr, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.xf[idx], self.xr[idx], self.y[idx]


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
# Training and Evaluation Functions
# ============================================================================

def fine_tune_cross_attention(model, train_loader, Xf_test, Xr_test, y_test, device, 
                              num_epochs=200, learning_rate=1e-5, patience=20):
    """Fine-tune cross-attention model with early stopping based on training loss."""
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    best_train_loss = float('inf')
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        n_batches = 0
        
        for xf, xr, y in train_loader:
            xf, xr, y = xf.to(device), xr.to(device), y.to(device)
            
            optimizer.zero_grad()
            pred_ddg, pred_ilddt = model(xf, xr)
            
            loss = criterion(pred_ddg, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        avg_train_loss = total_loss / n_batches
        
        # Early stopping based on training loss
        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            best_epoch = epoch + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        # Stop if no improvement for 'patience' epochs
        if epochs_without_improvement >= patience:
            print(f"  Early stopping at epoch {epoch + 1}")
            break
    
    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    print(f"  ✓ Fine-tuning complete!")
    print(f"    Best train loss: {best_train_loss:.6f} at epoch {best_epoch}")
    
    model.eval()
    return model


def fine_tune_simple(model, train_loader, Xf_test, Xr_test, y_test, device,
                     num_epochs=200, learning_rate=1e-5, patience=20):
    """Fine-tune simple model with early stopping based on training loss."""
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    best_train_loss = float('inf')
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        total_train_loss = 0
        n_batches = 0
        
        for xf, xr, y in train_loader:
            xf, xr, y = xf.to(device), xr.to(device), y.to(device)
            
            optimizer.zero_grad()
            pred_ddg = model(xf, xr).squeeze()
            
            loss = criterion(pred_ddg, y)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            n_batches += 1
        
        avg_train_loss = total_train_loss / n_batches
        
        # Early stopping based on training loss
        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            best_epoch = epoch + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        # Stop if no improvement for 'patience' epochs
        if epochs_without_improvement >= patience:
            print(f"  Early stopping at epoch {epoch + 1}")
            break
    
    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    print(f"  ✓ Fine-tuning complete!")
    print(f"    Best train loss: {best_train_loss:.6f} at epoch {best_epoch}")
    
    model.eval()
    return model


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

def plot_comparison_metrics(cross_attn_results, simple_results, output_dir):
    """
    Create comparison plots showing how both models perform across training sizes.
    """
    train_percentages = sorted(cross_attn_results.keys())
    
    # Extract metrics for both models
    ca_pearson = [cross_attn_results[pct]['metrics']['pearson_r'] for pct in train_percentages]
    ca_spearman = [cross_attn_results[pct]['metrics']['spearman_r'] for pct in train_percentages]
    ca_rmse = [cross_attn_results[pct]['metrics']['rmse'] for pct in train_percentages]
    
    simple_pearson = [simple_results[pct]['metrics']['pearson_r'] for pct in train_percentages]
    simple_spearman = [simple_results[pct]['metrics']['spearman_r'] for pct in train_percentages]
    simple_rmse = [simple_results[pct]['metrics']['rmse'] for pct in train_percentages]
    
    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Pearson correlation plot
    axes[0].plot(train_percentages, ca_pearson, 'o-', linewidth=2, markersize=8, 
                 color='blue', label='Cross-Attention')
    axes[0].plot(train_percentages, simple_pearson, 's--', linewidth=2, markersize=8, 
                 color='red', label='Simple')
    axes[0].set_xlabel('Training Set Size (%)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Pearson Correlation (r)', fontsize=12, fontweight='bold')
    axes[0].set_title('Pearson Correlation vs Training Size', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(train_percentages)
    axes[0].legend(fontsize=10)
    
    # Spearman correlation plot
    axes[1].plot(train_percentages, ca_spearman, 'o-', linewidth=2, markersize=8, 
                 color='blue', label='Cross-Attention')
    axes[1].plot(train_percentages, simple_spearman, 's--', linewidth=2, markersize=8, 
                 color='red', label='Simple')
    axes[1].set_xlabel('Training Set Size (%)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Spearman Correlation (ρ)', fontsize=12, fontweight='bold')
    axes[1].set_title('Spearman Correlation vs Training Size', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(train_percentages)
    axes[1].legend(fontsize=10)
    
    # RMSE plot
    axes[2].plot(train_percentages, ca_rmse, 'o-', linewidth=2, markersize=8, 
                 color='blue', label='Cross-Attention')
    axes[2].plot(train_percentages, simple_rmse, 's--', linewidth=2, markersize=8, 
                 color='red', label='Simple')
    axes[2].set_xlabel('Training Set Size (%)', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('RMSE (kcal/mol)', fontsize=12, fontweight='bold')
    axes[2].set_title('RMSE vs Training Size', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xticks(train_percentages)
    axes[2].legend(fontsize=10)
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'model_comparison_metrics.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.savefig(plot_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"\n✓ Saved comparison metrics plot to {plot_path}")
    plt.close()


def plot_scatter_comparison(cross_attn_results, simple_results, output_dir):
    """
    Create scatter plots comparing both models at each training size.
    """
    train_percentages = sorted(cross_attn_results.keys())
    n_plots = len(train_percentages)
    
    # Cross-attention scatter plots
    fig, axes = plt.subplots(2, n_plots, figsize=(6*n_plots, 10))
    
    for idx, pct in enumerate(train_percentages):
        # Cross-attention
        ca_result = cross_attn_results[pct]
        ca_pred = ca_result['predictions']
        ca_true = ca_result['true_values']
        ca_metrics = ca_result['metrics']
        
        ax = axes[0, idx]
        ax.scatter(ca_true, ca_pred, alpha=0.5, s=20, c='blue')
        
        min_val = min(min(ca_true), min(ca_pred))
        max_val = max(max(ca_true), max(ca_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
        
        ax.set_xlabel('True ΔΔG (kcal/mol)', fontsize=11)
        ax.set_ylabel('Predicted ΔΔG (kcal/mol)', fontsize=11)
        ax.set_title(f'Cross-Attention: {pct}% Training', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        legend_text = f"Pearson: {ca_metrics['pearson_r']:.3f}\n"
        legend_text += f"Spearman: {ca_metrics['spearman_r']:.3f}\n"
        legend_text += f"RMSE: {ca_metrics['rmse']:.3f}"
        ax.text(0.05, 0.95, legend_text, transform=ax.transAxes,
                verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                fontsize=9)
        
        # Simple model
        simple_result = simple_results[pct]
        simple_pred = simple_result['predictions']
        simple_true = simple_result['true_values']
        simple_metrics = simple_result['metrics']
        
        ax = axes[1, idx]
        ax.scatter(simple_true, simple_pred, alpha=0.5, s=20, c='red')
        
        min_val = min(min(simple_true), min(simple_pred))
        max_val = max(max(simple_true), max(simple_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
        
        ax.set_xlabel('True ΔΔG (kcal/mol)', fontsize=11)
        ax.set_ylabel('Predicted ΔΔG (kcal/mol)', fontsize=11)
        ax.set_title(f'Simple: {pct}% Training', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        legend_text = f"Pearson: {simple_metrics['pearson_r']:.3f}\n"
        legend_text += f"Spearman: {simple_metrics['spearman_r']:.3f}\n"
        legend_text += f"RMSE: {simple_metrics['rmse']:.3f}"
        ax.text(0.05, 0.95, legend_text, transform=ax.transAxes,
                verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='salmon', alpha=0.8),
                fontsize=9)
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'scatter_comparison_grid.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.savefig(plot_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"✓ Saved scatter comparison grid to {plot_path}")
    plt.close()


# ============================================================================
# Argument Parser
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Few-shot learning comparison: Train and compare Cross-Attention vs Simple models'
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
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for results and plots')
    
    # Optional arguments
    parser.add_argument('--train_ratios', type=float, nargs='+',
                       default=[0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
                       help='Training set ratios (default: 0.0 to 0.8 at 0.1 intervals)')
    parser.add_argument('--batch_size_ca', type=int, default=16,
                       help='Batch size for cross-attention model (default: 16)')
    parser.add_argument('--batch_size_simple', type=int, default=8,
                       help='Batch size for simple model (default: 8)')
    parser.add_argument('--num_epochs', type=int, default=200,
                       help='Max training epochs (default: 200)')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                       help='Learning rate (default: 1e-5)')
    parser.add_argument('--patience', type=int, default=20,
                       help='Early stopping patience (default: 20)')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    return parser.parse_args()


# ============================================================================
# Main Function
# ============================================================================

def main():
    args = parse_args()
    
    print("\n" + "="*60)
    print("FEW-SHOT LEARNING MODEL COMPARISON")
    print("Cross-Attention vs Simple Model")
    print("="*60)
    
    print("\nConfiguration:")
    print(f"  Cross-Attn model: {args.cross_attn_base_model}")
    print(f"  Simple model:     {args.simple_base_model}")
    print(f"  Cache:            {args.cache_path}")
    print(f"  NPZ dir:          {args.npz_dir}")
    print(f"  Output dir:       {args.output_dir}")
    print(f"  Train ratios:     {[f'{r*100:.0f}%' for r in args.train_ratios]}")
    print(f"  Epochs:           {args.num_epochs}")
    print(f"  Learning rate:    {args.learning_rate}")
    print(f"  Patience:         {args.patience}")
    print(f"  Random seed:      {args.random_seed}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Device: {device}")
    
    # Set random seed
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.random_seed)
    
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
    
    # Load base models
    print(f"\n{'='*60}")
    print("LOADING BASE MODELS")
    print(f"{'='*60}")
    
    # Cross-attention model
    ca_base_model = FullModelCrossAttention().to(device)
    ca_checkpoint = torch.load(args.cross_attn_base_model, map_location=device, weights_only=False)
    if isinstance(ca_checkpoint, dict) and 'model_state_dict' in ca_checkpoint:
        ca_base_model.load_state_dict(ca_checkpoint['model_state_dict'])
    else:
        ca_base_model.load_state_dict(ca_checkpoint)
    ca_base_state = ca_base_model.state_dict()
    print("✓ Cross-Attention base model loaded")
    
    # Simple model
    simple_base_model = DDGPredictorSimple(
        input_dim=768,
        hidden_dim=1024,
        dropout_rate=0.2,
        num_hidden=4
    ).to(device)
    simple_base_state = torch.load(args.simple_base_model, map_location=device)
    simple_base_model.load_state_dict(simple_base_state)
    print("✓ Simple base model loaded")
    
    # Store results
    cross_attn_results = {}
    simple_results = {}
    
    # Iterate over training ratios
    for train_ratio in args.train_ratios:
        print(f"\n{'='*60}")
        print(f"TRAINING WITH {train_ratio*100:.0f}% OF DATA")
        print(f"{'='*60}")
        
        if train_ratio == 0.0:
            # No fine-tuning, just evaluate base models
            print("\nNo fine-tuning (0% training data) - using base models")
            
            # Cross-Attention
            ca_model = FullModelCrossAttention().to(device)
            ca_model.load_state_dict(copy.deepcopy(ca_base_state))
            Xf_test_ca, Xr_test_ca, y_test_ca, ids_test = Xf_ca, Xr_ca, y_ca, ids_ca
            
            # Simple
            simple_model = DDGPredictorSimple(
                input_dim=768,
                hidden_dim=1024,
                dropout_rate=0.2,
                num_hidden=4
            ).to(device)
            simple_model.load_state_dict(copy.deepcopy(simple_base_state))
            Xf_test_simple, Xr_test_simple, y_test_simple = Xf_simple, Xr_simple, y_simple
            
        else:
            # Split data (use same indices for both models)
            indices = np.arange(len(y_ca))
            train_idx, test_idx = train_test_split(
                indices,
                test_size=1.0 - train_ratio,
                random_state=args.random_seed,
                shuffle=True
            )
            
            # Cross-Attention data
            Xf_train_ca = Xf_ca[train_idx]
            Xr_train_ca = Xr_ca[train_idx]
            y_train_ca = y_ca[train_idx]
            Xf_test_ca = Xf_ca[test_idx]
            Xr_test_ca = Xr_ca[test_idx]
            y_test_ca = y_ca[test_idx]
            
            # Simple data
            Xf_train_simple = Xf_simple[train_idx]
            Xr_train_simple = Xr_simple[train_idx]
            y_train_simple = y_simple[train_idx]
            Xf_test_simple = Xf_simple[test_idx]
            Xr_test_simple = Xr_simple[test_idx]
            y_test_simple = y_simple[test_idx]
            
            ids_test = ids_ca[test_idx]
            
            print(f"Training set:   {len(train_idx)} samples ({len(train_idx)/len(y_ca)*100:.1f}%)")
            print(f"Test set:       {len(test_idx)} samples ({len(test_idx)/len(y_ca)*100:.1f}%)")
            
            # Fine-tune Cross-Attention model
            print("\n--- Training Cross-Attention Model ---")
            ca_model = FullModelCrossAttention().to(device)
            ca_model.load_state_dict(copy.deepcopy(ca_base_state))
            
            train_dataset_ca = DdgDataset(Xf_train_ca, Xr_train_ca, y_train_ca)
            train_loader_ca = DataLoader(train_dataset_ca, batch_size=args.batch_size_ca, shuffle=True)
            
            ca_model = fine_tune_cross_attention(
                ca_model, train_loader_ca, Xf_test_ca, Xr_test_ca, y_test_ca,
                device, args.num_epochs, args.learning_rate, args.patience
            )
            
            # Fine-tune Simple model
            print("\n--- Training Simple Model ---")
            simple_model = DDGPredictorSimple(
                input_dim=768,
                hidden_dim=1024,
                dropout_rate=0.2,
                num_hidden=4
            ).to(device)
            simple_model.load_state_dict(copy.deepcopy(simple_base_state))
            
            train_dataset_simple = DdgDataset(Xf_train_simple, Xr_train_simple, y_train_simple)
            train_loader_simple = DataLoader(train_dataset_simple, batch_size=args.batch_size_simple, shuffle=True)
            
            simple_model = fine_tune_simple(
                simple_model, train_loader_simple, Xf_test_simple, Xr_test_simple, y_test_simple,
                device, args.num_epochs, args.learning_rate, args.patience
            )
        
        # Evaluate both models
        print(f"\n--- Evaluating Models on Test Set ---")
        
        ca_predictions, ca_metrics = evaluate_cross_attention(
            ca_model, Xf_test_ca, Xr_test_ca, y_test_ca, device
        )
        
        simple_predictions, simple_metrics = evaluate_simple(
            simple_model, Xf_test_simple, Xr_test_simple, y_test_simple, device
        )
        
        print(f"\nCross-Attention performance:")
        print(f"  Pearson r:  {ca_metrics['pearson_r']:.4f}")
        print(f"  Spearman ρ: {ca_metrics['spearman_r']:.4f}")
        print(f"  RMSE:       {ca_metrics['rmse']:.4f}")
        
        print(f"\nSimple model performance:")
        print(f"  Pearson r:  {simple_metrics['pearson_r']:.4f}")
        print(f"  Spearman ρ: {simple_metrics['spearman_r']:.4f}")
        print(f"  RMSE:       {simple_metrics['rmse']:.4f}")
        
        # Store results
        pct = train_ratio * 100
        
        cross_attn_results[pct] = {
            'predictions': ca_predictions,
            'true_values': y_test_ca,
            'ids': ids_test,
            'metrics': ca_metrics,
            'train_ratio': train_ratio,
            'n_train': 0 if train_ratio == 0.0 else len(train_idx),
            'n_test': len(y_test_ca)
        }
        
        simple_results[pct] = {
            'predictions': simple_predictions,
            'true_values': y_test_simple,
            'ids': ids_test,
            'metrics': simple_metrics,
            'train_ratio': train_ratio,
            'n_train': 0 if train_ratio == 0.0 else len(train_idx),
            'n_test': len(y_test_simple)
        }
        
        # Save individual models
        if train_ratio > 0.0:
            ca_model_path = os.path.join(args.output_dir, f'cross_attn_model_{int(pct)}pct.pth')
            torch.save(ca_model.state_dict(), ca_model_path)
            
            simple_model_path = os.path.join(args.output_dir, f'simple_model_{int(pct)}pct.pth')
            torch.save(simple_model.state_dict(), simple_model_path)
    
    # Create comparison plots
    print(f"\n{'='*60}")
    print("CREATING COMPARISON PLOTS")
    print(f"{'='*60}")
    
    plot_comparison_metrics(cross_attn_results, simple_results, args.output_dir)
    plot_scatter_comparison(cross_attn_results, simple_results, args.output_dir)
    
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
            'n_train': ca_result['n_train'],
            'n_test': ca_result['n_test'],
            'ca_pearson_r': ca_result['metrics']['pearson_r'],
            'ca_spearman_r': ca_result['metrics']['spearman_r'],
            'ca_rmse': ca_result['metrics']['rmse'],
            'simple_pearson_r': simple_result['metrics']['pearson_r'],
            'simple_spearman_r': simple_result['metrics']['spearman_r'],
            'simple_rmse': simple_result['metrics']['rmse']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(args.output_dir, 'comparison_summary.csv')
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
        
        pred_path = os.path.join(args.output_dir, f'predictions_{int(pct)}pct.csv')
        df.to_csv(pred_path, index=False)
        print(f"✓ Saved predictions for {int(pct)}% to {pred_path}")
    
    # Save complete results as JSON
    json_results = {
        'cross_attention': {},
        'simple': {}
    }
    
    for pct in sorted(cross_attn_results.keys()):
        ca_result = cross_attn_results[pct]
        simple_result = simple_results[pct]
        
        json_results['cross_attention'][f'{int(pct)}pct'] = {
            'train_ratio': ca_result['train_ratio'],
            'n_train': ca_result['n_train'],
            'n_test': ca_result['n_test'],
            'metrics': ca_result['metrics']
        }
        
        json_results['simple'][f'{int(pct)}pct'] = {
            'train_ratio': simple_result['train_ratio'],
            'n_train': simple_result['n_train'],
            'n_test': simple_result['n_test'],
            'metrics': simple_result['metrics']
        }
    
    json_path = os.path.join(args.output_dir, 'all_results.json')
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"✓ Saved all results to {json_path}")
    
    # Print summary table
    print(f"\n{'='*60}")
    print("SUMMARY TABLE")
    print(f"{'='*60}")
    print(summary_df.to_string(index=False))
    
    print("\n" + "="*60)
    print("✓ ANALYSIS COMPLETE!")
    print("="*60)
    print(f"\nResults saved to: {args.output_dir}")
    print("  - comparison_summary.csv (overview)")
    print("  - model_comparison_metrics.png/pdf (Pearson, Spearman, RMSE)")
    print("  - scatter_comparison_grid.png/pdf (scatter plots)")
    print("  - predictions_*pct.csv (detailed predictions)")
    print("  - cross_attn_model_*pct.pth (fine-tuned models)")
    print("  - simple_model_*pct.pth (fine-tuned models)")
    print("  - all_results.json (complete metrics)")
    print("\n")


if __name__ == '__main__':
    main()