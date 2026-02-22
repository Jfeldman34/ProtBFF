#!/usr/bin/env python3
"""
evaluate_saved_models.py

Load saved models from cross-validation and evaluate on test sets to verify results.

Usage:
    python /n/netscratch/shakhnovich_lab/Lab/jonathanfeldman/test_protbff/ProtBFF/evaluate_saved_models.py \
    --cache model_benchmarking/score_caches/skempi_score_cache.npz \
    --model_save_dir model_benchmarking/skempi_prosst_output/ \
    --folds_dir *** \
    --output_dir "."
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr
import json
from tqdm import tqdm


# ============================================================================
# Model Architecture
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
            projected.append(proj(embeddings[:, i, :]))
        projected = torch.stack(projected, dim=1)
        attended, _ = self.cross_attention(projected, projected, projected)
        attended = self.attention_norm(attended + projected)
        attended_t = attended.transpose(1, 2)
        attn_logits = self.final_attention(attended_t)
        attn_weights = F.softmax(attn_logits, dim=-1)
        return (attn_weights * attended_t).sum(dim=-1)


class DDGPredictor(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=768, dropout_rate=0.35, num_hidden=3):
        super().__init__()
        layers = []
        in_dim = input_dim
        for i in range(num_hidden):
            out_dim = hidden_dim if i == 0 else hidden_dim // (2 ** i)
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout_rate)]
            in_dim = out_dim
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Dropout(dropout_rate * 0.8),
            nn.Linear(in_dim, 1)
        )
    
    def forward(self, xf, xr):
        return (self.head(self.mlp(xf)) - self.head(self.mlp(xr))) / 2


class FullModelCrossAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.pooling = CrossEmbeddingAttention()
        self.ddg_predictor = DDGPredictor()
        self.ilddt_predictor = DDGPredictor()
    
    def forward(self, xf, xr):
        xf_pooled = self.pooling(xf)
        xr_pooled = self.pooling(xr)
        return (
            self.ddg_predictor(xf_pooled, xr_pooled).squeeze(-1),
            self.ilddt_predictor(xf_pooled, xr_pooled).squeeze(-1)
        )


# ============================================================================
# Dataset Class
# ============================================================================

class DdgDataset:
    """Dataset for ddG and ilddt multi-task learning."""
    
    def __init__(self, xf, xr, y, ilddt):
        self.xf = np.asarray(xf, dtype=np.float32)
        self.xr = np.asarray(xr, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32)
        self.ilddt = np.asarray(ilddt, dtype=np.float32)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.xf[idx], self.xr[idx], self.y[idx], self.ilddt[idx]


# ============================================================================
# Data Loading
# ============================================================================

def load_preprocessed_data(cache_path):
    """Load preprocessed data from cache file."""
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cache file not found: {cache_path}")
    
    print(f"Loading preprocessed data from: {cache_path}")
    data = np.load(cache_path, allow_pickle=True)
    
    # Check required keys
    required_keys = ['Xf', 'Xr', 'y', 'ilddt', 'ids']
    missing_keys = [k for k in required_keys if k not in data]
    if missing_keys:
        raise ValueError(f"Cache missing required keys: {missing_keys}")
    
    Xf = data['Xf']
    Xr = data['Xr']
    y = data['y']
    ilddt = data['ilddt']
    complex_names = data['ids']
    
    print(f"Loaded {len(y)} samples")
    
    return Xf, Xr, y, ilddt, complex_names


# ============================================================================
# Model Evaluation
# ============================================================================

def evaluate_model(model, test_loader, device):
    """
    Evaluate model on test set.
    
    Returns:
        predictions_ddg, true_ddg, predictions_ilddt, true_ilddt
    """
    model.eval()
    
    predictions_ddg = []
    true_ddg = []
    predictions_ilddt = []
    true_ilddt = []
    
    with torch.no_grad():
        for batch in test_loader:
            xf, xr, y, ilddt = batch
            xf, xr = xf.to(device), xr.to(device)
            
            pred_ddg, pred_ilddt = model(xf, xr)
            
            # Convert to numpy
            pred_ddg_np = pred_ddg.cpu().numpy()
            pred_ilddt_np = pred_ilddt.cpu().numpy()
            
            predictions_ddg.extend(pred_ddg_np if pred_ddg_np.ndim > 0 else [pred_ddg_np])
            true_ddg.extend(y.numpy() if hasattr(y, 'numpy') else [y])
            predictions_ilddt.extend(pred_ilddt_np if pred_ilddt_np.ndim > 0 else [pred_ilddt_np])
            true_ilddt.extend(ilddt.numpy() if hasattr(ilddt, 'numpy') else [ilddt])
    
    return (np.array(predictions_ddg), np.array(true_ddg),
            np.array(predictions_ilddt), np.array(true_ilddt))


def calculate_metrics(pred_ddg, true_ddg, pred_ilddt, true_ilddt):
    """Calculate correlation metrics."""
    pear_ddg = pearsonr(true_ddg, pred_ddg)[0]
    spear_ddg = spearmanr(true_ddg, pred_ddg)[0]
    pear_ilddt = pearsonr(true_ilddt, pred_ilddt)[0]
    spear_ilddt = spearmanr(true_ilddt, pred_ilddt)[0]
    
    return pear_ddg, spear_ddg, pear_ilddt, spear_ilddt


def load_test_indices_from_fold_dir(folds_dir, ids, n_folds=10):
    """
    Load test indices for each fold from predefined fold directories.
    ids should be the short PDB codes (after underscore), matching what was used during training.
    """
    fold_test_indices = []

    for fold_num in range(1, n_folds + 1):
        fold_dir = os.path.join(folds_dir, f"fold_{fold_num}")
        test_file = os.path.join(fold_dir, "test_complex_ids.txt")

        if not os.path.exists(test_file):
            raise FileNotFoundError(f"Test file not found: {test_file}")

        with open(test_file, 'r') as f:
            test_ids_raw = [line.strip() for line in f if line.strip()]

        test_pdb_codes = set()
        for raw_id in test_ids_raw:
            if '_' in raw_id:
                test_pdb_codes.add(raw_id.split('_', 1)[1])
            else:
                test_pdb_codes.add(raw_id)

        test_indices = np.array([i for i, id_str in enumerate(ids) if id_str in test_pdb_codes])

        matched = set(ids[test_indices]) if len(test_indices) > 0 else set()
        unmatched = test_pdb_codes - matched
        print(f"Fold {fold_num}: {len(test_indices)} test samples matched")
        if unmatched:
            print(f"  Warning: {len(unmatched)} test PDB codes not found in cache")
            if len(unmatched) <= 5:
                print(f"  Unmatched: {list(unmatched)}")

        fold_test_indices.append(test_indices)

    return fold_test_indices


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate saved models from cross-validation training"
    )
    parser.add_argument('--cache', type=str, required=True)
    parser.add_argument('--model_save_dir', type=str, required=True)
    parser.add_argument('--folds_dir', type=str, required=True,
                       help='Directory containing fold_1/.../fold_10 subdirectories with test_complex_ids.txt')
    parser.add_argument('--output_dir', type=str, default='evaluation_results')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_folds', type=int, default=10)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    os.makedirs(args.output_dir, exist_ok=True)

    Xf, Xr, y, ilddt, complex_names = load_preprocessed_data(args.cache)

    # Extract short PDB codes for fold matching (part after underscore)
    ids = np.array([s.split('_', 1)[1] if '_' in str(s) else str(s) for s in complex_names])

    print(f"\nLoading test splits from: {args.folds_dir}")
    fold_test_indices = load_test_indices_from_fold_dir(args.folds_dir, ids, args.n_folds)

    print(f"\n{'='*70}")
    print("EVALUATING SAVED MODELS")
    print(f"{'='*70}\n")

    all_predictions_ddg = []
    all_true_values_ddg = []
    all_predictions_ilddt = []
    all_true_values_ilddt = []
    all_complex_names = []
    all_fold_indices = []
    fold_results = []

    for fold_idx in range(args.n_folds):
        print(f"\n{'='*70}")
        print(f"FOLD {fold_idx + 1}/{args.n_folds}")
        print(f"{'='*70}")

        checkpoint_path = os.path.join(args.model_save_dir, f'fold_{fold_idx}_checkpoint.pth')
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(args.model_save_dir, f'fold_{fold_idx}_full_model.pth')
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint not found, skipping fold {fold_idx + 1}")
            continue

        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Use fold dir indices instead of checkpoint test_idx
        test_idx = fold_test_indices[fold_idx]
        print(f"Test samples (from fold dir): {len(test_idx)}")

        if len(test_idx) == 0:
            print(f"No test samples found for fold {fold_idx + 1}, skipping")
            continue

        test_dataset = DdgDataset(Xf[test_idx], Xr[test_idx], y[test_idx], ilddt[test_idx])
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        model = FullModelCrossAttention().to(device)
        model.load_state_dict(checkpoint['model_state_dict'])

        pred_ddg, true_ddg, pred_ilddt, true_ilddt = evaluate_model(model, test_loader, device)
        pear_ddg, spear_ddg, pear_ilddt, spear_ilddt = calculate_metrics(
            pred_ddg, true_ddg, pred_ilddt, true_ilddt
        )

        print(f"  ddG    - Pearson: {pear_ddg:.4f}, Spearman: {spear_ddg:.4f}")
        print(f"  ilddt  - Pearson: {pear_ilddt:.4f}, Spearman: {spear_ilddt:.4f}")

        all_predictions_ddg.extend(pred_ddg)
        all_true_values_ddg.extend(true_ddg)
        all_predictions_ilddt.extend(pred_ilddt)
        all_true_values_ilddt.extend(true_ilddt)
        all_complex_names.extend(complex_names[test_idx])
        all_fold_indices.extend([fold_idx] * len(test_idx))

        fold_results.append({
            'fold': fold_idx + 1,
            'n_test': len(test_idx),
            'pearson_ddg': float(pear_ddg),
            'spearman_ddg': float(spear_ddg),
            'pearson_ilddt': float(pear_ilddt),
            'spearman_ilddt': float(spear_ilddt)
        })

    # rest of main() is unchanged from your original
    # Calculate overall metrics
    print(f"\n{'='*70}")
    print("OVERALL RESULTS")
    print(f"{'='*70}")
    
    overall_pear_ddg = pearsonr(all_true_values_ddg, all_predictions_ddg)[0]
    overall_spear_ddg = spearmanr(all_true_values_ddg, all_predictions_ddg)[0]
    overall_pear_ilddt = pearsonr(all_true_values_ilddt, all_predictions_ilddt)[0]
    overall_spear_ilddt = spearmanr(all_true_values_ilddt, all_predictions_ilddt)[0]
    
    print(f"Total samples evaluated: {len(all_true_values_ddg)}")
    print(f"ddG    - Pearson: {overall_pear_ddg:.4f}, Spearman: {overall_spear_ddg:.4f}")
    print(f"ilddt  - Pearson: {overall_pear_ilddt:.4f}, Spearman: {overall_spear_ilddt:.4f}")
    
    # Print fold-by-fold comparison
    print(f"\n{'='*70}")
    print("FOLD-BY-FOLD RESULTS")
    print(f"{'='*70}")
    print(f"{'Fold':<6} {'N':<6} {'ddG Pearson':<14} {'ddG Spearman':<14} "
          f"{'ilddt Pearson':<16} {'ilddt Spearman':<16}")
    print("-" * 70)
    
    for result in fold_results:
        print(f"{result['fold']:<6} {result['n_test']:<6} "
              f"{result['pearson_ddg']:<14.4f} {result['spearman_ddg']:<14.4f} "
              f"{result['pearson_ilddt']:<16.4f} {result['spearman_ilddt']:<16.4f}")
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'complex_name': all_complex_names,
        'predicted_ddG': all_predictions_ddg,
        'true_ddG': all_true_values_ddg,
        'predicted_ilddt': all_predictions_ilddt,
        'true_ilddt': all_true_values_ilddt,
        'fold': all_fold_indices
    })
    
    csv_path = os.path.join(args.output_dir, 'evaluation_predictions.csv')
    predictions_df.to_csv(csv_path, index=False)
    print(f"\nSaved predictions to: {csv_path}")
    
    # Save metrics
    metrics = {
        'overall': {
            'pearson_ddg': float(overall_pear_ddg),
            'spearman_ddg': float(overall_spear_ddg),
            'pearson_ilddt': float(overall_pear_ilddt),
            'spearman_ilddt': float(overall_spear_ilddt),
            'n_samples': len(all_true_values_ddg)
        },
        'folds': fold_results
    }
    
    json_path = os.path.join(args.output_dir, 'evaluation_metrics.json')
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to: {json_path}")
    
    print(f"\n{'='*70}")
    print("EVALUATION COMPLETE")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
    
