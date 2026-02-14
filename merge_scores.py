#!/usr/bin/env python3
"""
preprocess_merged_scores.py

Preprocess merged NPZ files from integrated_pipeline.py for ML model training.

This script:
1. Loads merged NPZ files containing scores and embeddings
2. Applies max pooling with embedding differences
3. Extracts ilddt values
4. Creates final feature vectors (Xf, Xr) ready for model training

Usage:
    python preprocess_merged_scores.py \
        --merged_dir merged_output/ \
        --output_cache preprocessed_data.npz
"""

import os
import argparse
import numpy as np
from tqdm import tqdm


def max_pooling(score_vec, diff_mat):
    """
    Apply max pooling: weight embedding differences by scores and take max.
    
    Args:
        score_vec: (L,) array of per-residue scores
        diff_mat: (L, D) array of embedding differences
    
    Returns:
        (D,) array of pooled features
    """
    weighted = diff_mat * score_vec[:, None]
    return np.max(weighted, axis=0)


def preprocess(merged_npz_dir, cache_path=None):
    """
    Preprocess data from merged NPZ files.
    
    Each merged NPZ file should contain:
    - 'Xf': forward embedding differences (wildtype - optimized)
    - 'Xr': reverse embedding differences (optimized - wildtype)
    - 'ddG': binding affinity change
    - 'lddt': local distance difference test scores (per-residue)
    - 'ilddt': global ilddt value (scalar)
    - 'sasa': solvent accessible surface area (per-residue)
    - 'dihedral': dihedral/flexibility scores (per-residue)
    - 'burial': burial scores (per-residue)
    - 'interface': interface scores (per-residue)
    - 'index': numeric structure identifier
    - 'protein_id': protein identifier from CSV (e.g., "0_1CSE")
    
    Returns:
        Xf_final: (N, D) forward pooled features
        Xr_final: (N, D) reverse pooled features
        y: (N,) ddG values
        ilddt_values: (N,) global ilddt values  
        ids: (N,) extracted protein codes (e.g., "1CSE" from "0_1CSE")
        complex_names: (N,) full protein identifiers (e.g., "0_1CSE")
        mutations: (N,) mutation strings (set to None if not available)
    """
    
    # Check if cache exists
    if cache_path and os.path.exists(cache_path):
        print(f"Loading from cache: {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        
        # Extract IDs from complex_names (split after underscore)
        complex_names = data['ids']  # Full protein IDs like "0_1CSE"
        ids = np.array([s.split('_')[1] if '_' in str(s) else str(s) for s in complex_names])
        
        mutations = data.get('mutations', None)
        
        return data['Xf'], data['Xr'], data['y'], data['ilddt'], ids, complex_names, mutations
    
    # Debug counters
    missing_keys = 0
    missing_embeddings = 0
    length_mismatch = 0
    missing_ilddt = 0
    missing_protein_id = 0
    processed = 0
    
    # Result containers
    Xf_final, Xr_final, y, ilddt_values = [], [], [], []
    ids_list, complex_names_list = [], []
    
    # Find all NPZ files
    npz_files = [f for f in os.listdir(merged_npz_dir) if f.endswith('.npz')]
    
    print(f"Found {len(npz_files)} NPZ files in {merged_npz_dir}")
    
    for fname in tqdm(npz_files, desc='Preprocessing'):
        try:
            data = np.load(os.path.join(merged_npz_dir, fname), allow_pickle=True)
        except Exception as e:
            print(f"Error loading {fname}: {e}")
            continue
        
        # Check for required score keys
        required_keys = ['ddG', 'lddt', 'sasa', 'dihedral', 'burial', 'interface', 'index']
        if not all(key in data for key in required_keys):
            missing_keys += 1
            missing = [k for k in required_keys if k not in data]
            print(f"Skipping {fname}: missing keys {missing}")
            continue
        
        # Check for embeddings
        if 'Xf' not in data or 'Xr' not in data:
            missing_embeddings += 1
            print(f"Skipping {fname}: missing embeddings")
            continue
        
        # Load embeddings (already differences)
        Xf_emb = data['Xf']  # wildtype - optimized
        Xr_emb = data['Xr']  # optimized - wildtype
        
        # Load scores
        ddg = data['ddG']
        lddt = data['lddt']
        sasa_vec = data['sasa']
        dihedral_vec = data['dihedral']
        burial = data['burial']
        interface = data['interface']
        
        # Get protein ID
        if 'protein_id' in data:
            # Convert to string and handle array types
            protein_id = data['protein_id']
            if isinstance(protein_id, np.ndarray):
                protein_id = str(protein_id.item())
            else:
                protein_id = str(protein_id)
        else:
            # Fallback to numeric index if protein_id not available
            missing_protein_id += 1
            idx = data['index'].item() if hasattr(data['index'], 'item') else data['index']
            protein_id = str(int(idx[0]) if isinstance(idx, np.ndarray) else int(idx))
        
        # Extract ID code (part after underscore, if present)
        if '_' in protein_id:
            id_code = protein_id.split('_', 1)[1]  # e.g., "0_1CSE" -> "1CSE"
        else:
            id_code = protein_id
        
        # Extract ilddt (global value)
        if 'ilddt' in data:
            ilddt_val = data['ilddt'].item() if hasattr(data['ilddt'], 'item') else data['ilddt']
            if isinstance(ilddt_val, np.ndarray):
                ilddt_val = ilddt_val[0]
        else:
            missing_ilddt += 1
            ilddt_val = np.nan
        
        # Ensure all score vectors are 1D
        if lddt.ndim == 2 and lddt.shape[0] == 1:
            lddt = lddt.flatten()
        if sasa_vec.ndim == 2 and sasa_vec.shape[0] == 1:
            sasa_vec = sasa_vec.flatten()
        if dihedral_vec.ndim == 2 and dihedral_vec.shape[0] == 1:
            dihedral_vec = dihedral_vec.flatten()
        if burial.ndim == 2 and burial.shape[0] == 1:
            burial = burial.flatten()
        if interface.ndim == 2 and interface.shape[0] == 1:
            interface = interface.flatten()
        
        # Handle NaN values in lddt
        nan_mask = np.isnan(lddt)
        if nan_mask.any():
            mean_lddt = np.nanmean(lddt)
            if np.isnan(mean_lddt):
                mean_lddt = 0.5  # Fallback if all NaN
            lddt[nan_mask] = mean_lddt
        
        # Verify embedding dimensions
        if Xf_emb.ndim != 2 or Xr_emb.ndim != 2:
            print(f"Skipping {fname}: embeddings must be 2D (got Xf: {Xf_emb.ndim}D, Xr: {Xr_emb.ndim}D)")
            continue
        
        L = Xf_emb.shape[0]  # Number of residues
        
        # Check length consistency
        score_lengths = {
            'burial': burial.shape[0],
            'interface': interface.shape[0],
            'lddt': lddt.shape[0],
            'sasa': sasa_vec.shape[0],
            'dihedral': dihedral_vec.shape[0]
        }
        
        if any(L != length for length in score_lengths.values()):
            length_mismatch += 1
            print(f"Skipping {fname}: length mismatch - L={L}, scores={score_lengths}")
            continue
        
        # The embeddings are already differences:
        # Xf = wildtype - optimized
        # Xr = optimized - wildtype
        diff_fwd = Xf_emb
        diff_rev = Xr_emb
        
        # Invert lddt (higher lddt = better, so we invert for max pooling)
        lddt_inv = 1.0 - lddt
        
        # Apply max pooling for each score type
        # Order: interface, burial, lddt_inv, sasa, dihedral
        pi_fwd = max_pooling(interface, diff_fwd)
        pi_rev = max_pooling(interface, diff_rev)
        
        pb_fwd = max_pooling(burial, diff_fwd)
        pb_rev = max_pooling(burial, diff_rev)
        
        lddt_fwd = max_pooling(lddt_inv, diff_fwd)
        lddt_rev = max_pooling(lddt_inv, diff_rev)
        
        sasa_fwd = max_pooling(sasa_vec, diff_fwd)
        sasa_rev = max_pooling(sasa_vec, diff_rev)
        
        flex_fwd = max_pooling(dihedral_vec, diff_fwd)
        flex_rev = max_pooling(dihedral_vec, diff_rev)
        
        # Concatenate pooled vectors
        # Order: interface, burial, lddt_inv, sasa, dihedral
        feat_fwd = np.concatenate([pi_fwd, pb_fwd, lddt_fwd, sasa_fwd, flex_fwd], axis=0)
        feat_rev = np.concatenate([pi_rev, pb_rev, lddt_rev, sasa_rev, flex_rev], axis=0)
        
        # Append to results
        Xf_final.append(feat_fwd)
        Xr_final.append(feat_rev)
        y.append(float(ddg.item() if hasattr(ddg, 'item') else ddg))
        ids_list.append(id_code)            # "1CSE" (for fold matching)
        complex_names_list.append(protein_id)  # "0_1CSE" (full identifier)
        ilddt_values.append(float(ilddt_val))
        processed += 1
    
    # Print debug summary
    print("\n" + "="*60)
    print("PREPROCESSING SUMMARY")
    print("="*60)
    print(f"Total NPZ files found:       {len(npz_files)}")
    print(f"Processed successfully:      {processed}")
    print(f"Missing required keys:       {missing_keys}")
    print(f"Missing embeddings:          {missing_embeddings}")
    print(f"Length mismatches:           {length_mismatch}")
    print(f"Missing ilddt values:        {missing_ilddt}")
    print(f"Missing protein_id:          {missing_protein_id}")
    print("="*60 + "\n")
    
    if not Xf_final:
        raise ValueError("No data passed preprocessing! Check your merged NPZ files.")
    
    # Convert to numpy arrays
    Xf_final = np.vstack(Xf_final)
    Xr_final = np.vstack(Xr_final)
    y = np.array(y, dtype=np.float32)
    ilddt_values = np.array(ilddt_values, dtype=np.float32)
    ids = np.array(ids_list)  # Extracted codes like "1CSE"
    complex_names = np.array(complex_names_list)  # Full IDs like "0_1CSE"
    mutations = None  # Not available from merged NPZ files
    
    print(f"Final shapes:")
    print(f"  Xf: {Xf_final.shape}")
    print(f"  Xr: {Xr_final.shape}")
    print(f"  y: {y.shape}")
    print(f"  ilddt: {ilddt_values.shape}")
    print(f"  ids: {ids.shape}")
    print(f"  complex_names: {complex_names.shape}")
    print(f"\nSample IDs:")
    print(f"  ids (for matching): {ids[:5]}")
    print(f"  complex_names (full): {complex_names[:5]}")
    
    # Save to cache if requested
    if cache_path:
        print(f"\nSaving to cache: {cache_path}")
        np.savez(
            cache_path,
            Xf=Xf_final,
            Xr=Xr_final,
            y=y,
            ilddt=ilddt_values,
            ids=complex_names,  # Save full IDs in cache (will be split on load)
            mutations=mutations
        )
    
    return Xf_final, Xr_final, y, ilddt_values, ids, complex_names, mutations


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess merged NPZ files for ML model training"
    )
    
    parser.add_argument(
        '--merged_dir',
        type=str,
        required=True,
        help='Directory containing merged NPZ files from integrated_pipeline.py'
    )
    
    parser.add_argument(
        '--output_cache',
        type=str,
        required=True,
        help='Output path for preprocessed cache file (.npz)'
    )
    
    parser.add_argument(
        '--no_cache',
        action='store_true',
        help='Do not save cache file (for testing)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("MERGED SCORE PREPROCESSING")
    print("="*60)
    print(f"Input directory:  {args.merged_dir}")
    print(f"Output cache:     {args.output_cache}")
    print("="*60 + "\n")
    
    # Run preprocessing
    cache_path = None if args.no_cache else args.output_cache
    
    Xf, Xr, y, ilddt, ids, complex_names, mutations = preprocess(
        merged_npz_dir=args.merged_dir,
        cache_path=cache_path
    )
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE")
    print("="*60)
    print(f"Created feature arrays:")
    print(f"  Forward features (Xf):  {Xf.shape}")
    print(f"  Reverse features (Xr):  {Xr.shape}")
    print(f"  Target values (y):      {y.shape}")
    print(f"  IDs (for matching):     {ids.shape}")
    print(f"  Complex names (full):   {complex_names.shape}")
    print(f"  ilddt values:           {ilddt.shape}")
    print(f"  Mutations:              {'Not available' if mutations is None else mutations.shape}")
    
    # Print statistics
    print(f"\nData statistics:")
    print(f"  ddG range:     [{y.min():.3f}, {y.max():.3f}]")
    print(f"  ddG mean:      {y.mean():.3f} ± {y.std():.3f}")
    print(f"  ilddt range:   [{np.nanmin(ilddt):.3f}, {np.nanmax(ilddt):.3f}]")
    print(f"  ilddt mean:    {np.nanmean(ilddt):.3f} ± {np.nanstd(ilddt):.3f}")
    print(f"  ilddt NaN:     {np.sum(np.isnan(ilddt))}/{len(ilddt)}")
    
    print(f"\nSample identifiers:")
    print(f"  IDs: {ids[:3]} (for fold matching)")
    print(f"  Complex names: {complex_names[:3]} (full identifiers)")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()