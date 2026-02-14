#!/usr/bin/env python3
"""
integrated_pipeline.py

Integrated pipeline that:
1. Calculates all biophysical scores (burial, interface, sasa, dihedral, lddt)
2. Extracts scores into individual NPZ files per structure
3. Merges all scores and embeddings into final NPZ files

Usage:
    python integrated_pipeline.py \
        --csv mutations.csv \
        --pdb_wt_dir wildtype/ \
        --pdb_opt_dir optimized/ \
        --ost_json_dir ost_json/ \
        --wt_embedding_dir wildtype_embeddings/ \
        --opt_embedding_dir optimized_embeddings/ \
        --output_dir merged_output/
"""

import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
from functools import lru_cache
from pathlib import Path
from collections import defaultdict
import json
import re

# Import score functions
from scores.burial_score import burial_scores
from scores.interface_score import interface_scores
from scores.sasa_score import sasa_scores
from scores.dihedral_score import compute_dihedral_scores
from scores.lddt_score import lddt_scores

# ============================================================================
# PDB Utilities
# ============================================================================

@lru_cache(maxsize=2000)
def parse_pdb_cached(pdb_path):
    """Cache PDB parsing to avoid re-parsing the same file."""
    from Bio.PDB import PDBParser
    parser = PDBParser(QUIET=True)
    return parser.get_structure('struct', pdb_path)


def get_all_residue_mutations(pdb_path):
    """
    Return a list of mutation-like strings for all residues in the structure.
    Returns: list of (mutation_str, chain, resnum, inscode)
    """
    structure = parse_pdb_cached(pdb_path)
    model = structure[0]
    mutations = []
    for chain in model:
        chain_id = chain.id
        for residue in chain:
            if residue.id[0] != ' ':
                continue
            resnum = residue.id[1]
            inscode = residue.id[2]
            mutation = f"A{chain_id}{resnum}A"
            mutations.append((mutation, chain_id, resnum, inscode))
    return mutations


def build_pdb_index(pdb_dir):
    """Build an index mapping idx -> filename for faster lookups."""
    index = {}
    if not pdb_dir or not os.path.isdir(pdb_dir):
        return index
    
    for filename in os.listdir(pdb_dir):
        if filename.endswith(".pdb") and "BLOOM" not in filename:
            parts = filename.split('_')
            if len(parts) >= 1:
                try:
                    # Extract first number from filename
                    idx = int(re.findall(r'\d+', parts[0])[0])
                    index[idx] = filename
                except (ValueError, IndexError):
                    continue
    return index


def find_pdb_file(pdb_dir, idx, pdb_index):
    """Find PDB file matching pattern {idx}_*.pdb."""
    if idx in pdb_index:
        return os.path.join(pdb_dir, pdb_index[idx])
    return None


# ============================================================================
# Embedding Utilities
# ============================================================================

def load_embedding(directory, index):
    """Load the embedding .npy file matching the given index."""
    if not directory or not os.path.exists(directory):
        return None
    
    for filename in os.listdir(directory):
        if filename.endswith('.npy'):
            # Extract index from filename
            try:
                file_index = int(re.findall(r'\d+', filename)[0])
                if file_index == index:
                    filepath = os.path.join(directory, filename)
                    return np.load(filepath, allow_pickle=True)
            except (ValueError, IndexError):
                continue
    return None


# ============================================================================
# Step 1: Calculate All Scores
# ============================================================================

def calculate_all_scores(csv_path, ddg_column, pdb_wt_dir, pdb_opt_dir, ost_json_dir, temp_dir):
    """
    Calculate all biophysical scores for each structure.
    Saves raw scores to temporary CSV files.
    """
    print("\n" + "="*80)
    print("STEP 1: CALCULATING ALL SCORES")
    print("="*80 + "\n")
    
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Filter by ddG column
    if ddg_column not in df.columns:
        raise ValueError(f"Column '{ddg_column}' not found in CSV")
    
    df = df.dropna(subset=[ddg_column]).reset_index(drop=True)
    print(f"Processing {len(df)} structures with non-NaN {ddg_column}")
    
    # Build PDB indices
    pdb_wt_index = build_pdb_index(pdb_wt_dir)
    pdb_opt_index = build_pdb_index(pdb_opt_dir)
    
    # Create temp directory
    os.makedirs(temp_dir, exist_ok=True)
    
    # Get unique indices
    unique_indices = df['index'].unique()
    
    # Store results for each score type
    score_results = {
        'burial': [],
        'interface': [],
        'sasa': [],
        'dihedral': [],
        'lddt': []
    }
    
    # Process each structure
    for idx in tqdm(unique_indices, desc="Calculating scores"):
        # Find PDB files
        pdb_path = find_pdb_file(pdb_opt_dir, idx, pdb_opt_index)
        pdb_wt_path = find_pdb_file(pdb_wt_dir, idx, pdb_wt_index)
        
        if pdb_path is None or not os.path.exists(pdb_path):
            continue
        
        # Get residue list
        mutations = get_all_residue_mutations(pdb_path)
        mutation_strs = [m[0] for m in mutations]
        
        if len(mutations) == 0:
            continue
        
        # Calculate burial scores
        try:
            burial = burial_scores(pdb_path, mutation_strs)
            for (_, chain, resnum, inscode), score in zip(mutations, burial):
                score_results['burial'].append({
                    'index': idx, 'chain': chain, 'resnum': resnum,
                    'inscode': inscode, 'score': score
                })
        except Exception as e:
            print(f"Burial failed for {idx}: {e}")
        
        # Calculate interface scores
        try:
            interface = interface_scores(pdb_path, mutation_strs)
            for (_, chain, resnum, inscode), score in zip(mutations, interface):
                score_results['interface'].append({
                    'index': idx, 'chain': chain, 'resnum': resnum,
                    'inscode': inscode, 'score': score
                })
        except Exception as e:
            print(f"Interface failed for {idx}: {e}")
        
        # Calculate SASA scores
        try:
            sasa = sasa_scores(pdb_path, mutation_strs)
            for (_, chain, resnum, inscode), score in zip(mutations, sasa):
                score_results['sasa'].append({
                    'index': idx, 'chain': chain, 'resnum': resnum,
                    'inscode': inscode, 'score': score
                })
        except Exception as e:
            print(f"SASA failed for {idx}: {e}")
        
        # Calculate dihedral scores (requires wildtype)
        if pdb_wt_path and os.path.exists(pdb_wt_path):
            try:
                dihedral = compute_dihedral_scores(pdb_wt_path, pdb_path, mutation_strs)
                for (_, chain, resnum, inscode), score in zip(mutations, dihedral):
                    score_results['dihedral'].append({
                        'index': idx, 'chain': chain, 'resnum': resnum,
                        'inscode': inscode, 'score': score
                    })
            except Exception as e:
                print(f"Dihedral failed for {idx}: {e}")
        
        # Calculate lddt scores (requires OST JSON)
        if ost_json_dir and os.path.exists(ost_json_dir):
            try:
                lddt, ilddt = lddt_scores(pdb_path, mutation_strs, ost_json_dir=ost_json_dir)
                for (_, chain, resnum, inscode), score in zip(mutations, lddt):
                    score_results['lddt'].append({
                        'index': idx, 'chain': chain, 'resnum': resnum,
                        'inscode': inscode, 'score': score, 'ilddt': ilddt
                    })
            except Exception as e:
                print(f"lDDT failed for {idx}: {e}")
    
    # Save score CSVs
    score_csvs = {}
    for score_type, results in score_results.items():
        if len(results) > 0:
            csv_file = os.path.join(temp_dir, f"{score_type}_scores.csv")
            pd.DataFrame(results).to_csv(csv_file, index=False)
            score_csvs[score_type] = csv_file
            print(f"Saved {score_type} scores: {len(results)} entries")
    
    return score_csvs


# ============================================================================
# Step 2: Extract Scores to NPZ Files
# ============================================================================

def extract_scores_to_npz(score_csvs, npz_output_dir):
    """
    Extract scores from CSV files into individual NPZ files per structure.
    Applies normalization for dihedral scores.
    """
    print("\n" + "="*80)
    print("STEP 2: EXTRACTING SCORES TO NPZ FILES")
    print("="*80 + "\n")
    
    os.makedirs(npz_output_dir, exist_ok=True)
    
    # Create subdirectories for each score type
    score_dirs = {}
    for score_type in score_csvs.keys():
        score_dir = os.path.join(npz_output_dir, score_type)
        os.makedirs(score_dir, exist_ok=True)
        score_dirs[score_type] = score_dir
    
    # Process each score type
    for score_type, csv_file in score_csvs.items():
        print(f"\nProcessing {score_type} scores...")
        
        df = pd.read_csv(csv_file)
        unique_indices = df['index'].unique()
        
        for idx in tqdm(unique_indices, desc=f"Extracting {score_type}"):
            idx_data = df[df['index'] == idx].copy()
            scores = idx_data['score'].values
            
            # Apply normalization for dihedral scores
            if score_type == 'dihedral':
                min_score = np.nanmin(scores)
                max_score = np.nanmax(scores)
                
                if max_score - min_score > 0:
                    scores = (scores - min_score) / (max_score - min_score)
                else:
                    scores = np.full_like(scores, 0.5)
            
            # Save to NPZ
            output_file = os.path.join(score_dirs[score_type], f"{score_type}_{idx}.npz")
            np.savez(output_file, **{score_type: scores})
        
        print(f"Created {len(unique_indices)} NPZ files for {score_type}")
    
    return score_dirs


# ============================================================================
# Step 3: Merge All Scores and Embeddings
# ============================================================================

def merge_all_data(score_dirs, wt_embedding_dir, opt_embedding_dir, csv_path, ddg_column, output_dir, ost_json_dir=None, id_column='#Pdb'):
    """
    Merge all scores and embeddings into final NPZ files.
    Also extracts global ilddt values from OST JSON files and saves protein IDs from CSV.
    
    Args:
        id_column: Column name in CSV to use as protein identifier (default: '#Pdb')
    """
    print("\n" + "="*80)
    print("STEP 3: MERGING ALL DATA")
    print("="*80 + "\n")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Read CSV for ddG values and protein IDs
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[ddg_column]).reset_index(drop=True)
    
    # Create mapping: index -> protein_id
    id_mapping = {}
    if id_column in df.columns:
        for _, row in df.iterrows():
            idx = int(row['index'])
            id_mapping[idx] = str(row[id_column])
        print(f"Loaded {len(id_mapping)} protein IDs from column '{id_column}'")
    else:
        print(f"Warning: ID column '{id_column}' not found in CSV. Will use numeric indices.")
    
    # Get all NPZ files grouped by index
    grouped_files = defaultdict(lambda: {})
    
    for score_type, score_dir in score_dirs.items():
        if not os.path.exists(score_dir):
            continue
        
        for filename in os.listdir(score_dir):
            if filename.endswith('.npz'):
                try:
                    idx = int(re.findall(r'\d+', filename)[0])
                    grouped_files[idx][score_type] = os.path.join(score_dir, filename)
                except (ValueError, IndexError):
                    continue
    
    print(f"Found {len(grouped_files)} unique structures to merge")
    
    # Build index of OST JSON files if directory provided
    ost_json_index = {}
    if ost_json_dir and os.path.exists(ost_json_dir):
        print(f"Indexing OST JSON files from {ost_json_dir}")
        for filename in os.listdir(ost_json_dir):
            if filename.endswith('.json'):
                try:
                    idx = int(re.findall(r'\d+', filename)[0])
                    ost_json_index[idx] = os.path.join(ost_json_dir, filename)
                except (ValueError, IndexError):
                    continue
        print(f"Found {len(ost_json_index)} OST JSON files")
    
    # Process each structure
    for idx in tqdm(sorted(grouped_files.keys()), desc="Merging"):
        merged_data = {}
        
        # Load all score arrays
        for score_type, filepath in grouped_files[idx].items():
            try:
                data = np.load(filepath, allow_pickle=True)
                # Extract the score array
                if score_type in data:
                    merged_data[score_type] = data[score_type]
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
        
        # Skip if no scores loaded
        if len(merged_data) == 0:
            continue
        
        # Get ddG value
        idx_df = df[df['index'] == idx]
        if len(idx_df) > 0:
            ddg_value = idx_df[ddg_column].iloc[0]
            merged_data['ddG'] = np.array([ddg_value])
        
        # Extract ilddt from OST JSON if available
        if idx in ost_json_index:
            try:
                with open(ost_json_index[idx], 'r') as f:
                    lddt_json = json.load(f)
                
                ilddt_val = lddt_json.get('ilddt')
                if ilddt_val is not None:
                    merged_data['ilddt'] = np.array([ilddt_val])
                else:
                    merged_data['ilddt'] = np.array([np.nan])
            except Exception as e:
                print(f"Error loading ilddt from JSON for index {idx}: {e}")
                merged_data['ilddt'] = np.array([np.nan])
        else:
            # If no OST JSON found, set to NaN
            merged_data['ilddt'] = np.array([np.nan])
        
        # Save numeric index
        merged_data['index'] = np.array([idx])
        
        # Save protein ID from CSV
        if idx in id_mapping:
            merged_data['protein_id'] = id_mapping[idx]
        else:
            merged_data['protein_id'] = str(idx)
        
        # Load embeddings
        wildtype_emb = load_embedding(wt_embedding_dir, idx)
        optimized_emb = load_embedding(opt_embedding_dir, idx)
        
        if wildtype_emb is not None and optimized_emb is not None:
            # Ensure compatible shapes
            if wildtype_emb.shape == optimized_emb.shape:
                # Compute differences
                Xf = wildtype_emb - optimized_emb  # wildtype - optimized
                Xr = optimized_emb - wildtype_emb  # optimized - wildtype
                
                merged_data['Xf'] = Xf
                merged_data['Xr'] = Xr
        
        # Save merged file
        output_file = os.path.join(output_dir, f"merged_{idx}.npz")
        np.savez(output_file, **merged_data)
    
    print(f"\nSuccessfully created {len(grouped_files)} merged NPZ files in {output_dir}")
    
    # Print summary of ilddt extraction
    if ost_json_dir:
        ilddt_count = sum(1 for idx in grouped_files.keys() if idx in ost_json_index)
        print(f"Extracted ilddt for {ilddt_count}/{len(grouped_files)} structures")


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Integrated pipeline: Calculate, Extract, and Merge biophysical scores"
    )
    
    # Input files
    parser.add_argument('--csv', type=str, required=True,
                       help='CSV file with index and ddG columns')
    parser.add_argument('--ddg_column', type=str, default='ddG_LY-CoV555',
                       help='Column name to filter by (default: ddG_LY-CoV555)')
    parser.add_argument('--id_column', type=str, default='#Pdb',
                       help='Column name to use as protein identifier (default: #Pdb)')
    
    # PDB directories
    parser.add_argument('--pdb_wt_dir', type=str, required=True,
                       help='Directory containing wildtype PDB files')
    parser.add_argument('--pdb_opt_dir', type=str, required=True,
                       help='Directory containing optimized PDB files')
    parser.add_argument('--ost_json_dir', type=str, default=None,
                       help='Directory containing OST JSON files (for lddt)')
    
    # Embedding directories
    parser.add_argument('--wt_embedding_dir', type=str, default=None,
                       help='Directory containing wildtype embeddings (.npy files)')
    parser.add_argument('--opt_embedding_dir', type=str, default=None,
                       help='Directory containing optimized embeddings (.npy files)')
    
    # Output directories
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for merged NPZ files')
    parser.add_argument('--temp_dir', type=str, default='.temp_scores',
                       help='Temporary directory for intermediate files')
    
    # Processing options
    parser.add_argument('--skip_calculation', action='store_true',
                       help='Skip score calculation (use existing CSV files)')
    parser.add_argument('--skip_extraction', action='store_true',
                       help='Skip NPZ extraction (use existing NPZ files)')
    
    args = parser.parse_args()
    
    t_start = time.time()
    
    print("\n" + "="*80)
    print("INTEGRATED BIOPHYSICAL SCORE PIPELINE")
    print("="*80)
    print(f"CSV: {args.csv}")
    print(f"ddG column: {args.ddg_column}")
    print(f"Wildtype PDBs: {args.pdb_wt_dir}")
    print(f"Optimized PDBs: {args.pdb_opt_dir}")
    print(f"OST JSONs: {args.ost_json_dir or 'Not provided'}")
    print(f"Output: {args.output_dir}")
    print("="*80)
    
    # Step 1: Calculate scores
    if not args.skip_calculation:
        score_csvs = calculate_all_scores(
            args.csv, args.ddg_column,
            args.pdb_wt_dir, args.pdb_opt_dir, args.ost_json_dir,
            args.temp_dir
        )
    else:
        # Load existing CSVs
        score_csvs = {}
        for score_type in ['burial', 'interface', 'sasa', 'dihedral', 'lddt']:
            csv_file = os.path.join(args.temp_dir, f"{score_type}_scores.csv")
            if os.path.exists(csv_file):
                score_csvs[score_type] = csv_file
    
    # Step 2: Extract to NPZ
    if not args.skip_extraction:
        npz_dir = os.path.join(args.temp_dir, 'npz_scores')
        score_dirs = extract_scores_to_npz(score_csvs, npz_dir)
    else:
        # Use existing NPZ directory
        npz_dir = os.path.join(args.temp_dir, 'npz_scores')
        score_dirs = {}
        for score_type in ['burial', 'interface', 'sasa', 'dihedral', 'lddt']:
            score_dir = os.path.join(npz_dir, score_type)
            if os.path.exists(score_dir):
                score_dirs[score_type] = score_dir
    
    # Step 3: Merge all data
    merge_all_data(
        score_dirs,
        args.wt_embedding_dir,
        args.opt_embedding_dir,
        args.csv,
        args.ddg_column,
        args.output_dir,
        args.ost_json_dir,  # Pass OST JSON dir for ilddt extraction
        args.id_column  # Pass ID column name
    )
    
    total_time = time.time() - t_start
    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print(f"Total runtime: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"Output files saved to: {args.output_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()