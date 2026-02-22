#!/usr/bin/env python3
"""
integrated_pipeline.py

Integrated pipeline that:
1. Calculates all biophysical scores (burial, interface, sasa, dihedral, lddt)
2. Extracts scores into individual NPZ files per structure
3. Merges all scores and embeddings into final NPZ files

Usage:
    python /n/netscratch/shakhnovich_lab/Lab/jonathanfeldman/test_protbff/ProtBFF/data_pipeline/calculate_all_scores.py \
        --csv  /n/netscratch/shakhnovich_lab/Lab/jonathanfeldman/test_protbff/ProtBFF/data/SKEMPI2_filtered_final.csv \
        --pdb_wt_dir /n/netscratch/shakhnovich_lab/Lab/jonathanfeldman/test_protbff/ProtBFF/data/wildtype/ \
        --pdb_opt_dir /n/netscratch/shakhnovich_lab/Lab/jonathanfeldman/test_protbff/ProtBFF/data/optimized/ \
        --ost_json_dir /n/netscratch/shakhnovich_lab/Lab/jonathanfeldman/AF3Complex/lddt_dir/ \
        --wt_embedding_dir /n/netscratch/shakhnovich_lab/Lab/jonathanfeldman/test_protbff/ProtBFF/data/wildtype_embeddings_2048/ \
        --opt_embedding_dir /n/netscratch/shakhnovich_lab/Lab/jonathanfeldman/test_protbff/ProtBFF/data/optimized_embeddings_2048/ \
        --output_dir /n/netscratch/shakhnovich_lab/Lab/jonathanfeldman/test_protbff/ProtBFF/data/merged_output/ \
   	    --n_workers 32
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
from multiprocessing import Pool

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
            try:
                file_index = int(re.findall(r'\d+', filename)[0])
                if file_index == index:
                    filepath = os.path.join(directory, filename)
                    return np.load(filepath, allow_pickle=True)
            except (ValueError, IndexError):
                continue
    return None


# ============================================================================
# Step 1: Worker Function (must be module-level for multiprocessing pickling)
# ============================================================================

def process_single_structure(args):
    """
    Worker function for parallel score calculation.
    Each worker processes one structure independently.
    """
    import warnings

    idx, pdb_opt_dir, pdb_wt_dir, ost_json_dir, pdb_opt_index, pdb_wt_index, temp_dir = args
    results = {'burial': [], 'interface': [], 'sasa': [], 'dihedral': [], 'lddt': []}

    pdb_path = find_pdb_file(pdb_opt_dir, idx, pdb_opt_index)
    pdb_wt_path = find_pdb_file(pdb_wt_dir, idx, pdb_wt_index)

    if pdb_path is None or not os.path.exists(pdb_path):
        return results

    try:
        mutations = get_all_residue_mutations(pdb_path)
    except Exception as e:
        print(f"PDB parsing failed for {idx}: {e}")
        return results

    mutation_strs = [m[0] for m in mutations]

    if len(mutations) == 0:
        return results

    # Burial scores
    try:
        burial = burial_scores(pdb_path, mutation_strs)
        for (_, chain, resnum, inscode), score in zip(mutations, burial):
            results['burial'].append({
                'index': idx, 'chain': chain, 'resnum': resnum,
                'inscode': inscode, 'score': score
            })
    except Exception as e:
        print(f"Burial failed for {idx}: {e}")

    # Interface scores
    try:
        interface = interface_scores(pdb_path, mutation_strs)
        for (_, chain, resnum, inscode), score in zip(mutations, interface):
            results['interface'].append({
                'index': idx, 'chain': chain, 'resnum': resnum,
                'inscode': inscode, 'score': score
            })
    except Exception as e:
        print(f"Interface failed for {idx}: {e}")

    # SASA scores
    try:
        sasa = sasa_scores(pdb_path, mutation_strs)
        for (_, chain, resnum, inscode), score in zip(mutations, sasa):
            results['sasa'].append({
                'index': idx, 'chain': chain, 'resnum': resnum,
                'inscode': inscode, 'score': score
            })
    except Exception as e:
        print(f"SASA failed for {idx}: {e}")

    # Dihedral scores (requires wildtype)
    if pdb_wt_path and os.path.exists(pdb_wt_path):
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                dihedral = compute_dihedral_scores(pdb_wt_path, pdb_path, mutation_strs)

            if any("invalid value encountered" in str(w.message) for w in caught):
                bad_scores = [(m, float(s)) for m, s in zip(mutation_strs, dihedral)
                              if np.isnan(s) or np.isinf(s)]
                flagged_entry = {
                'index': int(idx),
                'pdb_path': pdb_path,
                'pdb_wt_path': pdb_wt_path,
                'score_type': 'dihedral',
                'all_scores': [(m, float(s)) for m, s in zip(mutation_strs, dihedral)],
                'bad_scores': [(m, float(s)) for m, s in zip(mutation_strs, dihedral)
                               if np.isnan(s) or np.isinf(s)]
                                 }
                flagged_file = os.path.join(temp_dir, f"flagged_{idx}.json")
                with open(flagged_file, 'w') as f:
                    json.dump(flagged_entry, f, indent=2)

            for (_, chain, resnum, inscode), score in zip(mutations, dihedral):
                results['dihedral'].append({
                    'index': idx, 'chain': chain, 'resnum': resnum,
                    'inscode': inscode, 'score': score
                })
        except Exception as e:
            print(f"Dihedral failed for {idx}: {e}")

    # lDDT scores (requires OST JSON)
    if ost_json_dir and os.path.exists(ost_json_dir):
        try:
            lddt, ilddt = lddt_scores(pdb_path, mutation_strs, ost_json_dir=ost_json_dir)
            for (_, chain, resnum, inscode), score in zip(mutations, lddt):
                results['lddt'].append({
                    'index': idx, 'chain': chain, 'resnum': resnum,
                    'inscode': inscode, 'score': score, 'ilddt': ilddt
                })
        except Exception as e:
            print(f"lDDT failed for {idx}: {e}")

    return results
# ============================================================================
# Step 1: Calculate All Scores
# ============================================================================

def calculate_all_scores(csv_path, ddg_column, pdb_wt_dir, pdb_opt_dir, ost_json_dir, temp_dir, n_workers=1):
    """
    Calculate all biophysical scores for each structure.
    Saves raw scores to temporary CSV files.
    """
    print("\n" + "="*80)
    print("STEP 1: CALCULATING ALL SCORES")
    print("="*80 + "\n")

    df = pd.read_csv(csv_path)

    if ddg_column not in df.columns:
        raise ValueError(f"Column '{ddg_column}' not found in CSV")

    df = df.dropna(subset=[ddg_column]).reset_index(drop=True)
    if 'index' not in df.columns:
        df['index'] = df['#Pdb'].str.split('_').str[0].astype(int)
    print(f"Processing {len(df)} structures with non-NaN {ddg_column}")

    pdb_wt_index = build_pdb_index(pdb_wt_dir)
    pdb_opt_index = build_pdb_index(pdb_opt_dir)

    os.makedirs(temp_dir, exist_ok=True)

    unique_indices = df['index'].unique()

    # Build args list for workers
    args_list = [
    (idx, pdb_opt_dir, pdb_wt_dir, ost_json_dir, pdb_opt_index, pdb_wt_index, temp_dir)
    for idx in unique_indices
    ]
    score_results = {'burial': [], 'interface': [], 'sasa': [], 'dihedral': [], 'lddt': []}

    if n_workers > 1:
        print(f"Using {n_workers} parallel workers")
        with Pool(n_workers) as pool:
            all_results = list(tqdm(
                pool.imap(process_single_structure, args_list),
                total=len(args_list),
                desc="Calculating scores"
            ))
    else:
        print("Using 1 worker (sequential)")
        all_results = [
            process_single_structure(args)
            for args in tqdm(args_list, desc="Calculating scores")
        ]

    # Merge results from all workers
    for result in all_results:
        for score_type in score_results:
            score_results[score_type].extend(result[score_type])

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

    score_dirs = {}
    for score_type in score_csvs.keys():
        score_dir = os.path.join(npz_output_dir, score_type)
        os.makedirs(score_dir, exist_ok=True)
        score_dirs[score_type] = score_dir

    for score_type, csv_file in score_csvs.items():
        print(f"\nProcessing {score_type} scores...")

        df = pd.read_csv(csv_file)
        unique_indices = df['index'].unique()

        for idx in tqdm(unique_indices, desc=f"Extracting {score_type}"):
            idx_data = df[df['index'] == idx].copy()
            scores = idx_data['score'].values

            if score_type == 'dihedral':
                min_score = np.nanmin(scores)
                max_score = np.nanmax(scores)
                if max_score - min_score > 0:
                    scores = (scores - min_score) / (max_score - min_score)
                else:
                    scores = np.full_like(scores, 0.5)

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
    """
    print("\n" + "="*80)
    print("STEP 3: MERGING ALL DATA")
    print("="*80 + "\n")

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[ddg_column]).reset_index(drop=True)
    if 'index' not in df.columns:
        df['index'] = df['#Pdb'].str.split('_').str[0].astype(int)

    id_mapping = {}
    if id_column in df.columns:
        for _, row in df.iterrows():
            idx = int(row['index'])
            id_mapping[idx] = str(row[id_column])
        print(f"Loaded {len(id_mapping)} protein IDs from column '{id_column}'")
    else:
        print(f"Warning: ID column '{id_column}' not found in CSV. Will use numeric indices.")

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

    for idx in tqdm(sorted(grouped_files.keys()), desc="Merging"):
        merged_data = {}

        for score_type, filepath in grouped_files[idx].items():
            try:
                data = np.load(filepath, allow_pickle=True)
                if score_type in data:
                    merged_data[score_type] = data[score_type]
            except Exception as e:
                print(f"Error loading {filepath}: {e}")

        if len(merged_data) == 0:
            continue

        idx_df = df[df['index'] == idx]
        if len(idx_df) > 0:
            ddg_value = idx_df[ddg_column].iloc[0]
            merged_data['ddG'] = np.array([ddg_value])

        if idx in ost_json_index:
            try:
                with open(ost_json_index[idx], 'r') as f:
                    lddt_json = json.load(f)
                ilddt_val = lddt_json.get('ilddt')
                merged_data['ilddt'] = np.array([ilddt_val if ilddt_val is not None else np.nan])
            except Exception as e:
                print(f"Error loading ilddt from JSON for index {idx}: {e}")
                merged_data['ilddt'] = np.array([np.nan])
        else:
            merged_data['ilddt'] = np.array([np.nan])

        merged_data['index'] = np.array([idx])
        merged_data['protein_id'] = id_mapping.get(idx, str(idx))

        wildtype_emb = load_embedding(wt_embedding_dir, idx)
        optimized_emb = load_embedding(opt_embedding_dir, idx)

        if wildtype_emb is not None and optimized_emb is not None:
            if wildtype_emb.shape == optimized_emb.shape:
                merged_data['Xf'] = wildtype_emb - optimized_emb
                merged_data['Xr'] = optimized_emb - wildtype_emb

        output_file = os.path.join(output_dir, f"merged_{idx}.npz")
        np.savez(output_file, **merged_data)

    print(f"\nSuccessfully created {len(grouped_files)} merged NPZ files in {output_dir}")

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

    parser.add_argument('--csv', type=str, required=True)
    parser.add_argument('--ddg_column', type=str, default='ddG')
    parser.add_argument('--id_column', type=str, default='#Pdb')
    parser.add_argument('--pdb_wt_dir', type=str, required=True)
    parser.add_argument('--pdb_opt_dir', type=str, required=True)
    parser.add_argument('--ost_json_dir', type=str, default=None)
    parser.add_argument('--wt_embedding_dir', type=str, default=None)
    parser.add_argument('--opt_embedding_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--temp_dir', type=str, default='.temp_scores')
    parser.add_argument('--n_workers', type=int, default=1,
                        help='Number of parallel workers for score calculation (default: 1). '
                             'Set to number of available CPUs to parallelize.')
    parser.add_argument('--skip_calculation', action='store_true')
    parser.add_argument('--skip_extraction', action='store_true')

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
    print(f"Workers: {args.n_workers}")
    print("="*80)

    if not args.skip_calculation:
        score_csvs = calculate_all_scores(
            args.csv, args.ddg_column,
            args.pdb_wt_dir, args.pdb_opt_dir, args.ost_json_dir,
            args.temp_dir, args.n_workers
        )
    else:
        score_csvs = {}
        for score_type in ['burial', 'interface', 'sasa', 'dihedral', 'lddt']:
            csv_file = os.path.join(args.temp_dir, f"{score_type}_scores.csv")
            if os.path.exists(csv_file):
                score_csvs[score_type] = csv_file

    if not args.skip_extraction:
        npz_dir = os.path.join(args.temp_dir, 'npz_scores')
        score_dirs = extract_scores_to_npz(score_csvs, npz_dir)
    else:
        npz_dir = os.path.join(args.temp_dir, 'npz_scores')
        score_dirs = {}
        for score_type in ['burial', 'interface', 'sasa', 'dihedral', 'lddt']:
            score_dir = os.path.join(npz_dir, score_type)
            if os.path.exists(score_dir):
                score_dirs[score_type] = score_dir

    merge_all_data(
        score_dirs,
        args.wt_embedding_dir,
        args.opt_embedding_dir,
        args.csv,
        args.ddg_column,
        args.output_dir,
        args.ost_json_dir,
        args.id_column
    )

    total_time = time.time() - t_start
    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print(f"Total runtime: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"Output files saved to: {args.output_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()