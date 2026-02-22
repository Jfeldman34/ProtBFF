#!/usr/bin/env python3
"""
lddt_score.py

Extract per-residue local lDDT scores and global ilddt from OST comparison JSON files.
This module reads pre-computed OST JSON files that contain local-lddt data.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple


def parse_mutation_string(mutation: str) -> tuple:
    """
    Parse mutation string to extract chain and residue number.
    
    Expected formats:
    - "A{chain}{resnum}A" (most score types)
    - "R{chain}{resnum}A" (flexibility score)
    
    Returns:
        (chain, resnum) tuple
    """
    # Skip first character (A or R), extract chain and resnum
    if len(mutation) < 4:
        raise ValueError(f"Invalid mutation string: {mutation}")
    
    chain = mutation[1]
    resnum_str = mutation[2:-1]  # Everything between chain and final 'A'
    
    try:
        resnum = int(resnum_str)
    except ValueError:
        raise ValueError(f"Could not parse residue number from mutation: {mutation}")
    
    return chain, resnum


def find_ost_json(pdb_path: str, ost_json_dir: str) -> Optional[str]:
    pdb_basename = Path(pdb_path).stem
    
    for suffix in [f"{pdb_basename}_ost_output.json", f"{pdb_basename}.json"]:
        json_path = os.path.join(ost_json_dir, suffix)
        if os.path.exists(json_path):
            return json_path
    
    return None


def load_local_lddt_data(json_path: str) -> Tuple[Dict[tuple, float], float]:
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    local_lddt_dict = {}
    
    if 'local_lddt' in data:
        local_lddt = data['local_lddt']
        
        # Format 1: flat dict with keys like "E.1.", "I.38."
        if isinstance(local_lddt, dict):
            for key, lddt_value in local_lddt.items():
                parts = key.split('.')
                if len(parts) >= 2:
                    chain = parts[0]
                    try:
                        resnum = int(parts[1])
                        local_lddt_dict[(chain, resnum)] = lddt_value
                    except ValueError:
                        continue
        
        # Format 2: list of objects with 'chain', 'rnum', 'lddt' fields
        elif isinstance(local_lddt, list):
            for residue_data in local_lddt:
                chain = residue_data.get('chain', '')
                resnum = residue_data.get('rnum', None)
                lddt_value = residue_data.get('lddt', np.nan)
                if resnum is not None:
                    local_lddt_dict[(chain, resnum)] = lddt_value
    
    ilddt_value = data.get('ilddt', np.nan)
    
    return local_lddt_dict, ilddt_value


def lddt_scores(pdb_path: str, mutations: List[str], ost_json_dir: str = None) -> Tuple[List[float], float]:
    """
    Extract local lDDT scores and global ilddt for specified residues from pre-computed OST JSON.
    
    Parameters:
        pdb_path: Path to the PDB file
        mutations: List of mutation strings (e.g., ["AA123A", "RB456A"])
        ost_json_dir: Directory containing OST JSON files (required)
    
    Returns:
        Tuple of:
        - List of local lDDT scores corresponding to each mutation
        - Global ilddt value for the entire structure
    """
    if ost_json_dir is None:
        raise ValueError("ost_json_dir must be specified for lddt_scores")
    
    # Find corresponding OST JSON file
    json_path = find_ost_json(pdb_path, ost_json_dir)
    
    if json_path is None:
        print(f"Warning: No OST JSON found for {pdb_path} in {ost_json_dir}")
        return [np.nan] * len(mutations), np.nan
    
    # Load local lDDT data and global ilddt
    try:
        local_lddt_dict, ilddt_value = load_local_lddt_data(json_path)
    except Exception as e:
        print(f"Error loading OST JSON {json_path}: {e}")
        return [np.nan] * len(mutations), np.nan
    
    # Extract scores for each mutation
    scores = []
    for mutation in mutations:
        try:
            chain, resnum = parse_mutation_string(mutation)
            score = local_lddt_dict.get((chain, resnum), np.nan)
            scores.append(score)
        except ValueError as e:
            print(f"Warning: {e}")
            scores.append(np.nan)
    
    return scores, ilddt_value


# For backwards compatibility and standalone testing
def compute_lddt_scores(pdb_path: str, mutations: List[str], ost_json_dir: str = None) -> Tuple[List[float], float]:
    """Alias for lddt_scores to maintain consistent naming with other score modules."""
    return lddt_scores(pdb_path, mutations, ost_json_dir)


if __name__ == "__main__":
    # Simple test
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python lddt_score.py <pdb_file> <ost_json_dir>")
        sys.exit(1)
    
    pdb_file = sys.argv[1]
    ost_json_dir = sys.argv[2]
    
    # Test with some dummy mutations
    test_mutations = ["AA123A", "AB456A"]
    
    scores, ilddt = lddt_scores(pdb_file, test_mutations, ost_json_dir)
    
    print(f"PDB: {pdb_file}")
    print(f"JSON dir: {ost_json_dir}")
    print(f"Local lDDT scores: {scores}")
    print(f"Global ilddt: {ilddt}")