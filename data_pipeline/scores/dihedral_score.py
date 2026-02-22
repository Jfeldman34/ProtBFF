#!/usr/bin/env python3

"""
dihedral_score.py

A script measuring structural changes in side-chain dihedral angles after optimization.

Tunables:
  - CHI_DEFINITIONS: Side-chain chi angle atom tuples per residue type.
"""

import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.vectors import calc_dihedral, Vector

# ============================ Tunable parameters ============================
# Define which dihedrals to track per residue type
CHI_DEFINITIONS = {
    'ARG': [('N','CA','CB','CG'), ('CA','CB','CG','CD'), ('CB','CG','CD','NE'), ('CG','CD','NE','CZ')],
    'ASN': [('N','CA','CB','CG'), ('CA','CB','CG','OD1')],
    'ASP': [('N','CA','CB','CG'), ('CA','CB','CG','OD1')],
    'GLN': [('N','CA','CB','CG'), ('CA','CB','CG','CD'), ('CB','CG','CD','OE1')],
    'GLU': [('N','CA','CB','CG'), ('CA','CB','CG','CD'), ('CB','CG','CD','OE1')],
    'HIS': [('N','CA','CB','CG'), ('CA','CB','CG','ND1')],
    'ILE': [('N','CA','CB','CG1'), ('CA','CB','CG1','CD1')],
    'LEU': [('N','CA','CB','CG'), ('CA','CB','CG','CD1')],
    'LYS': [('N','CA','CB','CG'), ('CA','CB','CG','CD'), ('CB','CG','CD','CE'), ('CG','CD','CE','NZ')],
    'MET': [('N','CA','CB','CG'), ('CA','CB','CG','SD'), ('CB','CG','SD','CE')],
    'PHE': [('N','CA','CB','CG'), ('CA','CB','CG','CD1')],
    'PRO': [('N','CA','CB','CG'), ('CA','CB','CG','CD')],
    'SER': [('N','CA','CB','OG')],
    'THR': [('N','CA','CB','OG1')],
    'TRP': [('N','CA','CB','CG'), ('CA','CB','CG','CD1')],
    'TYR': [('N','CA','CB','CG'), ('CA','CB','CG','CD1')],
    'VAL': [('N','CA','CB','CG1')]
}
# ===========================================================================

def _get_structure(pdb_path):
    """Parse PDB file and return structure."""
    parser = PDBParser(QUIET=True)
    return parser.get_structure('struct', pdb_path)

def _get_mutation_residue(structure, mutation):
    """Get the mutated residue from structure."""
    if len(mutation) < 4:
        raise ValueError(f"Mutation string '{mutation}' too short.")
    chain_id = mutation[1]
    try:
        pos = int(mutation[2:-1])
    except ValueError:
        raise ValueError(f"Cannot parse residue number from '{mutation}'.")
    for res in structure[0][chain_id]:
        if res.id[1] == pos:
            return res
    raise KeyError(f"Residue {chain_id}{pos} not found in PDB.")

def compute_dihedral_score(pdb_wt, pdb_opt, mutation):
    """
    Compute dihedral score: sum of absolute chi-angle changes (°) for all side chains except the mutated residue.
    Returns a score normalized to [0,1] by dividing by 360 degrees.
    """
    wt, opt = _get_structure(pdb_wt), _get_structure(pdb_opt)
    mut_res = _get_mutation_residue(wt, mutation)
    total_delta = 0.0
    # Compare only first model
    for res_wt, res_opt in zip(wt[0].get_residues(), opt[0].get_residues()):
        if res_wt.id[0] != ' ' or res_wt == mut_res:
            continue
        name = res_wt.get_resname()
        if name not in CHI_DEFINITIONS:
            continue
        for atom_names in CHI_DEFINITIONS[name]:
            try:
                coords_wt = [Vector(res_wt[a].get_coord()) for a in atom_names]
                coords_opt = [Vector(res_opt[a].get_coord()) for a in atom_names]
                chi_wt = np.degrees(calc_dihedral(*coords_wt))
                chi_opt = np.degrees(calc_dihedral(*coords_opt))
                diff = abs(chi_opt - chi_wt) % 360
                delta = min(diff, 360 - diff)
                # ignore symmetric 180° flips in aromatics
                if name in ('PHE','TYR','HIS') and abs(delta - 180) < 1e-3:
                    continue
                total_delta += delta
            except KeyError:
                continue
    # Normalize by 360 degrees to get a score in [0,1]
    return total_delta/360.0

def compute_dihedral_scores(pdb_wt, pdb_opt, mutations):
    """Compute dihedral scores for multiple mutations."""
    return [compute_dihedral_score(pdb_wt, pdb_opt, m) for m in mutations]

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Compute dihedral scores for mutations.")
    p.add_argument("wildtype_pdb", nargs='?', default="data/SKEMPI2/SKEMPI2_cache/wildtype/174_1BRS.pdb",
                   help="Path to wildtype PDB file")
    p.add_argument("optimized_pdb", nargs='?', default="data/SKEMPI2/SKEMPI2_cache/optimized/174_1BRS.pdb",
                   help="Path to optimized PDB file")
    p.add_argument("mutations", nargs='*', default=["HA100L"],
                   help="Mutation strings, e.g. HA100L")
    args = p.parse_args()
    if not args.mutations:
        p.error("At least one mutation must be provided")
    for m in args.mutations:
        score = compute_dihedral_score(args.wildtype_pdb, args.optimized_pdb, m)
        print(f"{m}\tDihedral: {score:.3f}") 