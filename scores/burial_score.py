# burial_score.py

import numpy as np
from Bio.PDB import PDBParser

def calc_virtual_cb(n_coord: np.ndarray, ca_coord: np.ndarray, c_coord: np.ndarray) -> np.ndarray:
    """
    Calculate pseudo C-beta coordinates for glycine residues.
    """
    v_n = n_coord - ca_coord
    v_c = c_coord - ca_coord
    v_n /= np.linalg.norm(v_n)
    v_c /= np.linalg.norm(v_c)
    bisec = v_n + v_c
    bisec /= np.linalg.norm(bisec)
    length = 1.522
    return ca_coord + bisec * length

def _get_cb_coordinates(pdb_path: str):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('struct', pdb_path)
    model = next(structure.get_models())
    cb_coords = {}
    for chain in model:
        for residue in chain:
            if residue.id[0] != ' ':
                continue
            if 'CA' not in residue or 'N' not in residue or 'C' not in residue:
                continue
            ca = residue['CA'].get_coord()
            if 'CB' in residue:
                cb = residue['CB'].get_coord()
            else:
                cb = calc_virtual_cb(residue['N'].get_coord(), ca, residue['C'].get_coord())
            key = (chain.id, residue.id[1], residue.id[2])
            cb_coords[key] = cb
    keys = list(cb_coords.keys())
    coords = np.vstack([cb_coords[k] for k in keys])
    return cb_coords, keys, coords

def _compute_centrality(idx: int, coords: np.ndarray) -> int:
    dists = np.linalg.norm(coords - coords[idx], axis=1)
    dists = np.delete(dists, idx)
    return int((dists < 10.0).sum())

def _compute_score_for_mutation(cb_coords, keys, coords, mutation: str, neighbor_count: int) -> float:
    """
    Raw burial: sum of centralities over k closest Cβ.
    """
    if len(mutation) < 4:
        raise ValueError(f"Mutation string '{mutation}' too short.")
    chain_id = mutation[1]
    try:
        pos = int(mutation[2:-1])
    except ValueError:
        raise ValueError(f"Cannot parse residue number from '{mutation}'.")
    candidates = [k for k in cb_coords if k[0] == chain_id and k[1] == pos]
    if not candidates:
        raise KeyError(f"Residue {chain_id}{pos} not found in PDB.")
    mut_key = candidates[0]
    mut_cb = cb_coords[mut_key]

    dists = np.linalg.norm(coords - mut_cb, axis=1)
    idxk = np.argsort(dists)[:neighbor_count]
    score = sum(_compute_centrality(i, coords) for i in idxk)
    return float(score)

def burial_score(pdb_path: str, mutation: str, neighbor_count: int = 9) -> float:
    """
    Compute **normalized** burial score for one mutation.
    """
    cb_coords, keys, coords = _get_cb_coordinates(pdb_path)

    raw_all = [
        _compute_score_for_mutation(cb_coords, keys, coords,
                                   f"A{chain}{res}A",
                                   neighbor_count)
        for chain, res, _ in keys
    ]
    max_raw = max(raw_all) if raw_all else 1.0
    if max_raw == 0:
        return 0.0

    raw = _compute_score_for_mutation(cb_coords, keys, coords,
                                     mutation, neighbor_count)
    return raw / max_raw

def burial_scores(pdb_path: str, mutations: list, neighbor_count: int = 9) -> list:
    """
    Compute normalized burial scores for multiple mutations.
    """
    cb_coords, keys, coords = _get_cb_coordinates(pdb_path)

    raw_all = [
        _compute_score_for_mutation(cb_coords, keys, coords,
                                   f"A{chain}{res}A",
                                   neighbor_count)
        for chain, res, _ in keys
    ]
    max_raw = max(raw_all) if raw_all else 1.0
    if max_raw == 0:
        return [0.0] * len(mutations)

    return [
        _compute_score_for_mutation(cb_coords, keys, coords,
                                   mut, neighbor_count) / max_raw
        for mut in mutations
    ]
