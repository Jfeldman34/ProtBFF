# sasa_score.py

import numpy as np
from Bio.PDB import PDBParser
import freesasa

def _clean_atom_name(atom_name):
    """
    Clean atom name to remove spaces and ensure it's valid for freesasa.
    """
    # Remove all spaces completely and strip any whitespace
    cleaned = atom_name.replace(' ', '').strip()
    # Take only the first 4 characters if longer
    cleaned = cleaned[:4]
    return cleaned

def _get_residue_coordinates(pdb_path: str):
    """
    Get residue coordinates and metadata from PDB file.
    Similar to _get_cb_coordinates but returns residue-level information.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('struct', pdb_path)
    model = next(structure.get_models())
    
    residue_info = {}
    for chain in model:
        for residue in chain:
            if residue.id[0] != ' ':  # Skip hetero atoms
                continue
            if 'CA' not in residue:
                continue
            
            # Get CA coordinates as reference point
            ca_coord = residue['CA'].get_coord()
            key = (chain.id, residue.id[1], residue.id[2])
            residue_info[key] = {
                'ca_coord': ca_coord,
                'residue': residue,
                'chain': chain,
                'res_id': residue.id[1]
            }
    
    keys = list(residue_info.keys())
    return residue_info, keys

def _calculate_residue_sasa(residue, chain, pdb_path):
    """
    Calculate SASA for a specific residue within its chain context.
    """
    # Use the original PDB file directly with freesasa
    try:
        # Use freesasa's PDB parser directly
        structure = freesasa.Structure(pdb_path)
        result = freesasa.calc(structure)
        
        # Get SASA for the specific residue
        residue_sasa = 0.0
        res_id = residue.id[1]
        res_name = residue.get_resname()
        chain_id = chain.id
        
        try:
            # Access residue areas by chain and residue number
            residue_areas = result.residueAreas()
            
            # Try to access by chain ID first
            if chain_id in residue_areas:
                chain_areas = residue_areas[chain_id]
                if res_id in chain_areas:
                    atom_areas = chain_areas[res_id]
                    for atom in residue:
                        atom_name = atom.get_name().strip()
                        if atom_name in atom_areas:
                            atom_sasa = atom_areas[atom_name]
                            residue_sasa += atom_sasa
            
            # If not found, try by residue name
            elif res_name in residue_areas:
                atom_areas = residue_areas[res_name]
                for atom in residue:
                    atom_name = atom.get_name().strip()
                    if atom_name in atom_areas:
                        atom_sasa = atom_areas[atom_name]
                        residue_sasa += atom_sasa
                
        except Exception as e:
            pass
        
        return float(residue_sasa)
        
    except Exception as e:
        return 0.0

def _compute_sasa_for_residue(residue_info, keys, mutation: str, pdb_path: str) -> float:
    """
    Compute SASA for a specific residue mutation.
    """
    if len(mutation) < 4:
        raise ValueError(f"Mutation string '{mutation}' too short.")
    
    # Parse mutation: format is like "RA88A" where:
    # First char: original amino acid (R)
    # Second char: chain ID (A) 
    # Middle: residue number (88)
    # Last char: new amino acid (A)
    original_aa = mutation[0]
    chain_id = mutation[1]
    try:
        pos = int(mutation[2:-1])
    except ValueError:
        raise ValueError(f"Cannot parse residue number from '{mutation}'.")
    new_aa = mutation[-1]
    
    candidates = [k for k in residue_info if k[0] == chain_id and k[1] == pos]
    if not candidates:
        raise KeyError(f"Residue {chain_id}{pos} not found in PDB.")
    
    # Get the residue object
    mut_key = candidates[0]
    residue_obj = residue_info[mut_key]['residue']
    
    return _calculate_residue_sasa(residue_obj, residue_info[mut_key]['chain'], pdb_path)

def _compute_sasa_for_all_residues(residue_info, keys, pdb_path: str) -> list:
    """
    Compute SASA for all residues in the structure.
    """
    sasa_values = []
    
    for chain_id, res_id, _ in keys:
        # Create a dummy mutation string for consistency
        mutation = f"A{chain_id}{res_id}A"
        try:
            sasa = _compute_sasa_for_residue(residue_info, keys, mutation, pdb_path)
            sasa_values.append(sasa)
        except (KeyError, ValueError):
            sasa_values.append(0.0)
    
    return sasa_values

def sasa_score(pdb_path: str, mutation: str, sigma_sasa: float = 2.5) -> float:
    """
    Compute **normalized** SASA score for one mutation.
    Higher SASA = more exposed = higher score.
    Uses Gaussian-weighted averaging of nearby residues for robustness.
    """
    residue_info, keys = _get_residue_coordinates(pdb_path)
    
    # Calculate SASA for all residues
    sasa_all = _compute_sasa_for_all_residues(residue_info, keys, pdb_path)
    max_sasa = max(sasa_all) if sasa_all else 1.0
    if max_sasa == 0:
        return 0.0
    
    # Calculate SASA for the specific mutation
    sasa_mut = _compute_sasa_for_residue(residue_info, keys, mutation, pdb_path)
    
    # Normalize by maximum SASA in the structure
    return sasa_mut / max_sasa

def get_residue_sasa_freesasa(pdb_path, chain_id, res_id, sigma_sasa=2.5):
    """
    Compute SASA for a specific residue (by chain and residue number) using freesasa.
    Uses Gaussian-weighted averaging of nearby residues for robustness.
    """
    structure = freesasa.Structure(pdb_path)
    result = freesasa.calc(structure)
    residue_areas = result.residueAreas()
    
    if chain_id not in residue_areas:
        print(f"Chain {chain_id} not found in freesasa result.")
        return 0.0
    
    chain_areas = residue_areas[chain_id]
    all_residues = list(chain_areas.keys())
    
    # Convert res_id to string for comparison since freesasa uses string keys
    res_id_str = str(res_id)
    if res_id_str not in chain_areas:
        print(f"Residue {res_id} not found in chain {chain_id}.")
        # Try to find the closest available residue
        try:
            res_ids_int = [int(r) for r in all_residues]
            closest = min(res_ids_int, key=lambda x: abs(x - res_id))
            print(f"Closest available residue is {closest}")
            if abs(closest - res_id) <= 5:  # Within 5 residues
                print(f"Using residue {closest} instead of {res_id}")
                res_id_str = str(closest)
            else:
                return 0.0
        except:
            return 0.0
    
    # Get SASA for the target residue and nearby residues
    target_sasa = chain_areas[res_id_str].total
    
    # Find nearby residues within the same chain for Gaussian weighting
    nearby_sasas = []
    nearby_weights = []
    
    res_ids_int = [int(r) for r in all_residues]
    for other_res_id_str in all_residues:
        other_res_id = int(other_res_id_str)
        distance = abs(other_res_id - res_id)
        
        # Only consider residues within reasonable distance (e.g., 10 residues)
        if distance <= 10:
            other_sasa = chain_areas[other_res_id_str].total
            weight = np.exp(-(distance**2) / (2 * sigma_sasa**2))
            nearby_sasas.append(other_sasa)
            nearby_weights.append(weight)
    
    # Calculate Gaussian-weighted average
    if nearby_weights and sum(nearby_weights) > 0:
        weighted_sasa = sum(s * w for s, w in zip(nearby_sasas, nearby_weights)) / sum(nearby_weights)
        return weighted_sasa
    else:
        return target_sasa

def sasa_scores(pdb_path: str, mutations: list, sigma_sasa: float = 2.5) -> list:
    """
    Compute normalized SASA scores for multiple mutations.
    Uses Gaussian-weighted averaging of nearby residues for robustness.
    """
    # Compute SASA for all residues in the structure
    structure = freesasa.Structure(pdb_path)
    result = freesasa.calc(structure)
    residue_areas = result.residueAreas()
    
    all_sasa = []
    for chain_id in residue_areas:
        for res_id in residue_areas[chain_id]:
            sasa = residue_areas[chain_id][res_id].total
            all_sasa.append(sasa)
    max_sasa = max(all_sasa) if all_sasa else 1.0
    
    scores = []
    for mutation in mutations:
        if len(mutation) < 4:
            print(f"Mutation string '{mutation}' too short.")
            scores.append(0.0)
            continue
        chain_id = mutation[1]
        try:
            res_id = int(mutation[2:-1])
        except ValueError:
            print(f"Cannot parse residue number from '{mutation}'.")
            scores.append(0.0)
            continue
        sasa = get_residue_sasa_freesasa(pdb_path, chain_id, res_id, sigma_sasa)
        norm_sasa = sasa / max_sasa if max_sasa > 0 else 0.0
        scores.append(norm_sasa)
    return scores

#print(sasa_scores("data/SKEMPI2/SKEMPI2_cache/wildtype/64_1IAR.pdb", ["RA88A", "EA9Q", "KA84D", "IA5A", "RA85A"], sigma_sasa=1.0))