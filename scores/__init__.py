"""
Biophysical score calculation modules for protein structures.

Available score types:
- burial: k-nearest neighbor burial score
- interface: Interface contact score with Gaussian weighting
- flexibility: B-factor based flexibility score
- sasa: Solvent accessible surface area score
- dihedral: Dihedral angle change score (requires wildtype structure)
- lddt: Local distance difference test score (requires pre-computed OST JSON files)
"""

from .burial_score import burial_scores
from .interface_score import interface_scores
from .flexibility_score import flexibility_scores
from .sasa_score import sasa_scores
from .dihedral_score import compute_dihedral_scores
from .lddt_score import lddt_scores

__all__ = [
    'burial_scores',
    'interface_scores',
    'flexibility_scores',
    'sasa_scores',
    'compute_dihedral_scores',
    'lddt_scores',
]