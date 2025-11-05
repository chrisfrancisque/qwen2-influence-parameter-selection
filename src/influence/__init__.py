"""
Influence computation for parameter selection
"""

from .compute_scores import compute_influence_scores
from .parameter_selector import select_parameters_to_mask

__all__ = [
    'compute_influence_scores',
    'select_parameters_to_mask'
]
