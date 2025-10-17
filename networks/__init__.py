"""
Neural networks package for AlphaZero Connect-4.

This package contains PyTorch neural network implementations:
- PolicyNetwork: Predicts action probabilities
- ValueNetwork: Evaluates board positions
"""

from .policy_network import PolicyNetwork
from .value_network import ValueNetwork

__all__ = ['PolicyNetwork', 'ValueNetwork']