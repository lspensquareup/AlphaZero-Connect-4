"""
Training package for AlphaZero Connect-4.

This package contains training and evaluation utilities:
- Tournament: AI vs AI tournaments and evaluation
- Trainer: Neural network training loops (future implementation)
"""

from .tournament import Tournament
from .trainer import Trainer

__all__ = ['Tournament', 'Trainer']