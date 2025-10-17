"""
Random agent for Connect-4.

This agent makes random valid moves and serves as a baseline for comparison.
"""

import numpy as np
from .base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """Agent that selects random valid moves."""
    
    def __init__(self, name: str = "Random Agent", seed: int = None):
        """
        Initialize the random agent.
        
        Args:
            name: Human-readable name for this agent
            seed: Random seed for reproducible behavior (optional)
        """
        super().__init__(name)
        self.rng = np.random.RandomState(seed)
    
    def select_action(self, board: np.ndarray, action_mask: np.ndarray) -> int:
        """
        Select a random valid action.
        
        Args:
            board: 6x7 board state (unused by random agent)
            action_mask: Valid moves mask
            
        Returns:
            Randomly selected valid column (0-6)
        """
        valid_actions = np.where(action_mask == 1)[0]
        
        if len(valid_actions) == 0:
            # Shouldn't happen, but fallback to column 0
            return 0
        
        return self.rng.choice(valid_actions)
    
    def set_seed(self, seed: int):
        """Set random seed for reproducible behavior."""
        self.rng = np.random.RandomState(seed)