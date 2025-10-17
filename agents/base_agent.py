"""
Base agent class for Connect-4 agents.

This module defines the abstract base class that all Connect-4 agents must inherit from.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Dict, Any


class BaseAgent(ABC):
    """Abstract base class for all Connect-4 agents."""
    
    def __init__(self, name: str):
        """
        Initialize the base agent.
        
        Args:
            name: Human-readable name for this agent
        """
        self.name = name
        self.player_id = None  # Will be set to 1 or -1 when game starts
    
    def set_player_id(self, player_id: int):
        """
        Set which player this agent represents.
        
        Args:
            player_id: 1 for player 1, -1 for player 2
        """
        assert player_id in [1, -1], "Player ID must be 1 or -1"
        self.player_id = player_id
    
    @abstractmethod
    def select_action(self, board: np.ndarray, action_mask: np.ndarray) -> int:
        """
        Select an action given the current board state.
        
        Args:
            board: 6x7 numpy array representing the board state
                   0 = empty, 1 = player 1, -1 = player 2
            action_mask: 7-element binary array indicating valid moves
                        1 = valid column, 0 = invalid column
        
        Returns:
            Column index (0-6) to play in
        """
        pass
    
    def game_over(self, board: np.ndarray, winner: int = None):
        """
        Called when a game ends. Can be overridden for learning agents.
        
        Args:
            board: Final board state
            winner: 1 if player 1 won, -1 if player 2 won, 0 if tie, None if unknown
        """
        pass
    
    def __str__(self) -> str:
        return f"{self.name} (Player {self.player_id})"
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', player_id={self.player_id})"