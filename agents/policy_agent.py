"""
Policy-based neural network agent for Connect-4.

This agent uses a neural network to predict action probabilities.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional
from .base_agent import BaseAgent


class PolicyAgent(BaseAgent):
    """Agent that uses a policy network to select actions."""
    
    def __init__(self, name: str = "Policy Agent", network_path: Optional[str] = None):
        """
        Initialize the policy agent.
        
        Args:
            name: Human-readable name for this agent
            network_path: Path to saved policy network weights (optional)
        """
        super().__init__(name)
        self.network = None
        self.temperature = 1.0  # For action selection randomness
        
        if network_path:
            self.load_network(network_path)
    
    def load_network(self, network_path: str):
        """Load a trained policy network."""
        from ..networks.policy_network import PolicyNetwork
        
        self.network = PolicyNetwork()
        self.network.load_state_dict(torch.load(network_path, map_location='cpu'))
        self.network.eval()
    
    def set_network(self, network):
        """Set the policy network directly."""
        self.network = network
        self.network.eval()
    
    def set_temperature(self, temperature: float):
        """Set temperature for action selection (higher = more random)."""
        self.temperature = temperature
    
    def select_action(self, board: np.ndarray, action_mask: np.ndarray) -> int:
        """
        Select action using policy network.
        
        Args:
            board: 6x7 board state
            action_mask: Valid moves mask
            
        Returns:
            Selected column (0-6)
        """
        if self.network is None:
            # Fallback to random if no network loaded
            valid_actions = np.where(action_mask == 1)[0]
            return np.random.choice(valid_actions)
        
        # Prepare input for network
        # Adjust board perspective for current player
        player_board = board * self.player_id
        board_tensor = torch.FloatTensor(player_board).unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            action_probs = self.network(board_tensor)
            action_probs = action_probs.squeeze(0)  # Remove batch dimension
        
        # Apply action mask (set invalid actions to very low probability)
        masked_probs = action_probs.clone()
        masked_probs[action_mask == 0] = -float('inf')
        
        # Apply temperature and softmax
        if self.temperature > 0:
            action_probs = F.softmax(masked_probs / self.temperature, dim=0)
            # Sample from the distribution
            action = torch.multinomial(action_probs, 1).item()
        else:
            # Greedy selection (temperature = 0)
            action = torch.argmax(masked_probs).item()
        
        return action
    
    def get_action_probabilities(self, board: np.ndarray, action_mask: np.ndarray) -> np.ndarray:
        """
        Get action probabilities without sampling.
        
        Returns:
            Array of action probabilities for each column
        """
        if self.network is None:
            # Uniform distribution over valid moves
            probs = action_mask.astype(float)
            probs = probs / probs.sum() if probs.sum() > 0 else probs
            return probs
        
        # Adjust board perspective for current player
        player_board = board * self.player_id
        board_tensor = torch.FloatTensor(player_board).unsqueeze(0)
        
        with torch.no_grad():
            action_probs = self.network(board_tensor)
            action_probs = action_probs.squeeze(0)
        
        # Apply action mask
        masked_probs = action_probs.clone()
        masked_probs[action_mask == 0] = -float('inf')
        
        # Convert to probabilities
        probs = F.softmax(masked_probs, dim=0)
        return probs.numpy()