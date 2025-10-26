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
    
    @property
    def device(self):
        """Get the device of the network."""
        if self.network is not None and hasattr(self.network, 'parameters'):
            return next(self.network.parameters()).device
        return torch.device('cpu')
    
    def set_temperature(self, temperature: float):
        """Set temperature for action selection (higher = more random)."""
        self.temperature = temperature
    
    def select_action(self, board, action_mask, temperature=0.1):
        """
        Select action using policy network with temperature sampling.
        
        Args:
            board: Current board state (6x7 numpy array)
            action_mask: Boolean mask of valid actions (7-element array)
            temperature: Sampling temperature (lower = more deterministic)
            
        Returns:
            Selected action (column index)
        """
        if self.network is None:
            # Fallback to random legal move if no network loaded
            valid_actions = np.where(action_mask == 1)[0]
            return np.random.choice(valid_actions) if len(valid_actions) > 0 else 3
        
        # Convert board to tensor - use board directly without perspective transformation
        # The network should learn to play from any player's perspective
        board_tensor = torch.FloatTensor(board).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Get action probabilities from network
            action_logits = self.network(board_tensor)
            action_logits = action_logits.squeeze(0).cpu()  # Move back to CPU
            
            # Apply action mask (set invalid actions to very negative values)
            masked_logits = action_logits.clone()
            masked_logits[action_mask == 0] = -float('inf')
            
            # Apply temperature and convert to probabilities
            if temperature > 0:
                scaled_logits = masked_logits / temperature
                action_probs = F.softmax(scaled_logits, dim=0)
                
                # Sample from the probability distribution
                action = torch.multinomial(action_probs, 1).item()
            else:
                # Greedy selection (take best action)
                action = torch.argmax(masked_logits).item()
            
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
        
        # Convert board to tensor - use board directly without perspective transformation
        board_tensor = torch.FloatTensor(board).unsqueeze(0).to(self.device)  # Move to device
        
        with torch.no_grad():
            action_logits = self.network(board_tensor)
            action_logits = action_logits.squeeze(0).cpu()  # Move back to CPU
        
        # Apply action mask
        masked_logits = action_logits.clone()
        masked_logits[action_mask == 0] = -float('inf')
        
        # Convert to probabilities
        probs = F.softmax(masked_logits, dim=0)
        return probs.numpy()