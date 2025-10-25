"""
Policy network for Connect-4.

This module implements a convolutional neural network that predicts action probabilities
for a given board state.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    """
    Convolutional neural network for predicting action probabilities in Connect-4.
    
    Input: 6x7 board state (values: -1, 0, 1)
    Output: 7 action probabilities (one for each column)
    """
    
    def __init__(self, hidden_size: int = 128, num_conv_layers: int = 3):
        """
        Initialize the policy network.
        
        Args:
            hidden_size: Number of hidden units in fully connected layers
            num_conv_layers: Number of convolutional layers
        """
        super(PolicyNetwork, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_conv_layers = num_conv_layers
        
        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        
        # First conv layer: 1 input channel (board), 32 output channels
        self.conv_layers.append(nn.Conv2d(1, 8, kernel_size=3, padding=1))
        
        # Additional conv layers: 8 -> 8 channels
        for _ in range(num_conv_layers - 1):
            self.conv_layers.append(nn.Conv2d(8, 8, kernel_size=3, padding=1))
        
        # Batch normalization for each conv layer
        self.batch_norms = nn.ModuleList([nn.BatchNorm2d(8) for _ in range(num_conv_layers)])
        
        # Calculate the size after convolutions (6*7*8 = 336)
        conv_output_size = 6 * 7 * 8
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_policy = nn.Linear(hidden_size, 7)  # 7 columns for actions
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 6, 7) representing board states
            
        Returns:
            Action probabilities tensor of shape (batch_size, 7)
        """
        # Add channel dimension: (batch_size, 6, 7) -> (batch_size, 1, 6, 7)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        # Convolutional layers with ReLU activation and batch norm
        for conv, bn in zip(self.conv_layers, self.batch_norms):
            x = F.relu(bn(conv(x)))
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)  # (batch_size, 6*7*32)
        
        # Fully connected layers with ReLU and dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        # Policy head - output raw logits (no softmax here, done in agent)
        policy_logits = self.fc_policy(x)
        
        return policy_logits
    
    def predict_probabilities(self, board_state):
        """
        Predict action probabilities for a single board state.
        
        Args:
            board_state: 6x7 numpy array or tensor
            
        Returns:
            Action probabilities as numpy array
        """
        if not isinstance(board_state, torch.Tensor):
            board_state = torch.FloatTensor(board_state)
        
        if board_state.dim() == 2:
            board_state = board_state.unsqueeze(0)  # Add batch dimension
        
        self.eval()
        with torch.no_grad():
            logits = self.forward(board_state)
            probabilities = F.softmax(logits, dim=1)
            return probabilities.squeeze(0).numpy()
    
    def get_num_parameters(self):
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ResidualPolicyNetwork(PolicyNetwork):
    """
    Enhanced policy network with residual connections.
    
    This version includes skip connections to help with training deeper networks.
    """
    
    def __init__(self, hidden_size: int = 128, num_residual_blocks: int = 3):
        """
        Initialize the residual policy network.
        
        Args:
            hidden_size: Number of hidden units
            num_residual_blocks: Number of residual blocks
        """
        super(PolicyNetwork, self).__init__()  # Skip immediate parent, call grandparent
        
        self.hidden_size = hidden_size
        self.num_residual_blocks = num_residual_blocks
        
        # Initial convolution
        self.initial_conv = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.initial_bn = nn.BatchNorm2d(64)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList()
        for _ in range(num_residual_blocks):
            self.residual_blocks.append(ResidualBlock(64))
        
        # Policy head
        self.policy_conv = nn.Conv2d(64, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 6 * 7, 7)
    
    def forward(self, x):
        """Forward pass with residual connections."""
        # Add channel dimension if needed
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        # Initial convolution
        x = F.relu(self.initial_bn(self.initial_conv(x)))
        
        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Policy head
        x = F.relu(self.policy_bn(self.policy_conv(x)))
        x = x.view(x.size(0), -1)
        policy_logits = self.policy_fc(x)
        
        return policy_logits


class ResidualBlock(nn.Module):
    """Residual block for deeper networks."""
    
    def __init__(self, channels: int):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        """Forward pass with skip connection."""
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Add skip connection
        out += residual
        out = F.relu(out)
        
        return out