"""
Value network for Connect-4.

This module implements a convolutional neural network that evaluates board positions
and returns a value estimate.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueNetwork(nn.Module):
    """
    Convolutional neural network for evaluating Connect-4 board positions.
    
    Input: 6x7 board state (values: -1, 0, 1)
    Output: Single value estimate in range [-1, 1]
    """
    
    def __init__(self, hidden_size: int = 128, num_conv_layers: int = 3):
        """
        Initialize the value network.
        
        Args:
            hidden_size: Number of hidden units in fully connected layers
            num_conv_layers: Number of convolutional layers
        """
        super(ValueNetwork, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_conv_layers = num_conv_layers
        
        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        
        # First conv layer: 1 input channel (board), 32 output channels
        self.conv_layers.append(nn.Conv2d(1, 32, kernel_size=3, padding=1))
        
        # Additional conv layers: 32 -> 32 channels
        for _ in range(num_conv_layers - 1):
            self.conv_layers.append(nn.Conv2d(32, 32, kernel_size=3, padding=1))
        
        # Batch normalization for each conv layer
        self.batch_norms = nn.ModuleList([nn.BatchNorm2d(32) for _ in range(num_conv_layers)])
        
        # Calculate the size after convolutions (6*7*32 = 1344)
        conv_output_size = 6 * 7 * 32
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_value = nn.Linear(hidden_size, 1)  # Single value output
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 6, 7) representing board states
            
        Returns:
            Value estimates tensor of shape (batch_size, 1)
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
        
        # Value head - output single value with tanh activation
        value = torch.tanh(self.fc_value(x))
        
        return value
    
    def evaluate_position(self, board_state):
        """
        Evaluate a single board position.
        
        Args:
            board_state: 6x7 numpy array or tensor
            
        Returns:
            Value estimate as float in range [-1, 1]
        """
        if not isinstance(board_state, torch.Tensor):
            board_state = torch.FloatTensor(board_state)
        
        if board_state.dim() == 2:
            board_state = board_state.unsqueeze(0)  # Add batch dimension
        
        self.eval()
        with torch.no_grad():
            value = self.forward(board_state)
            return value.item()
    
    def get_num_parameters(self):
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ResidualValueNetwork(ValueNetwork):
    """
    Enhanced value network with residual connections.
    
    This version includes skip connections to help with training deeper networks.
    """
    
    def __init__(self, hidden_size: int = 128, num_residual_blocks: int = 3):
        """
        Initialize the residual value network.
        
        Args:
            hidden_size: Number of hidden units
            num_residual_blocks: Number of residual blocks
        """
        super(ValueNetwork, self).__init__()  # Skip immediate parent, call grandparent
        
        self.hidden_size = hidden_size
        self.num_residual_blocks = num_residual_blocks
        
        # Initial convolution
        self.initial_conv = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.initial_bn = nn.BatchNorm2d(64)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList()
        for _ in range(num_residual_blocks):
            self.residual_blocks.append(ResidualBlock(64))
        
        # Value head
        self.value_conv = nn.Conv2d(64, 32, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 6 * 7, hidden_size)
        self.value_fc2 = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.3)
    
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
        
        # Value head
        x = F.relu(self.value_bn(self.value_conv(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.value_fc1(x))
        x = self.dropout(x)
        value = torch.tanh(self.value_fc2(x))
        
        return value


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


class CombinedNetwork(nn.Module):
    """
    Combined policy and value network sharing convolutional layers.
    
    This network outputs both action probabilities and value estimates,
    sharing the feature extraction layers for efficiency.
    """
    
    def __init__(self, hidden_size: int = 128, num_conv_layers: int = 3):
        """
        Initialize the combined network.
        
        Args:
            hidden_size: Number of hidden units
            num_conv_layers: Number of shared convolutional layers
        """
        super(CombinedNetwork, self).__init__()
        
        self.hidden_size = hidden_size
        
        # Shared convolutional layers
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(nn.Conv2d(1, 32, kernel_size=3, padding=1))
        
        for _ in range(num_conv_layers - 1):
            self.conv_layers.append(nn.Conv2d(32, 32, kernel_size=3, padding=1))
        
        self.batch_norms = nn.ModuleList([nn.BatchNorm2d(32) for _ in range(num_conv_layers)])
        
        # Shared fully connected layer
        conv_output_size = 6 * 7 * 32
        self.shared_fc = nn.Linear(conv_output_size, hidden_size)
        
        # Policy head
        self.policy_fc = nn.Linear(hidden_size, 7)
        
        # Value head
        self.value_fc = nn.Linear(hidden_size, 1)
        
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        """
        Forward pass returning both policy and value.
        
        Args:
            x: Input board state tensor
            
        Returns:
            Tuple of (policy_logits, value_estimate)
        """
        # Add channel dimension if needed
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        # Shared convolutional layers
        for conv, bn in zip(self.conv_layers, self.batch_norms):
            x = F.relu(bn(conv(x)))
        
        # Flatten and shared FC layer
        x = x.view(x.size(0), -1)
        shared_features = F.relu(self.shared_fc(x))
        shared_features = self.dropout(shared_features)
        
        # Policy head
        policy_logits = self.policy_fc(shared_features)
        
        # Value head
        value = torch.tanh(self.value_fc(shared_features))
        
        return policy_logits, value
    
    def get_policy(self, x):
        """Get only policy output."""
        policy_logits, _ = self.forward(x)
        return policy_logits
    
    def get_value(self, x):
        """Get only value output."""
        _, value = self.forward(x)
        return value