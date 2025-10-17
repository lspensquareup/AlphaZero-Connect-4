"""
Neural network trainer for Connect-4 agents.

This module will contain training loops for policy and value networks.
Currently a placeholder for future implementation.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Dict, Optional
from collections import deque
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Trainer:
    """
    Neural network trainer for Connect-4 agents.
    
    This class will handle training of policy and value networks using
    self-play data and various training algorithms.
    """
    
    def __init__(self, save_dir: str = "models"):
        """
        Initialize the trainer.
        
        Args:
            save_dir: Directory to save trained models
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Training data storage
        self.training_data = deque(maxlen=100000)  # Store recent training examples
        
        # Training statistics
        self.training_stats = {
            'policy_losses': [],
            'value_losses': [],
            'combined_losses': [],
            'learning_rates': [],
            'epochs': 0
        }
    
    def train_policy_network(self, network, training_data: List[Tuple], 
                           num_epochs: int = 100, batch_size: int = 32,
                           learning_rate: float = 0.001) -> Dict:
        """
        Train a policy network using supervised learning.
        
        Args:
            network: PolicyNetwork to train
            training_data: List of (board_state, action_probabilities) tuples
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            
        Returns:
            Training statistics dictionary
        """
        print(f"Training policy network for {num_epochs} epochs...")
        
        # Setup optimizer and loss function
        optimizer = optim.Adam(network.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        network.train()
        epoch_losses = []
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            # Create batches
            for i in range(0, len(training_data), batch_size):
                batch = training_data[i:i + batch_size]
                
                # Prepare batch data
                boards = torch.FloatTensor([item[0] for item in batch])
                target_probs = torch.FloatTensor([item[1] for item in batch])
                
                # Forward pass
                optimizer.zero_grad()
                predicted_logits = network(boards)
                
                # Calculate loss
                loss = criterion(predicted_logits, target_probs)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / max(num_batches, 1)
            epoch_losses.append(avg_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{num_epochs}, Loss: {avg_loss:.6f}")
        
        # Save trained model
        model_path = os.path.join(self.save_dir, f"policy_network_epoch_{num_epochs}.pth")
        torch.save(network.state_dict(), model_path)
        print(f"Policy network saved to {model_path}")
        
        return {
            'final_loss': epoch_losses[-1] if epoch_losses else 0,
            'epoch_losses': epoch_losses,
            'num_epochs': num_epochs,
            'model_path': model_path
        }
    
    def train_value_network(self, network, training_data: List[Tuple],
                          num_epochs: int = 100, batch_size: int = 32,
                          learning_rate: float = 0.001) -> Dict:
        """
        Train a value network using supervised learning.
        
        Args:
            network: ValueNetwork to train
            training_data: List of (board_state, value) tuples
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            
        Returns:
            Training statistics dictionary
        """
        print(f"Training value network for {num_epochs} epochs...")
        
        # Setup optimizer and loss function
        optimizer = optim.Adam(network.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        network.train()
        epoch_losses = []
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            # Create batches
            for i in range(0, len(training_data), batch_size):
                batch = training_data[i:i + batch_size]
                
                # Prepare batch data
                boards = torch.FloatTensor([item[0] for item in batch])
                target_values = torch.FloatTensor([[item[1]] for item in batch])
                
                # Forward pass
                optimizer.zero_grad()
                predicted_values = network(boards)
                
                # Calculate loss
                loss = criterion(predicted_values, target_values)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / max(num_batches, 1)
            epoch_losses.append(avg_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{num_epochs}, Loss: {avg_loss:.6f}")
        
        # Save trained model
        model_path = os.path.join(self.save_dir, f"value_network_epoch_{num_epochs}.pth")
        torch.save(network.state_dict(), model_path)
        print(f"Value network saved to {model_path}")
        
        return {
            'final_loss': epoch_losses[-1] if epoch_losses else 0,
            'epoch_losses': epoch_losses,
            'num_epochs': num_epochs,
            'model_path': model_path
        }
    
    def generate_training_data_from_games(self, game_records: List[Dict]) -> Tuple[List, List]:
        """
        Generate training data from recorded games.
        
        Args:
            game_records: List of game records with board states and outcomes
            
        Returns:
            Tuple of (policy_data, value_data) lists
        """
        policy_data = []
        value_data = []
        
        for game in game_records:
            board_states = game.get('board_states', [])
            actions = game.get('actions', [])
            winner = game.get('winner', 0)
            
            for i, (board_state, action) in enumerate(zip(board_states, actions)):
                # Create action probability vector (one-hot for supervised learning)
                action_probs = np.zeros(7)
                action_probs[action] = 1.0
                
                # Calculate value based on game outcome and current player
                current_player = 1 if i % 2 == 0 else -1
                if winner == 0:  # Tie
                    value = 0.0
                elif winner == current_player:  # Current player wins
                    value = 1.0
                else:  # Current player loses
                    value = -1.0
                
                policy_data.append((board_state.copy(), action_probs))
                value_data.append((board_state.copy(), value))
        
        return policy_data, value_data
    
    def self_play_training_loop(self, policy_network, value_network, 
                              num_iterations: int = 100, games_per_iteration: int = 10):
        """
        Self-play training loop (placeholder for future implementation).
        
        This would implement the AlphaZero training algorithm with MCTS and self-play.
        """
        print("Self-play training loop not yet implemented.")
        print("This would involve:")
        print("1. Playing games using MCTS with current networks")
        print("2. Collecting training data from self-play games")
        print("3. Training networks on collected data")
        print("4. Evaluating new networks against previous versions")
        print("5. Updating networks if they perform better")
        
        # TODO: Implement full AlphaZero training loop
        pass
    
    def evaluate_networks(self, network1, network2, num_games: int = 20) -> Dict:
        """
        Evaluate two networks against each other.
        
        Args:
            network1: First network to evaluate
            network2: Second network to evaluate
            num_games: Number of games to play
            
        Returns:
            Evaluation results
        """
        from agents.policy_agent import PolicyAgent
        from training.tournament import Tournament
        
        # Create agents
        agent1 = PolicyAgent("Network 1")
        agent1.set_network(network1)
        
        agent2 = PolicyAgent("Network 2") 
        agent2.set_network(network2)
        
        # Run tournament
        tournament = Tournament(verbose=False)
        results = tournament.play_match(agent1, agent2, num_games)
        
        return results
    
    def save_training_stats(self, filename: str):
        """Save training statistics to file."""
        import json
        
        stats_path = os.path.join(self.save_dir, filename)
        with open(stats_path, 'w') as f:
            json.dump(self.training_stats, f, indent=2)
        
        print(f"Training statistics saved to {stats_path}")
    
    def load_training_stats(self, filename: str):
        """Load training statistics from file."""
        import json
        
        stats_path = os.path.join(self.save_dir, filename)
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                self.training_stats = json.load(f)
            print(f"Training statistics loaded from {stats_path}")
        else:
            print(f"Training statistics file not found: {stats_path}")


def create_sample_training_data(num_samples: int = 1000) -> Tuple[List, List]:
    """
    Create sample training data for testing (placeholder).
    
    This generates random board states and labels for initial testing.
    In practice, training data would come from self-play or expert games.
    """
    print(f"Generating {num_samples} sample training examples...")
    
    policy_data = []
    value_data = []
    
    for _ in range(num_samples):
        # Random board state
        board = np.random.choice([-1, 0, 1], size=(6, 7))
        
        # Random action probabilities (normalized)
        action_probs = np.random.random(7)
        action_probs = action_probs / action_probs.sum()
        
        # Random value
        value = np.random.uniform(-1, 1)
        
        policy_data.append((board, action_probs))
        value_data.append((board, value))
    
    return policy_data, value_data