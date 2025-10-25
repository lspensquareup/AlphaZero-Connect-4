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
import random

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from connect4_env import GymnasiumConnectFour
from agents.minimax_agent import MinimaxAgent
from agents.random_agent import RandomAgent


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
                           learning_rate: float = 0.001, device: torch.device = None) -> Dict:
        """
        Train a policy network using supervised learning.
        
        Args:
            network: PolicyNetwork to train
            training_data: List of (board_state, action_probabilities) tuples
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            device: Device to use for training (CPU, CUDA, MPS)
            
        Returns:
            Training statistics dictionary
        """
        if device is None:
            device = torch.device('cpu')
            
        print(f"Training policy network for {num_epochs} epochs...")
        
        # Move network to device
        network = network.to(device)
        
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
                boards = torch.FloatTensor([item[0] for item in batch]).to(device)
                target_probs = torch.FloatTensor([item[1] for item in batch]).to(device)
                
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
                          learning_rate: float = 0.001, device: torch.device = None) -> Dict:
        """
        Train a value network using supervised learning.
        
        Args:
            network: ValueNetwork to train
            training_data: List of (board_state, value) tuples
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            device: Device to use for training (CPU, CUDA, MPS)
            
        Returns:
            Training statistics dictionary
        """
        if device is None:
            device = torch.device('cpu')
            
        print(f"Training value network for {num_epochs} epochs...")
        
        # Move network to device
        network = network.to(device)
        
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
                boards = torch.FloatTensor([item[0] for item in batch]).to(device)
                target_values = torch.FloatTensor([[item[1]] for item in batch]).to(device)
                
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
    
    def evaluate_agent_vs_baseline(self, agent, baseline_agent=None, num_games: int = 10) -> Dict:
        """
        Evaluate an agent against a baseline (default: random agent).
        
        Args:
            agent: Agent to evaluate (can be PolicyAgent, ValueAgent, etc.)
            baseline_agent: Baseline agent (default: RandomAgent)
            num_games: Number of games to play
            
        Returns:
            Dictionary with win rate statistics
        """
        print(f"Evaluating agent vs baseline over {num_games} games...")
        
        if baseline_agent is None:
            from agents.random_agent import RandomAgent
            baseline_agent = RandomAgent("Random Baseline")
        
        # Play games
        wins = 0
        losses = 0
        ties = 0
        
        for game_num in range(num_games):
            # Alternate who goes first
            if game_num % 2 == 0:
                player1, player2 = agent, baseline_agent
            else:
                player1, player2 = baseline_agent, agent
            
            winner = self._play_single_evaluation_game(player1, player2)
            
            # Count results from agent's perspective
            if winner == 0:
                ties += 1
            elif (winner == 1 and player1 == agent) or (winner == -1 and player2 == agent):
                wins += 1
            else:
                losses += 1
        
        win_rate = wins / num_games
        
        results = {
            'wins': wins,
            'losses': losses,
            'ties': ties,
            'total_games': num_games,
            'win_rate': win_rate,
            'agent_name': agent.name,
            'baseline_name': baseline_agent.name
        }
        
        print(f"Results: {wins}W-{losses}L-{ties}T (Win rate: {win_rate:.2%})")
        return results
    
    def _play_single_evaluation_game(self, player1, player2):
        """Play a single game for evaluation purposes."""
        try:
            from connect4_env import GymnasiumConnectFour
            env = GymnasiumConnectFour()
            env.reset()
            
            terminated = False
            max_moves = 42
            move_count = 0
            
            while not terminated and move_count < max_moves:
                # Get current player
                current_agent = player1 if env.current_player == 1 else player2
                
                # Get action using the unified interface
                try:
                    # Try neural network interface first
                    if hasattr(current_agent, 'set_player_id'):
                        current_agent.set_player_id(env.current_player)
                    
                    board = env.board.copy()
                    action_mask = env._action_mask()
                    action = current_agent.select_action(board, action_mask)
                except TypeError:
                    # Fallback to environment interface
                    action = current_agent.select_action(env)
                
                # Make move
                step_result = env.step(action)
                if len(step_result) == 6:
                    _, reward, terminated, _, info, _ = step_result
                elif len(step_result) == 5:
                    _, reward, terminated, _, info = step_result
                else:
                    _, reward, terminated, info = step_result
                
                move_count += 1
                
                # Check for winner
                if terminated:
                    if 'winner' in info:
                        return info['winner']
                    elif reward != 0:
                        return env.current_player * -1  # Previous player won
                    else:
                        return 0  # Tie
            
            return 0  # Tie if max moves reached
            
        except Exception as e:
            print(f"Error in evaluation game: {e}")
            return 0  # Treat as tie
    
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
    
    def generate_minimax_training_data(self, num_games: int = 100, minimax_depth: int = 4) -> Tuple[List, List]:
        """
        Generate training data using minimax agent vs random agent games.
        
        Args:
            num_games: Number of games to simulate
            minimax_depth: Search depth for minimax agent
            
        Returns:
            Tuple of (policy_data, value_data) for training
        """
        # Generate game records
        game_records = generate_minimax_training_data(num_games, minimax_depth)
        
        # Convert to training data
        policy_data, value_data = self.generate_training_data_from_games(game_records)
        
        print(f"Generated {len(policy_data)} training examples from {num_games} games")
        
        return policy_data, value_data


def generate_minimax_training_data(num_games: int = 100, minimax_depth: int = 4) -> List[Dict]:
    """
    Generate training data by playing games between minimax and random agents.
    
    Args:
        num_games: Number of games to play
        minimax_depth: Depth for minimax search
        
    Returns:
        List of game records with board states, actions, and outcomes
    """
    print(f"Generating training data from {num_games} games (Minimax depth={minimax_depth})...")
    
    minimax_agent = MinimaxAgent(depth=minimax_depth)
    random_agent = RandomAgent()
    
    game_records = []
    
    for game_num in range(num_games):
        if (game_num + 1) % 20 == 0:
            print(f"Playing game {game_num + 1}/{num_games}")
        
        env = GymnasiumConnectFour()
        env.reset()
        
        # Randomly decide which agent goes first
        minimax_is_player1 = random.choice([True, False])
        
        board_states = []
        actions = []
        
        terminated = False
        winner = 0
        
        while not terminated:
            current_board = env.board.copy()
            
            # Choose action based on current player
            if (env.current_player == 1 and minimax_is_player1) or \
               (env.current_player == -1 and not minimax_is_player1):
                # Minimax agent's turn
                action = minimax_agent.select_action(env)
            else:
                # Random agent's turn
                action = random_agent.select_action(env)
            
            # Store the board state and action
            board_states.append(current_board)
            actions.append(action)
            
            # Make the move
            _, reward, terminated, _, info, _ = env.step(action)
            
            # Check game outcome
            if terminated:
                if info.get("reason") == "Win":
                    winner = info.get("winner", 0)
                elif info.get("reason") == "Tie":
                    winner = 0
                else:  # Illegal move (shouldn't happen with proper agents)
                    winner = 0
        
        # Store game record
        game_record = {
            'board_states': board_states,
            'actions': actions,
            'winner': winner,
            'minimax_is_player1': minimax_is_player1,
            'num_moves': len(actions)
        }
        game_records.append(game_record)
    
    print(f"Generated {len(game_records)} games")
    
    # Print some statistics
    wins_as_player1 = sum(1 for g in game_records if g['winner'] == 1)
    wins_as_player2 = sum(1 for g in game_records if g['winner'] == -1)
    ties = sum(1 for g in game_records if g['winner'] == 0)
    
    minimax_wins = 0
    for game in game_records:
        if game['winner'] == 1 and game['minimax_is_player1']:
            minimax_wins += 1
        elif game['winner'] == -1 and not game['minimax_is_player1']:
            minimax_wins += 1
    
    print(f"Results: Player 1 wins: {wins_as_player1}, Player -1 wins: {wins_as_player2}, Ties: {ties}")
    print(f"Minimax wins: {minimax_wins}/{num_games} ({100*minimax_wins/num_games:.1f}%)")
    
    return game_records


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