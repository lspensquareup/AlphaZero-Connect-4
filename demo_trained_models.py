"""
Example script showing how to use trained networks for gameplay.

This demonstrates loading saved models and using them to play Connect-4.
"""

import os
import sys
import torch
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from networks.policy_network import PolicyNetwork
from networks.value_network import ValueNetwork
from connect4_env import GymnasiumConnectFour
from agents.random_agent import RandomAgent


class TrainedPolicyAgent:
    """Agent that uses a trained neural network to select actions."""
    
    def __init__(self, policy_network, name="TrainedPolicy"):
        self.policy_network = policy_network
        self.name = name
        policy_network.eval()  # Set to evaluation mode
    
    def select_action(self, env):
        """Select action using trained policy network."""
        with torch.no_grad():
            board_tensor = torch.FloatTensor(env.board).unsqueeze(0)
            policy_logits = self.policy_network(board_tensor)
            
            # Apply action mask to prevent illegal moves
            action_mask = env._action_mask()
            masked_logits = policy_logits.clone()
            masked_logits[0, action_mask == 0] = float('-inf')
            
            # Use softmax to get probabilities and sample
            probs = torch.softmax(masked_logits, dim=1)
            action = torch.multinomial(probs, 1).item()
            
            return action


def load_trained_models(model_dir="models"):
    """Load the most recent trained models."""
    policy_path = os.path.join(model_dir, "policy_network_epoch_20.pth")
    value_path = os.path.join(model_dir, "value_network_epoch_20.pth")
    
    if not os.path.exists(policy_path):
        raise FileNotFoundError(f"Policy network not found at {policy_path}")
    if not os.path.exists(value_path):
        raise FileNotFoundError(f"Value network not found at {value_path}")
    
    # Create networks and load weights
    policy_net = PolicyNetwork()
    value_net = ValueNetwork()
    
    policy_net.load_state_dict(torch.load(policy_path, map_location='cpu'))
    value_net.load_state_dict(torch.load(value_path, map_location='cpu'))
    
    print(f"Loaded policy network from {policy_path}")
    print(f"Loaded value network from {value_path}")
    
    return policy_net, value_net


def play_game_with_analysis(agent1, agent2, value_net):
    """Play a game between two agents with position analysis."""
    env = GymnasiumConnectFour()
    env.reset()
    
    print(f"\n=== Game: {agent1.name} vs {agent2.name} ===")
    print("Initial position:")
    env.render()
    
    move_count = 0
    agents = [agent1, agent2]
    
    while True:
        current_agent = agents[move_count % 2]
        
        # Show position evaluation
        with torch.no_grad():
            board_tensor = torch.FloatTensor(env.board).unsqueeze(0)
            position_value = value_net(board_tensor).item()
        
        print(f"\nMove {move_count + 1}: {current_agent.name} to play")
        print(f"Position evaluation: {position_value:.3f} (from current player's perspective)")
        
        # Get action
        action = current_agent.select_action(env)
        print(f"Plays column {action}")
        
        # Make move
        _, reward, terminated, _, info, _ = env.step(action)
        env.render()
        
        if terminated:
            if info.get("reason") == "Win":
                winner = info.get("winner")
                winning_agent = agent1.name if winner == 1 else agent2.name
                print(f"\nGame Over! {winning_agent} wins!")
            elif info.get("reason") == "Tie":
                print("\nGame Over! It's a tie!")
            break
        
        move_count += 1
        if move_count > 42:  # Safety break
            print("\nGame stopped after 42 moves")
            break


def demonstrate_network_capabilities():
    """Demonstrate what the trained networks have learned."""
    print("=== Demonstrating Trained Network Capabilities ===\n")
    
    try:
        policy_net, value_net = load_trained_models()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run 'python train_with_minimax.py' first to generate trained models.")
        return
    
    # Create agents
    trained_agent = TrainedPolicyAgent(policy_net, "TrainedPolicy")
    random_agent = RandomAgent()
    random_agent.name = "Random"
    
    # Test on different board positions
    print("1. Testing network responses to different positions:\n")
    
    test_positions = [
        (np.zeros((6, 7)), "Empty board"),
        (create_test_position_1(), "Early game position"),
        (create_test_position_2(), "Mid-game position"),
    ]
    
    for board, description in test_positions:
        print(f"{description}:")
        env = GymnasiumConnectFour()
        env.board = board.copy()
        env.current_player = 1
        
        # Show policy preferences
        with torch.no_grad():
            board_tensor = torch.FloatTensor(board).unsqueeze(0)
            policy_logits = policy_net(board_tensor)
            probs = torch.softmax(policy_logits, dim=1).squeeze().numpy()
            position_value = value_net(board_tensor).item()
        
        print(f"  Position value: {position_value:.3f}")
        print(f"  Action preferences: {[f'{i}:{p:.3f}' for i, p in enumerate(probs)]}")
        print(f"  Preferred action: column {np.argmax(probs)}")
        print()
    
    # Play sample games
    print("2. Playing sample games:\n")
    
    # Game 1: Trained vs Random
    play_game_with_analysis(trained_agent, random_agent, value_net)
    
    # Game 2: Random vs Trained  
    play_game_with_analysis(random_agent, trained_agent, value_net)


def create_test_position_1():
    """Create an early game test position."""
    board = np.zeros((6, 7))
    board[5, 3] = 1  # Player 1 in center
    board[5, 2] = -1  # Player -1 adjacent
    return board


def create_test_position_2():
    """Create a mid-game test position."""
    board = np.zeros((6, 7))
    # Some pieces on the board
    board[5, 3] = 1
    board[4, 3] = -1
    board[5, 4] = 1
    board[5, 2] = -1
    board[4, 2] = 1
    return board


if __name__ == "__main__":
    print("AlphaZero Connect-4: Trained Model Demonstration\n")
    demonstrate_network_capabilities()