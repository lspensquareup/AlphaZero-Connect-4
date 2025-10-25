"""
Simple test script for minimax training data generation.

This script tests the core functionality without requiring all dependencies.
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from training.trainer import generate_minimax_training_data
from agents.minimax_agent import MinimaxAgent
from agents.random_agent import RandomAgent
from connect4_env import GymnasiumConnectFour


def test_basic_agents():
    """Test that our agents can play a simple game."""
    print("=== Testing Basic Agent Functionality ===\n")
    
    # Create environment and agents
    env = GymnasiumConnectFour()
    minimax_agent = MinimaxAgent(depth=3)
    random_agent = RandomAgent()
    
    print("Created environment and agents successfully")
    
    # Test single game
    env.reset()
    print(f"Initial board state: Player {env.current_player} to move")
    
    moves_played = 0
    max_moves = 20  # Prevent infinite games
    
    while moves_played < max_moves:
        # Get valid actions
        valid_actions = [i for i in range(7) if env._action_mask()[i] == 1]
        if not valid_actions:
            print("No valid actions available")
            break
        
        # Choose agent based on current player
        if env.current_player == 1:
            action = minimax_agent.select_action(env)
            agent_name = "Minimax"
        else:
            action = random_agent.select_action(env)
            agent_name = "Random"
        
        print(f"Move {moves_played + 1}: {agent_name} (Player {env.current_player}) plays column {action}")
        
        # Make move
        _, reward, terminated, _, info, _ = env.step(action)
        moves_played += 1
        
        if terminated:
            if info.get("reason") == "Win":
                winner = info.get("winner")
                print(f"Game Over! Player {winner} wins!")
            elif info.get("reason") == "Tie":
                print("Game Over! It's a tie!")
            else:
                print(f"Game Over! Reason: {info.get('reason', 'Unknown')}")
            break
    
    if moves_played >= max_moves:
        print("Game stopped after maximum moves")
    
    print("\nSingle game test completed successfully!\n")


def test_minimax_data_generation():
    """Test generating training data from multiple games."""
    print("=== Testing Training Data Generation ===\n")
    
    try:
        # Generate a small amount of training data
        print("Generating training data from 5 games...")
        game_records = generate_minimax_training_data(num_games=5, minimax_depth=3)
        
        print(f"Successfully generated {len(game_records)} game records")
        
        # Analyze the results
        total_moves = sum(len(game['actions']) for game in game_records)
        print(f"Total moves across all games: {total_moves}")
        print(f"Average moves per game: {total_moves / len(game_records):.1f}")
        
        # Count outcomes
        wins_player1 = sum(1 for g in game_records if g['winner'] == 1)
        wins_player2 = sum(1 for g in game_records if g['winner'] == -1) 
        ties = sum(1 for g in game_records if g['winner'] == 0)
        
        print(f"Outcomes: Player 1 wins: {wins_player1}, Player -1 wins: {wins_player2}, Ties: {ties}")
        
        # Count minimax performance
        minimax_wins = 0
        for game in game_records:
            if game['winner'] == 1 and game['minimax_is_player1']:
                minimax_wins += 1
            elif game['winner'] == -1 and not game['minimax_is_player1']:
                minimax_wins += 1
        
        print(f"Minimax wins: {minimax_wins}/{len(game_records)} ({100*minimax_wins/len(game_records):.1f}%)")
        
        print("\nTraining data generation test completed successfully!\n")
        
        return game_records
        
    except Exception as e:
        print(f"Error during training data generation: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("Testing Minimax Implementation for Connect-4 Training Data\n")
    
    try:
        # Test 1: Basic agent functionality
        test_basic_agents()
        
        # Test 2: Training data generation
        game_records = test_minimax_data_generation()
        
        if game_records:
            print("=== All Tests Passed! ===")
            print("The minimax agent is working correctly and can generate training data.")
            print("You can now use this data to train neural networks for AlphaZero.")
        else:
            print("=== Tests Failed ===")
            print("There were issues with the minimax implementation.")
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()