"""
Test script for the training visualization system.

This script demonstrates:
1. Live game visualization with agents
2. Training analytics dashboard
3. Performance metrics collection
"""

import sys
import os
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.minimax_agent import MinimaxAgent
from agents.random_agent import RandomAgent
from training_game_visualizer import GameSessionManager
from training.trainer import generate_minimax_training_data


def test_game_visualization():
    """Test the game visualization system."""
    print("=== Testing Game Visualization ===\n")
    
    # Create agents
    minimax_agent = MinimaxAgent(depth=3, name="Minimax-Depth-3")
    random_agent = RandomAgent()
    random_agent.name = "RandomPlayer"
    
    print(f"Created agents: {minimax_agent.name} vs {random_agent.name}")
    
    # Create session manager
    session = GameSessionManager()
    
    # Option to play visualized game
    print("\nOptions:")
    print("1. Play one visualized game (GUI)")
    print("2. Generate training data (no GUI)")
    print("3. Show session statistics")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        print("\nStarting visualized game...")
        print("This will open a GUI window showing the game in real-time.")
        
        result = session.play_visualized_game(
            minimax_agent, 
            random_agent, 
            auto_close=False,  # Keep window open
            speed=2.0  # 2x speed
        )
        
        print("\nGame Result:")
        print(f"Winner: Player {result.get('winner', 'Tie')}")
        print(f"Total moves: {result['total_moves']}")
        print(f"Game duration: {result['total_time']:.2f} seconds")
        
    elif choice == "2":
        print("\nGenerating training data (no visualization)...")
        
        start_time = time.time()
        game_records = generate_minimax_training_data(num_games=5, minimax_depth=3)
        generation_time = time.time() - start_time
        
        # Analyze results
        minimax_wins = sum(1 for g in game_records 
                          if (g['winner'] == 1 and g['minimax_is_player1']) or 
                             (g['winner'] == -1 and not g['minimax_is_player1']))
        
        print(f"\nTraining Data Generated:")
        print(f"Games: {len(game_records)}")
        print(f"Generation time: {generation_time:.2f} seconds")
        print(f"Minimax wins: {minimax_wins}/{len(game_records)} ({100*minimax_wins/len(game_records):.1f}%)")
        
        # Show move statistics
        total_moves = sum(g['num_moves'] for g in game_records)
        avg_moves = total_moves / len(game_records)
        print(f"Total training examples: {total_moves}")
        print(f"Average moves per game: {avg_moves:.1f}")
        
    elif choice == "3":
        print("\nSession Statistics:")
        print(session.get_session_summary())
    
    else:
        print("Invalid choice. Exiting.")


def demonstrate_analytics():
    """Demonstrate the analytics capabilities."""
    print("\n=== Analytics Demonstration ===\n")
    
    # Generate some sample data
    print("Generating sample training data for analytics...")
    
    game_records = generate_minimax_training_data(num_games=20, minimax_depth=4)
    
    # Analyze the data
    print("\nData Analysis:")
    
    # Game outcomes
    minimax_wins = sum(1 for g in game_records 
                      if (g['winner'] == 1 and g['minimax_is_player1']) or 
                         (g['winner'] == -1 and not g['minimax_is_player1']))
    random_wins = sum(1 for g in game_records 
                     if (g['winner'] == 1 and not g['minimax_is_player1']) or 
                        (g['winner'] == -1 and g['minimax_is_player1']))
    ties = sum(1 for g in game_records if g['winner'] == 0)
    
    print(f"Game Outcomes (20 games):")
    print(f"  Minimax wins: {minimax_wins} ({100*minimax_wins/20:.1f}%)")
    print(f"  Random wins: {random_wins} ({100*random_wins/20:.1f}%)")
    print(f"  Ties: {ties} ({100*ties/20:.1f}%)")
    
    # Game length analysis
    game_lengths = [g['num_moves'] for g in game_records]
    print(f"\nGame Length Analysis:")
    print(f"  Shortest game: {min(game_lengths)} moves")
    print(f"  Longest game: {max(game_lengths)} moves")
    print(f"  Average game: {sum(game_lengths)/len(game_lengths):.1f} moves")
    
    # Training data implications
    total_examples = sum(game_lengths)
    print(f"\nTraining Data Generated:")
    print(f"  Total position examples: {total_examples}")
    print(f"  Examples per game: {total_examples/20:.1f}")
    
    # Strategic insights
    minimax_as_p1_wins = sum(1 for g in game_records if g['winner'] == 1 and g['minimax_is_player1'])
    minimax_as_p2_wins = sum(1 for g in game_records if g['winner'] == -1 and not g['minimax_is_player1'])
    
    minimax_p1_games = sum(1 for g in game_records if g['minimax_is_player1'])
    minimax_p2_games = len(game_records) - minimax_p1_games
    
    print(f"\nStrategic Analysis:")
    if minimax_p1_games > 0:
        p1_winrate = 100 * minimax_as_p1_wins / minimax_p1_games
        print(f"  Minimax as Player 1: {minimax_as_p1_wins}/{minimax_p1_games} wins ({p1_winrate:.1f}%)")
    
    if minimax_p2_games > 0:
        p2_winrate = 100 * minimax_as_p2_wins / minimax_p2_games
        print(f"  Minimax as Player 2: {minimax_as_p2_wins}/{minimax_p2_games} wins ({p2_winrate:.1f}%)")


if __name__ == "__main__":
    print("AlphaZero Connect-4 Visualization Test Suite\n")
    
    print("This test suite demonstrates the visualization capabilities:")
    print("- Live game visualization with agent information")
    print("- Training data generation with statistics")
    print("- Performance analytics and metrics")
    
    try:
        # Test game visualization
        test_game_visualization()
        
        # Demonstrate analytics
        demonstrate_analytics()
        
        print("\n=== Test Suite Complete ===")
        print("\nTo use the full training dashboard, run:")
        print("python training_dashboard.py")
        
    except KeyboardInterrupt:
        print("\nTest suite interrupted by user.")
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()