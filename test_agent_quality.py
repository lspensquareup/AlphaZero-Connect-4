#!/usr/bin/env python3
"""
Test Random vs Minimax directly to see if the issue is in the evaluation system.
"""

from connect4_env import GymnasiumConnectFour
from agents.random_agent import RandomAgent
from agents.minimax_agent import MinimaxAgent

def test_random_vs_minimax(num_games=20):
    """Test Random vs Minimax directly."""
    print(f"🔍 Testing {num_games} games: Random vs Minimax")
    
    random_agent = RandomAgent("Random")
    minimax_agent = MinimaxAgent(depth=3, name="Minimax")
    
    random_wins_as_p1 = 0
    random_wins_as_p2 = 0
    minimax_wins_as_p1 = 0
    minimax_wins_as_p2 = 0
    ties = 0
    
    for game in range(num_games):
        env = GymnasiumConnectFour()
        env.reset()
        
        # Alternate who goes first
        if game % 2 == 0:
            player1, player2 = random_agent, minimax_agent
            player1_name, player2_name = "Random", "Minimax"
        else:
            player1, player2 = minimax_agent, random_agent
            player1_name, player2_name = "Minimax", "Random"
        
        terminated = False
        max_moves = 42
        move_count = 0
        
        while not terminated and move_count < max_moves:
            current_agent = player1 if env.current_player == 1 else player2
            action = current_agent.select_action(env)
            
            step_result = env.step(action)
            if len(step_result) == 6:
                _, reward, terminated, _, info, _ = step_result
            elif len(step_result) == 5:
                _, reward, terminated, _, info = step_result
            else:
                _, reward, terminated, info = step_result
            
            move_count += 1
            
            if terminated:
                if 'winner' in info:
                    winner = info['winner']
                    if winner == 1:
                        winner_name = player1_name
                        if player1_name == "Random":
                            random_wins_as_p1 += 1
                        else:
                            minimax_wins_as_p1 += 1
                    elif winner == -1:
                        winner_name = player2_name
                        if player2_name == "Random":
                            random_wins_as_p2 += 1
                        else:
                            minimax_wins_as_p2 += 1
                    else:
                        winner_name = "Tie"
                        ties += 1
                else:
                    winner_name = "Tie"
                    ties += 1
                
                print(f"Game {game+1}: P1={player1_name}, P2={player2_name}, Winner={winner_name} ({move_count} moves)")
                break
        
        if not terminated:
            ties += 1
            print(f"Game {game+1}: P1={player1_name}, P2={player2_name}, Winner=Tie (max moves)")
    
    print(f"\n📊 Results:")
    print(f"Random wins as Player 1: {random_wins_as_p1}")
    print(f"Random wins as Player 2: {random_wins_as_p2}")
    print(f"Minimax wins as Player 1: {minimax_wins_as_p1}")
    print(f"Minimax wins as Player 2: {minimax_wins_as_p2}")
    print(f"Ties: {ties}")
    
    total_random_wins = random_wins_as_p1 + random_wins_as_p2
    total_minimax_wins = minimax_wins_as_p1 + minimax_wins_as_p2
    
    print(f"\n🎯 Summary:")
    print(f"Random total wins: {total_random_wins} ({total_random_wins/num_games*100:.1f}%)")
    print(f"Minimax total wins: {total_minimax_wins} ({total_minimax_wins/num_games*100:.1f}%)")
    print(f"Ties: {ties} ({ties/num_games*100:.1f}%)")
    
    # Expected: Minimax should win significantly more than Random
    if total_minimax_wins < total_random_wins:
        print("🚨 UNEXPECTED: Random is beating Minimax!")
    elif total_minimax_wins == total_random_wins:
        print("🤔 UNEXPECTED: Random and Minimax are tied!")
    else:
        print("✅ EXPECTED: Minimax is beating Random")

if __name__ == "__main__":
    test_random_vs_minimax(20)