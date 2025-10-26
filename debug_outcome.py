#!/usr/bin/env python3
"""
Debug the winner interpretation issue.
"""

from connect4_env import GymnasiumConnectFour
from agents.random_agent import RandomAgent
from agents.minimax_agent import MinimaxAgent

def debug_game_outcome():
    print("🔍 Debugging Game Outcome Logic")
    
    # Test a few games and see what winners are returned
    env = GymnasiumConnectFour()
    random_agent = RandomAgent("Random")
    minimax_agent = MinimaxAgent(depth=2)
    
    for game_num in range(5):
        print(f"\n=== Game {game_num + 1} ===")
        env.reset()
        
        # Alternate who goes first
        if game_num % 2 == 0:
            player1, player2 = random_agent, minimax_agent
            print("Player 1 (Random) vs Player 2 (Minimax)")
        else:
            player1, player2 = minimax_agent, random_agent  
            print("Player 1 (Minimax) vs Player 2 (Random)")
        
        terminated = False
        move_count = 0
        max_moves = 42
        
        while not terminated and move_count < max_moves:
            # Get current player
            current_agent = player1 if env.current_player == 1 else player2
            
            # Get action
            try:
                if hasattr(current_agent, 'set_player_id'):
                    current_agent.set_player_id(env.current_player)
                
                board = env.board.copy()
                action_mask = env._action_mask()
                action = current_agent.select_action(board, action_mask)
            except TypeError:
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
                print(f"Game ended after {move_count} moves")
                print(f"Current player when terminated: {env.current_player}")
                print(f"Reward: {reward}")
                print(f"Info: {info}")
                
                if 'winner' in info:
                    winner = info['winner']
                    print(f"Winner from info: {winner}")
                elif reward != 0:
                    winner = env.current_player * -1  # Previous player won
                    print(f"Winner calculated: {winner} (previous player)")
                else:
                    winner = 0  # Tie
                    print("Tie game")
                
                # Determine who actually won
                if winner == 1:
                    actual_winner = "Player 1"
                    if player1 == random_agent:
                        actual_winner += " (Random)"
                    else:
                        actual_winner += " (Minimax)"
                elif winner == -1:
                    actual_winner = "Player 2"
                    if player2 == random_agent:
                        actual_winner += " (Random)"
                    else:
                        actual_winner += " (Minimax)"
                else:
                    actual_winner = "Tie"
                
                print(f"Actual winner: {actual_winner}")
                break
        
        if not terminated:
            print("Game reached max moves - tie")

if __name__ == "__main__":
    debug_game_outcome()