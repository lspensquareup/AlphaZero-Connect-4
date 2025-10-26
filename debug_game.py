#!/usr/bin/env python3
"""
Debug the game playing logic to see why Player 1 always wins.
"""

from connect4_env import GymnasiumConnectFour
from agents.random_agent import RandomAgent
from agents.minimax_agent import MinimaxAgent

def debug_game_playing():
    print("🔍 Debugging Game Playing Logic")
    
    # Test with two identical random agents
    agent1 = RandomAgent("Random1")
    agent2 = RandomAgent("Random2")
    
    env = GymnasiumConnectFour()
    
    for game in range(5):
        print(f"\n=== Game {game+1} ===")
        env.reset()
        print(f"Initial state: Player {env.current_player} to move")
        
        terminated = False
        move_count = 0
        max_moves = 42
        
        move_history = []
        
        while not terminated and move_count < max_moves:
            current_player = env.current_player
            current_agent = agent1 if current_player == 1 else agent2
            agent_name = "Random1" if current_player == 1 else "Random2"
            
            print(f"Move {move_count+1}: Player {current_player} ({agent_name}) to move")
            
            # Get action
            try:
                board = env.board.copy()
                action_mask = env._action_mask()
                action = current_agent.select_action(board, action_mask)
            except TypeError:
                action = current_agent.select_action(env)
            
            print(f"  Action: {action}")
            move_history.append((current_player, action))
            
            # Make move
            step_result = env.step(action)
            if len(step_result) == 6:
                obs, reward, terminated, _, info, _ = step_result
            elif len(step_result) == 5:
                obs, reward, terminated, _, info = step_result
            else:
                obs, reward, terminated, info = step_result
            
            print(f"  Reward: {reward}, Terminated: {terminated}")
            if terminated:
                print(f"  Info: {info}")
            
            move_count += 1
            
            # Print board state every few moves or at end
            if move_count % 5 == 0 or terminated:
                print("  Current board:")
                board = env.board
                for row in board:
                    print("  ", [int(x) for x in row])
            
            if terminated:
                print(f"\nGame ended after {move_count} moves")
                
                # Determine winner
                if 'winner' in info:
                    winner = info['winner']
                    print(f"Winner from info: {winner}")
                elif reward != 0:
                    winner = env.current_player * -1  # Previous player won
                    print(f"Winner calculated from reward: {winner}")
                else:
                    winner = 0
                    print("Tie game")
                
                print(f"Final current_player: {env.current_player}")
                print(f"Move history: {move_history}")
                
                break
        
        if not terminated:
            print("Game reached max moves without termination")

if __name__ == "__main__":
    debug_game_playing()