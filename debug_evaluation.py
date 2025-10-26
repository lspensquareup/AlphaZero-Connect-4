#!/usr/bin/env python3
"""
Debug the evaluation game winner detection logic.
"""

from connect4_env import GymnasiumConnectFour
from agents.random_agent import RandomAgent
from agents.minimax_agent import MinimaxAgent

def debug_evaluation_game():
    """Debug a single evaluation game to see winner detection."""
    from training.trainer import Trainer
    
    # Create trainer and agents
    trainer = Trainer(save_dir='./debug_models')
    random_agent = RandomAgent("Random")
    minimax_agent = MinimaxAgent(depth=2, name="Minimax")
    
    print("🔍 Testing evaluation game winner detection...")
    
    # Test 10 games
    for i in range(10):
        print(f"\n=== Game {i+1} ===")
        
        # Alternate player order
        if i % 2 == 0:
            player1, player2 = random_agent, minimax_agent
            print(f"Player 1: {player1.name}, Player 2: {player2.name}")
        else:
            player1, player2 = minimax_agent, random_agent
            print(f"Player 1: {player1.name}, Player 2: {player2.name}")
        
        # Play the game
        winner = trainer._play_single_evaluation_game(player1, player2)
        print(f"Returned winner: {winner}")
        
        if winner == 1:
            print(f"Winner: Player 1 ({player1.name})")
        elif winner == -1:
            print(f"Winner: Player 2 ({player2.name})")
        else:
            print("Result: Tie")

def debug_single_game_step_by_step():
    """Debug a single game step by step to see the winner detection."""
    env = GymnasiumConnectFour()
    env.reset()
    
    agent1 = RandomAgent("Random1")
    agent2 = MinimaxAgent(depth=2, name="Minimax2")
    
    print("🔍 Step-by-step game debug...")
    print(f"Initial current_player: {env.current_player}")
    
    terminated = False
    max_moves = 42
    move_count = 0
    
    while not terminated and move_count < max_moves:
        print(f"\nMove {move_count + 1}:")
        print(f"Current player: {env.current_player}")
        
        # Get current agent
        current_agent = agent1 if env.current_player == 1 else agent2
        print(f"Agent making move: {current_agent.name}")
        
        # Get action
        action = current_agent.select_action(env)
        print(f"Action selected: {action}")
        
        # Make move
        step_result = env.step(action)
        if len(step_result) == 6:
            board, reward, terminated, _, info, _ = step_result
        elif len(step_result) == 5:
            board, reward, terminated, _, info = step_result
        else:
            board, reward, terminated, info = step_result
        
        print(f"After move - Current player: {env.current_player}")
        print(f"Reward: {reward}, Terminated: {terminated}")
        print(f"Info: {info}")
        
        move_count += 1
        
        if terminated:
            print(f"\n🏁 Game ended!")
            print(f"Final current_player: {env.current_player}")
            
            if 'winner' in info:
                winner = info['winner']
                print(f"Winner from info: {winner}")
                if winner == 1:
                    print(f"Winner: Player 1 ({agent1.name})")
                elif winner == -1:
                    print(f"Winner: Player 2 ({agent2.name})")
            elif reward != 0:
                # This is the logic from trainer.py
                calculated_winner = env.current_player * -1
                print(f"No winner in info, calculated winner: {calculated_winner}")
            else:
                print("Tie game")
            break
    
    if not terminated:
        print("Game ended due to max moves (tie)")

if __name__ == "__main__":
    debug_evaluation_game()
    print("\n" + "="*50)
    debug_single_game_step_by_step()