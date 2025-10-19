"""
Minimax agent for Connect-4 using alpha-beta pruning.

This agent uses game tree search to find optimal moves,
providing high-quality training data for neural networks.
"""

import sys
import os
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base_agent import BaseAgent
from connect4_env import GymnasiumConnectFour


class MinimaxAgent(BaseAgent):
    """
    Minimax agent with alpha-beta pruning for Connect-4.
    
    This agent uses game tree search to find optimal moves,
    providing high-quality training data for neural networks.
    """
    
    def __init__(self, depth: int = 5, name: str = None):
        """
        Initialize minimax agent.
        
        Args:
            depth: Maximum search depth (higher = stronger but slower)
            name: Agent name (auto-generated if None)
        """
        self.depth = depth
        agent_name = name or f"Minimax(depth={depth})"
        super().__init__(agent_name)
    
    def select_action(self, env: GymnasiumConnectFour) -> int:
        """
        Select the best action using minimax with alpha-beta pruning.
        
        Args:
            env: Connect-4 environment
            
        Returns:
            Best action (column index)
        """
        valid_actions = [i for i in range(7) if env._action_mask()[i] == 1]
        
        if not valid_actions:
            return 0  # Fallback (shouldn't happen)
        
        # If only one valid action, return it immediately
        if len(valid_actions) == 1:
            return valid_actions[0]
        
        best_action = valid_actions[0]
        best_value = float('-inf')
        
        for action in valid_actions:
            # Make a copy of the environment to simulate the move
            env_copy = self._copy_env(env)
            env_copy.step(action)
            
            # Evaluate this move using minimax
            value = self._minimax(env_copy, self.depth - 1, float('-inf'), float('inf'), False)
            
            if value > best_value:
                best_value = value
                best_action = action
        
        return best_action
    
    def _minimax(self, env: GymnasiumConnectFour, depth: int, alpha: float, beta: float, maximizing: bool) -> float:
        """
        Minimax algorithm with alpha-beta pruning.
        
        Args:
            env: Game environment
            depth: Remaining search depth
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            maximizing: True if maximizing player's turn
            
        Returns:
            Evaluated position value
        """
        # Check terminal conditions
        if depth == 0:
            return self._evaluate_position(env)
        
        # Check if game is over
        terminated = self._is_terminal(env)
        if terminated:
            winner = self._get_winner(env)
            if winner == 1:  # Player 1 wins
                return 1000 + depth  # Prefer quicker wins
            elif winner == -1:  # Player -1 wins
                return -1000 - depth  # Avoid quicker losses
            else:  # Tie
                return 0
        
        valid_actions = [i for i in range(7) if env._action_mask()[i] == 1]
        
        if maximizing:
            max_eval = float('-inf')
            for action in valid_actions:
                env_copy = self._copy_env(env)
                env_copy.step(action)
                eval_score = self._minimax(env_copy, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Beta cutoff
            return max_eval
        else:
            min_eval = float('inf')
            for action in valid_actions:
                env_copy = self._copy_env(env)
                env_copy.step(action)
                eval_score = self._minimax(env_copy, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha cutoff
            return min_eval
    
    def _evaluate_position(self, env: GymnasiumConnectFour) -> float:
        """
        Evaluate a board position heuristically.
        
        Args:
            env: Game environment
            
        Returns:
            Position evaluation score
        """
        board = env.board
        score = 0
        
        # Check all possible 4-in-a-row positions
        for row in range(6):
            for col in range(7):
                # Horizontal
                if col <= 3:
                    window = board[row, col:col+4]
                    score += self._evaluate_window(window)
                
                # Vertical
                if row <= 2:
                    window = board[row:row+4, col]
                    score += self._evaluate_window(window)
                
                # Diagonal (positive slope)
                if row <= 2 and col <= 3:
                    window = [board[row+i, col+i] for i in range(4)]
                    score += self._evaluate_window(np.array(window))
                
                # Diagonal (negative slope)
                if row >= 3 and col <= 3:
                    window = [board[row-i, col+i] for i in range(4)]
                    score += self._evaluate_window(np.array(window))
        
        return score
    
    def _evaluate_window(self, window: np.ndarray) -> float:
        """
        Evaluate a 4-piece window.
        
        Args:
            window: Array of 4 board positions
            
        Returns:
            Window evaluation score
        """
        score = 0
        player1_count = np.sum(window == 1)
        player2_count = np.sum(window == -1)
        empty_count = np.sum(window == 0)
        
        # Can't have both players in the same window
        if player1_count > 0 and player2_count > 0:
            return 0
        
        # Score for player 1
        if player1_count == 4:
            score += 100
        elif player1_count == 3 and empty_count == 1:
            score += 10
        elif player1_count == 2 and empty_count == 2:
            score += 2
        
        # Score for player -1
        if player2_count == 4:
            score -= 100
        elif player2_count == 3 and empty_count == 1:
            score -= 10
        elif player2_count == 2 and empty_count == 2:
            score -= 2
        
        return score
    
    def _copy_env(self, env: GymnasiumConnectFour) -> GymnasiumConnectFour:
        """Create a copy of the environment."""
        new_env = GymnasiumConnectFour()
        new_env.board = env.board.copy()
        new_env.current_player = env.current_player
        return new_env
    
    def _is_terminal(self, env: GymnasiumConnectFour) -> bool:
        """Check if the game is in a terminal state."""
        # Check if board is full
        if np.all(env.board != 0):
            return True
        
        # Check for wins (simplified check)
        board = env.board
        for row in range(6):
            for col in range(7):
                if board[row, col] != 0:
                    if env.check_win(row, col):
                        return True
        
        return False
    
    def _get_winner(self, env: GymnasiumConnectFour) -> int:
        """Get the winner of the game (1, -1, or 0 for tie)."""
        board = env.board
        for row in range(6):
            for col in range(7):
                if board[row, col] != 0:
                    if env.check_win(row, col):
                        return board[row, col]
        return 0  # Tie
    
    def get_strategy_info(self) -> dict:
        """Return information about the agent's strategy."""
        return {
            'type': 'minimax',
            'depth': self.depth,
            'uses_alpha_beta_pruning': True,
            'evaluation_function': 'heuristic_position_scoring'
        }