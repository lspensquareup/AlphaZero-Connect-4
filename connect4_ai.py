"""
Rule-based AI agents for Connect-4.

This module contains various rule-based AI strategies for Connect-4,
which can be used as baselines and opponents for neural network training.
"""

import numpy as np
from agents.base_agent import BaseAgent


class MinimaxAgent(BaseAgent):
    """
    Minimax agent with alpha-beta pruning.
    
    This agent uses the minimax algorithm to search for the best move
    by looking ahead several moves and evaluating positions.
    """
    
    def __init__(self, name: str = "Minimax AI", depth: int = 5):
        """
        Initialize the minimax agent.
        
        Args:
            name: Human-readable name for this agent
            depth: Maximum search depth
        """
        super().__init__(name)
        self.depth = depth
    
    def select_action(self, board: np.ndarray, action_mask: np.ndarray) -> int:
        """Select action using minimax with alpha-beta pruning."""
        valid_actions = np.where(action_mask == 1)[0]
        
        if len(valid_actions) == 1:
            return valid_actions[0]
        
        best_action = valid_actions[0]
        best_value = -float('inf')
        
        for action in valid_actions:
            # Simulate the move
            new_board = board.copy()
            row = self._drop_piece(new_board, action, self.player_id)
            
            if row is None:
                continue
            
            # Check for immediate win
            if self._check_win(new_board, row, action):
                return action
            
            # Evaluate using minimax
            value = self._minimax(new_board, self.depth - 1, False, -float('inf'), float('inf'))
            
            if value > best_value:
                best_value = value
                best_action = action
        
        return best_action
    
    def _minimax(self, board: np.ndarray, depth: int, is_maximizing: bool, 
                 alpha: float, beta: float) -> float:
        """Minimax algorithm with alpha-beta pruning."""
        
        # Check terminal states
        winner = self._check_terminal(board)
        if winner is not None:
            if winner == self.player_id:
                return 1000 + depth  # Prefer quicker wins
            elif winner == -self.player_id:
                return -1000 - depth  # Avoid quicker losses
            else:
                return 0  # Tie
        
        if depth == 0:
            return self._evaluate_board(board)
        
        valid_actions = self._get_valid_actions(board)
        
        if is_maximizing:
            max_eval = -float('inf')
            for action in valid_actions:
                new_board = board.copy()
                row = self._drop_piece(new_board, action, self.player_id)
                if row is not None:
                    eval_score = self._minimax(new_board, depth - 1, False, alpha, beta)
                    max_eval = max(max_eval, eval_score)
                    alpha = max(alpha, eval_score)
                    if beta <= alpha:
                        break  # Alpha-beta pruning
            return max_eval
        else:
            min_eval = float('inf')
            for action in valid_actions:
                new_board = board.copy()
                row = self._drop_piece(new_board, action, -self.player_id)
                if row is not None:
                    eval_score = self._minimax(new_board, depth - 1, True, alpha, beta)
                    min_eval = min(min_eval, eval_score)
                    beta = min(beta, eval_score)
                    if beta <= alpha:
                        break  # Alpha-beta pruning
            return min_eval
    
    def _evaluate_board(self, board: np.ndarray) -> float:
        """Evaluate board position heuristically."""
        score = 0
        
        # Center column preference
        center_array = [int(i) for i in list(board[:, 3])]
        center_count = center_array.count(self.player_id)
        score += center_count * 3
        
        # Evaluate all possible 4-windows
        score += self._evaluate_windows(board, self.player_id)
        score -= self._evaluate_windows(board, -self.player_id)
        
        return score
    
    def _evaluate_windows(self, board: np.ndarray, player: int) -> int:
        """Evaluate all possible 4-piece windows."""
        score = 0
        
        # Horizontal
        for r in range(6):
            for c in range(4):
                window = board[r, c:c+4]
                score += self._evaluate_window(window, player)
        
        # Vertical
        for r in range(3):
            for c in range(7):
                window = board[r:r+4, c]
                score += self._evaluate_window(window, player)
        
        # Positive diagonal
        for r in range(3):
            for c in range(4):
                window = [board[r+i, c+i] for i in range(4)]
                score += self._evaluate_window(window, player)
        
        # Negative diagonal
        for r in range(3):
            for c in range(4):
                window = [board[r+3-i, c+i] for i in range(4)]
                score += self._evaluate_window(window, player)
        
        return score
    
    def _evaluate_window(self, window, player: int) -> int:
        """Evaluate a 4-piece window."""
        score = 0
        opp_player = -player
        
        if window.count(player) == 4:
            score += 100
        elif window.count(player) == 3 and window.count(0) == 1:
            score += 10
        elif window.count(player) == 2 and window.count(0) == 2:
            score += 2
        
        if window.count(opp_player) == 3 and window.count(0) == 1:
            score -= 80  # Block opponent
        
        return score
    
    def _drop_piece(self, board: np.ndarray, col: int, player: int):
        """Drop a piece and return the row index."""
        if board[0, col] != 0:
            return None
        
        for row in range(5, -1, -1):
            if board[row, col] == 0:
                board[row, col] = player
                return row
        return None
    
    def _check_win(self, board: np.ndarray, row: int, col: int) -> bool:
        """Check if the last move created a win."""
        player = board[row, col]
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dr, dc in directions:
            count = 1
            # Check positive direction
            r, c = row + dr, col + dc
            while 0 <= r < 6 and 0 <= c < 7 and board[r, c] == player:
                count += 1
                r, c = r + dr, c + dc
            
            # Check negative direction
            r, c = row - dr, col - dc
            while 0 <= r < 6 and 0 <= c < 7 and board[r, c] == player:
                count += 1
                r, c = r - dr, c - dc
            
            if count >= 4:
                return True
        
        return False
    
    def _check_terminal(self, board: np.ndarray):
        """Check if the game is over and return winner."""
        # Check for wins
        for row in range(6):
            for col in range(7):
                if board[row, col] != 0:
                    if self._check_win(board, row, col):
                        return board[row, col]
        
        # Check for tie
        if np.all(board[0, :] != 0):
            return 0
        
        return None
    
    def _get_valid_actions(self, board: np.ndarray) -> list:
        """Get list of valid column indices."""
        return [c for c in range(7) if board[0, c] == 0]


class GreedyAgent(BaseAgent):
    """
    Greedy agent that looks one move ahead.
    
    This agent prioritizes winning moves, then blocking opponent wins,
    then uses simple heuristics.
    """
    
    def __init__(self, name: str = "Greedy AI"):
        super().__init__(name)
    
    def select_action(self, board: np.ndarray, action_mask: np.ndarray) -> int:
        """Select action using greedy strategy."""
        valid_actions = np.where(action_mask == 1)[0]
        
        # 1. Check for winning move
        for action in valid_actions:
            test_board = board.copy()
            row = self._drop_piece(test_board, action, self.player_id)
            if row is not None and self._check_win(test_board, row, action):
                return action
        
        # 2. Block opponent's winning move
        for action in valid_actions:
            test_board = board.copy()
            row = self._drop_piece(test_board, action, -self.player_id)
            if row is not None and self._check_win(test_board, row, action):
                return action
        
        # 3. Prefer center columns
        center_actions = [a for a in valid_actions if a in [2, 3, 4]]
        if center_actions:
            return np.random.choice(center_actions)
        
        # 4. Random valid move
        return np.random.choice(valid_actions)
    
    def _drop_piece(self, board: np.ndarray, col: int, player: int):
        """Drop a piece and return the row index."""
        if board[0, col] != 0:
            return None
        
        for row in range(5, -1, -1):
            if board[row, col] == 0:
                board[row, col] = player
                return row
        return None
    
    def _check_win(self, board: np.ndarray, row: int, col: int) -> bool:
        """Check if the last move created a win."""
        player = board[row, col]
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dr, dc in directions:
            count = 1
            # Check positive direction
            r, c = row + dr, col + dc
            while 0 <= r < 6 and 0 <= c < 7 and board[r, c] == player:
                count += 1
                r, c = r + dr, c + dc
            
            # Check negative direction
            r, c = row - dr, col - dc
            while 0 <= r < 6 and 0 <= c < 7 and board[r, c] == player:
                count += 1
                r, c = r - dr, c - dc
            
            if count >= 4:
                return True
        
        return False


class CenterAgent(BaseAgent):
    """
    Simple agent that prefers center columns.
    
    This agent always tries to play in the center columns first,
    which is a reasonable strategy for Connect-4.
    """
    
    def __init__(self, name: str = "Center AI"):
        super().__init__(name)
    
    def select_action(self, board: np.ndarray, action_mask: np.ndarray) -> int:
        """Select action preferring center columns."""
        valid_actions = np.where(action_mask == 1)[0]
        
        # Preference order: center to edges
        preference_order = [3, 2, 4, 1, 5, 0, 6]
        
        for col in preference_order:
            if col in valid_actions:
                return col
        
        # Fallback (shouldn't happen)
        return valid_actions[0] if len(valid_actions) > 0 else 0