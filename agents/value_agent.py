"""
Value-based neural network agent for Connect-4.

This agent uses a neural network to evaluate board positions and select moves.
"""

import numpy as np
import torch
from typing import Optional
from .base_agent import BaseAgent


class ValueAgent(BaseAgent):
    """Agent that uses a value network to evaluate positions and select actions."""
    
    def __init__(self, name: str = "Value Agent", network_path: Optional[str] = None):
        """
        Initialize the value agent.
        
        Args:
            name: Human-readable name for this agent
            network_path: Path to saved value network weights (optional)
        """
        super().__init__(name)
        self.network = None
        self.lookahead_depth = 1  # How many moves to look ahead
        
        if network_path:
            self.load_network(network_path)
    
    def load_network(self, network_path: str):
        """Load a trained value network."""
        from ..networks.value_network import ValueNetwork
        
        self.network = ValueNetwork()
        self.network.load_state_dict(torch.load(network_path, map_location='cpu'))
        self.network.eval()
    
    def set_network(self, network):
        """Set the value network directly."""
        self.network = network
        self.network.eval()
    
    def set_lookahead_depth(self, depth: int):
        """Set how many moves to look ahead when evaluating positions."""
        self.lookahead_depth = max(1, depth)
    
    def evaluate_position(self, board: np.ndarray) -> float:
        """
        Evaluate a board position from current player's perspective.
        
        Args:
            board: 6x7 board state
            
        Returns:
            Value estimate (-1 to 1, where 1 is winning for current player)
        """
        if self.network is None:
            # Simple heuristic fallback
            return self._simple_evaluation(board)
        
        # Adjust board perspective for current player
        player_board = board * self.player_id
        board_tensor = torch.FloatTensor(player_board).unsqueeze(0)
        
        with torch.no_grad():
            value = self.network(board_tensor)
            return value.item()
    
    def select_action(self, board: np.ndarray, action_mask: np.ndarray) -> int:
        """
        Select action by evaluating resulting positions.
        
        Args:
            board: 6x7 board state
            action_mask: Valid moves mask
            
        Returns:
            Selected column (0-6)
        """
        valid_actions = np.where(action_mask == 1)[0]
        
        if len(valid_actions) == 1:
            return valid_actions[0]
        
        best_action = valid_actions[0]
        best_value = -float('inf')
        
        # Evaluate each possible move
        for action in valid_actions:
            # Simulate the move
            new_board = board.copy()
            row = self._drop_piece(new_board, action, self.player_id)
            
            if row is None:  # Invalid move (shouldn't happen with proper mask)
                continue
            
            # Check if this move wins immediately
            if self._check_win(new_board, row, action):
                return action  # Take winning move immediately
            
            # Evaluate the resulting position
            value = self._evaluate_position_with_lookahead(new_board, self.lookahead_depth - 1, False)
            
            if value > best_value:
                best_value = value
                best_action = action
        
        return best_action
    
    def _evaluate_position_with_lookahead(self, board: np.ndarray, depth: int, is_maximizing: bool) -> float:
        """Evaluate position with minimax lookahead."""
        if depth == 0:
            return self.evaluate_position(board)
        
        # Get valid moves
        action_mask = self._get_action_mask(board)
        valid_actions = np.where(action_mask == 1)[0]
        
        if len(valid_actions) == 0:  # Board full
            return 0.0
        
        current_player = self.player_id if is_maximizing else -self.player_id
        
        if is_maximizing:
            max_value = -float('inf')
            for action in valid_actions:
                new_board = board.copy()
                row = self._drop_piece(new_board, action, current_player)
                
                if row is not None and self._check_win(new_board, row, action):
                    return 1.0  # Winning position
                
                value = self._evaluate_position_with_lookahead(new_board, depth - 1, False)
                max_value = max(max_value, value)
            return max_value
        else:
            min_value = float('inf')
            for action in valid_actions:
                new_board = board.copy()
                row = self._drop_piece(new_board, action, current_player)
                
                if row is not None and self._check_win(new_board, row, action):
                    return -1.0  # Losing position
                
                value = self._evaluate_position_with_lookahead(new_board, depth - 1, True)
                min_value = min(min_value, value)
            return min_value
    
    def _simple_evaluation(self, board: np.ndarray) -> float:
        """Simple position evaluation heuristic."""
        # Count potential winning lines for each player
        player_score = self._count_threats(board, self.player_id)
        opponent_score = self._count_threats(board, -self.player_id)
        
        # Normalize to [-1, 1] range
        total_score = player_score - opponent_score
        return np.tanh(total_score / 10.0)
    
    def _count_threats(self, board: np.ndarray, player: int) -> int:
        """Count potential winning threats for a player."""
        threats = 0
        for row in range(6):
            for col in range(7):
                if board[row, col] == 0:  # Empty position
                    # Check if placing piece here creates threats
                    test_board = board.copy()
                    test_board[row, col] = player
                    threats += self._count_lines_of_length(test_board, row, col, player, 3)
                    threats += self._count_lines_of_length(test_board, row, col, player, 2) * 0.1
        return threats
    
    def _count_lines_of_length(self, board: np.ndarray, row: int, col: int, player: int, length: int) -> int:
        """Count lines of specified length starting from given position."""
        count = 0
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dr, dc in directions:
            line_count = 1
            # Check positive direction
            r, c = row + dr, col + dc
            while 0 <= r < 6 and 0 <= c < 7 and board[r, c] == player:
                line_count += 1
                r, c = r + dr, c + dc
            
            # Check negative direction
            r, c = row - dr, col - dc
            while 0 <= r < 6 and 0 <= c < 7 and board[r, c] == player:
                line_count += 1
                r, c = r - dr, c - dc
            
            if line_count >= length:
                count += 1
        
        return count
    
    def _drop_piece(self, board: np.ndarray, col: int, player: int) -> Optional[int]:
        """Drop a piece in the specified column. Returns row index or None if invalid."""
        if board[0, col] != 0:  # Column full
            return None
        
        for row in range(5, -1, -1):
            if board[row, col] == 0:
                board[row, col] = player
                return row
        return None
    
    def _check_win(self, board: np.ndarray, row: int, col: int) -> bool:
        """Check if placing a piece at (row, col) creates a win."""
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
    
    def _get_action_mask(self, board: np.ndarray) -> np.ndarray:
        """Get valid action mask for current board."""
        return np.array([int(board[0, c] == 0) for c in range(7)], dtype=np.int8)