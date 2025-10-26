"""
Pure MCTS (Monte Carlo Tree Search) agent for Connect-4.

This agent uses UCB (Upper Confidence Bound) selection, random expansion,
random rollouts, and backpropagation to find good moves through sampling.
No domain knowledge or heuristics are used - pure statistical search.
"""

import math
import random
import time
import numpy as np
from typing import Optional, List, Tuple, Dict
from .base_agent import BaseAgent


class MCTSNode:
    """A node in the MCTS search tree."""
    
    def __init__(self, board: np.ndarray, parent: Optional['MCTSNode'] = None, 
                 action: Optional[int] = None, player: int = 1):
        self.board = board.copy()
        self.parent = parent
        self.action = action  # Action that led to this node
        self.player = player  # Player who made the move to reach this state
        
        # MCTS statistics
        self.visits = 0
        self.wins = 0.0  # Total reward accumulated
        self.children: Dict[int, 'MCTSNode'] = {}
        self.untried_actions: List[int] = []
        
        # Initialize untried actions (legal moves from this state)
        self._initialize_legal_actions()
    
    def _initialize_legal_actions(self):
        """Initialize the list of untried actions (legal moves)."""
        self.untried_actions = []
        for col in range(7):  # Connect-4 has 7 columns
            if self.board[0, col] == 0:  # Top row is empty
                self.untried_actions.append(col)
    
    def is_fully_expanded(self) -> bool:
        """Check if all legal actions have been tried."""
        return len(self.untried_actions) == 0
    
    def is_terminal(self) -> bool:
        """Check if this is a terminal game state."""
        # Quick check for win conditions or full board
        return self._check_winner() != 0 or np.all(self.board[0, :] != 0)
    
    def _check_winner(self) -> int:
        """Check if there's a winner. Returns 1, -1, or 0."""
        # Check horizontal, vertical, and diagonal wins
        rows, cols = self.board.shape
        
        # Horizontal
        for r in range(rows):
            for c in range(cols - 3):
                if (self.board[r, c] != 0 and 
                    self.board[r, c] == self.board[r, c+1] == 
                    self.board[r, c+2] == self.board[r, c+3]):
                    return self.board[r, c]
        
        # Vertical
        for r in range(rows - 3):
            for c in range(cols):
                if (self.board[r, c] != 0 and 
                    self.board[r, c] == self.board[r+1, c] == 
                    self.board[r+2, c] == self.board[r+3, c]):
                    return self.board[r, c]
        
        # Diagonal (positive slope)
        for r in range(rows - 3):
            for c in range(cols - 3):
                if (self.board[r, c] != 0 and 
                    self.board[r, c] == self.board[r+1, c+1] == 
                    self.board[r+2, c+2] == self.board[r+3, c+3]):
                    return self.board[r, c]
        
        # Diagonal (negative slope)
        for r in range(3, rows):
            for c in range(cols - 3):
                if (self.board[r, c] != 0 and 
                    self.board[r, c] == self.board[r-1, c+1] == 
                    self.board[r-2, c+2] == self.board[r-3, c+3]):
                    return self.board[r, c]
        
        return 0  # No winner
    
    def get_reward(self, player: int) -> float:
        """Get the reward for the given player."""
        winner = self._check_winner()
        if winner == player:
            return 1.0
        elif winner == -player:
            return 0.0
        else:
            return 0.5  # Tie or ongoing game
    
    def ucb_score(self, exploration_param: float = math.sqrt(2)) -> float:
        """Calculate UCB (Upper Confidence Bound) score."""
        if self.visits == 0:
            return float('inf')  # Unvisited nodes have highest priority
        
        exploitation = self.wins / self.visits
        exploration = exploration_param * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration
    
    def select_child(self, exploration_param: float = math.sqrt(2)) -> 'MCTSNode':
        """Select child with highest UCB score."""
        return max(self.children.values(), 
                  key=lambda child: child.ucb_score(exploration_param))
    
    def expand(self) -> 'MCTSNode':
        """Expand the node by adding a new child."""
        if not self.untried_actions:
            raise ValueError("No untried actions available for expansion")
        
        action = self.untried_actions.pop()
        new_board = self._apply_action(action)
        new_player = -self.player  # Switch player
        
        child = MCTSNode(new_board, parent=self, action=action, player=new_player)
        self.children[action] = child
        return child
    
    def _apply_action(self, action: int) -> np.ndarray:
        """Apply an action to the current board and return new board state."""
        new_board = self.board.copy()
        
        # Find the lowest empty row in the column
        for row in range(5, -1, -1):  # Start from bottom
            if new_board[row, action] == 0:
                new_board[row, action] = self.player
                break
        
        return new_board
    
    def backpropagate(self, reward: float):
        """Backpropagate the reward up the tree."""
        self.visits += 1
        self.wins += reward
        if self.parent:
            # Switch perspective for parent (opponent's reward is 1 - our reward)
            self.parent.backpropagate(1.0 - reward)


class MCTSAgent(BaseAgent):
    """Pure MCTS agent using random rollouts and UCB selection."""
    
    def __init__(self, simulations: int = 1000, exploration_param: float = math.sqrt(2),
                 time_limit: Optional[float] = None, name: str = "MCTS Agent"):
        super().__init__(name)
        self.simulations = simulations
        self.exploration_param = exploration_param
        self.time_limit = time_limit  # Optional time limit in seconds
        
        # Statistics for analysis
        self.last_search_stats = {}
    
    def select_action(self, board: np.ndarray, action_mask: np.ndarray) -> int:
        """Select the best action using MCTS."""
        # Convert to the expected format
        current_player = self.player_id
        
        # Create root node
        root = MCTSNode(board, player=current_player)
        
        # If only one legal move, return it immediately
        legal_actions = np.where(action_mask == 1)[0]  # Fix: specify condition and extract first element of tuple
        if len(legal_actions) == 1:
            return int(legal_actions[0])  # Ensure return type is int
        
        # Run MCTS simulations
        start_time = time.time()
        simulations_run = 0
        
        for sim in range(self.simulations):
            # Check time limit
            if self.time_limit and (time.time() - start_time) > self.time_limit:
                break
            
            # Run one MCTS simulation
            self._run_simulation(root, current_player)
            simulations_run += 1
        
        # Select best action based on visit count (most robust)
        best_action = self._select_best_action(root)
        
        # Store statistics for analysis
        self._store_search_stats(root, simulations_run, time.time() - start_time)
        
        return best_action
    
    def _run_simulation(self, root: MCTSNode, root_player: int):
        """Run a single MCTS simulation."""
        node = root
        path = [root]
        
        # 1. Selection: Navigate down to a leaf using UCB
        while not node.is_terminal() and node.is_fully_expanded():
            node = node.select_child(self.exploration_param)
            path.append(node)
        
        # 2. Expansion: Add a new child if possible
        if not node.is_terminal() and not node.is_fully_expanded():
            node = node.expand()
            path.append(node)
        
        # 3. Rollout: Play random moves to terminal state
        reward = self._rollout(node, root_player)
        
        # 4. Backpropagation: Update statistics up the tree
        node.backpropagate(reward)
    
    def _rollout(self, node: MCTSNode, root_player: int) -> float:
        """Perform random rollout from the given node."""
        # Create a copy of the board for simulation
        board = node.board.copy()
        current_player = node.player
        
        # Play random moves until game ends
        while True:
            # Check if game is over
            winner = self._check_winner_fast(board)
            if winner != 0:
                if winner == root_player:
                    return 1.0
                elif winner == -root_player:
                    return 0.0
                else:
                    return 0.5  # Shouldn't happen, but safety
            
            # Check if board is full (tie)
            if np.all(board[0, :] != 0):
                return 0.5  # Tie
            
            # Get legal actions
            legal_actions = []
            for col in range(7):
                if board[0, col] == 0:  # Top row is empty
                    legal_actions.append(col)
            
            if len(legal_actions) == 0:
                return 0.5  # Tie (no legal moves)
            
            # Make random move
            action = random.choice(legal_actions)
            
            # Apply the move
            for row in range(5, -1, -1):  # Start from bottom
                if board[row, action] == 0:
                    board[row, action] = current_player
                    break
            
            # Switch player
            current_player = -current_player
    
    def _check_winner_fast(self, board: np.ndarray) -> int:
        """Fast winner check."""
        rows, cols = board.shape
        
        # Horizontal
        for r in range(rows):
            for c in range(cols - 3):
                if (board[r, c] != 0 and 
                    board[r, c] == board[r, c+1] == board[r, c+2] == board[r, c+3]):
                    return board[r, c]
        
        # Vertical
        for r in range(rows - 3):
            for c in range(cols):
                if (board[r, c] != 0 and 
                    board[r, c] == board[r+1, c] == board[r+2, c] == board[r+3, c]):
                    return board[r, c]
        
        # Diagonal (positive slope)
        for r in range(rows - 3):
            for c in range(cols - 3):
                if (board[r, c] != 0 and 
                    board[r, c] == board[r+1, c+1] == board[r+2, c+2] == board[r+3, c+3]):
                    return board[r, c]
        
        # Diagonal (negative slope)
        for r in range(3, rows):
            for c in range(cols - 3):
                if (board[r, c] != 0 and 
                    board[r, c] == board[r-1, c+1] == board[r-2, c+2] == board[r-3, c+3]):
                    return board[r, c]
        
        return 0
    
    def _select_best_action(self, root: MCTSNode) -> int:
        """Select the best action based on visit count."""
        if not root.children:
            # Fallback to random legal move
            return random.choice(root.untried_actions) if root.untried_actions else 3
        
        # Choose action with most visits (most robust)
        return max(root.children.items(), key=lambda x: x[1].visits)[0]
    
    def _store_search_stats(self, root: MCTSNode, simulations: int, time_taken: float):
        """Store statistics about the search for analysis."""
        self.last_search_stats = {
            'simulations_run': simulations,
            'time_taken': time_taken,
            'simulations_per_second': simulations / time_taken if time_taken > 0 else 0,
            'root_visits': root.visits,
            'children_count': len(root.children),
            'move_stats': {}
        }
        
        # Store per-move statistics
        for action, child in root.children.items():
            win_rate = child.wins / child.visits if child.visits > 0 else 0
            self.last_search_stats['move_stats'][action] = {
                'visits': child.visits,
                'win_rate': win_rate,
                'wins': child.wins
            }
    
    def get_move_analysis(self) -> Dict:
        """Get detailed analysis of the last move selection."""
        return self.last_search_stats.copy()
    
    def __str__(self) -> str:
        return f"MCTS({self.simulations} sims, C={self.exploration_param:.2f})"