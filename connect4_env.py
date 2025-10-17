import gymnasium as gym
import numpy as np
from gymnasium import spaces


class GymnasiumConnectFour(gym.Env):
    def __init__(self):
        super().__init__()
        
        # Define action space (7 columns for Connect Four)
        self.action_space = spaces.Discrete(7)
        
        # Define observation space (6x7 board)
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(6, 7), dtype=np.int8
        )
        
        # Initialize your game state
        self.reset()
    
    def step(self, action):
        # TODO: Implement your transition logic here
        # Must return: (observation, reward, terminated, truncated, info)

        # Check if column is full --> need to check the top row of the chosen column
        if self.board[0, action] != 0:
            return self.board.copy(), -1, True, False, {"reason": "Illegal move", "action_mask": self._action_mask()}, action
        elif self.board[0, action] == 0:
            # Find the lowest empty row in the chosen column
            for i in range(5, -1, -1):
                if self.board[i, action] == 0:
                    self.board[i, action] = self.current_player
                    break
            # Check for a win condition
            if self.check_win(i, action):
                # The current_player just made the winning move
                winner_player = self.current_player
                return self.board.copy(), 1, True, False, {"reason": "Win", "winner": winner_player, "action_mask": self._action_mask()}, action
            # Check for a tie
            if np.all(self.board != 0):
                return self.board.copy(), 0, True, False, {"reason": "Tie", "action_mask": self._action_mask()}, action
            # Switch players
            self.current_player = -self.current_player
            return self.board.copy(), 0, False, False, {"reason": "Valid move", "action_mask": self._action_mask()}, action

    def _action_mask(self) -> np.ndarray:
        # 1 if column top is empty (playable), else 0
        return np.array([int(self.board[0, c] == 0) for c in range(7)], dtype=np.int8)
    
    def reset(self, seed=None, options=None):
        # TODO: Reset environment to initial state
        # Must return: (observation, info)
        self.board = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
        self.current_player = 1
        return self.board.copy(), {"action_mask": self._action_mask()}

    def check_win(self, row, col):
        # Check horizontal, vertical, and diagonal (both directions) for a win
        for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            if self.check_direction(row, col, dr, dc):
                return True
        return False
    
    def check_direction(self, row, col, dr, dc):
        count = 1
        player = self.board[row, col]
        # Check in the positive direction
        r, c = row + dr, col + dc
        while self.in_bound(r, c) and self.board[r, c] == player:
            count += 1
            r, c = r + dr, c + dc
        # Check in the negative direction
        r, c = row - dr, col - dc
        while self.in_bound(r, c) and self.board[r, c] == player:
            count += 1
            r, c = r - dr, c - dc
        return count >= 4

    def in_bound(self, row, col):
        return 0 <= row < 6 and 0 <= col < 7

    def render(self, mode='human'):
        # TODO: Optional visualization
            # Simple ASCII visualization of the board
        print("\nCurrent board:")
        for row in self.board:
            print('|' + '|'.join([' ' if cell == 0 else ('X' if cell == 1 else 'O') for cell in row]) + '|')
        print(' ' + ' '.join(str(i) for i in range(len(self.board[0]))))