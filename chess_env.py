import gymnasium as gym
import numpy as np
import chess
from gymnasium import spaces

class GymChess(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(64*64*5)
        self.observation_space = spaces.Box(low=0, high=1, shape=(8, 8, 12), dtype=np.int8)
        self.board = chess.Board()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board.reset()
        obs = self._board_to_obs()
        info = {"action_mask": self._legal_action_mask()}
        return obs, info

    def step(self, action: int):
        move = self._decode_action(action)
        info = {}

        # Illegal move --> terminate with penalty
        if move not in self.board.legal_moves:
            return self._board_to_obs(), -1, True, False, {"illegal_action": True}
        
        # Apply move
        self.board.push(move)

        # Check terminal conditions
        terminated = self.board.is_game_over()
        truncated = False
        reward = 0
        if terminated:
            result = self.board.result() # 1-0, 0-1, 1/2-1/2
            if result = "1-0":
                reward = 1 if self.board.turn == chess.WHITE else -1
            elif result = "0-1":
                reward = -1 if self.board.turn == chess.WHITE else 1
            else:
                reward = 0
        obs = self.board_to_obs()
        info["action_mask"] = self._legal_action_mask()
        return obs, reward, terminated, truncated, info
    
    def render(self, mode="human"):
        print(self.board.unicode(borders=True))
        print()