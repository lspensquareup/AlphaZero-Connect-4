"""
Agents package for AlphaZero Connect-4.

This package contains different types of agents that can play Connect-4:
- BaseAgent: Abstract base class for all agents
- PolicyAgent: Neural network policy-based agent
- ValueAgent: Neural network value-based agent  
- RandomAgent: Random baseline agent
- MinimaxAgent: Game tree search agent with alpha-beta pruning
- MCTSAgent: Monte Carlo Tree Search agent with random rollouts
"""

from .base_agent import BaseAgent
from .policy_agent import PolicyAgent
from .value_agent import ValueAgent
from .random_agent import RandomAgent
from .minimax_agent import MinimaxAgent
from .mcts_agent import MCTSAgent

__all__ = ['BaseAgent', 'PolicyAgent', 'ValueAgent', 'RandomAgent', 'MinimaxAgent', 'MCTSAgent']