"""
Tournament system for Connect-4 AI agents.

This module provides functionality to run tournaments between different AI agents,
collect statistics, and evaluate agent performance.
"""

import time
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import sys
import os

# Add parent directory to path to import from project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from connect4_env import GymnasiumConnectFour
from agents.base_agent import BaseAgent


class Tournament:
    """
    Tournament system for evaluating Connect-4 agents.
    
    Supports round-robin tournaments, head-to-head matches, and statistical analysis.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the tournament system.
        
        Args:
            verbose: Whether to print detailed match results
        """
        self.verbose = verbose
        self.match_history = []
        self.agent_stats = defaultdict(lambda: {
            'wins': 0, 'losses': 0, 'ties': 0, 'games': 0,
            'win_rate': 0.0, 'points': 0.0
        })
    
    def play_match(self, agent1: BaseAgent, agent2: BaseAgent, 
                   num_games: int = 10) -> Dict[str, int]:
        """
        Play a match between two agents.
        
        Args:
            agent1: First agent
            agent2: Second agent  
            num_games: Number of games to play (agent1 starts half)
            
        Returns:
            Dictionary with match results
        """
        results = {
            'agent1_wins': 0,
            'agent2_wins': 0, 
            'ties': 0,
            'games': num_games
        }
        
        games_per_side = num_games // 2
        remaining_games = num_games % 2
        
        if self.verbose:
            print(f"\n🏆 Match: {agent1.name} vs {agent2.name}")
            print(f"Playing {num_games} games...")
        
        # Agent1 starts first for half the games
        for game_num in range(games_per_side):
            winner = self._play_single_game(agent1, agent2, game_num + 1)
            if winner == 1:
                results['agent1_wins'] += 1
            elif winner == -1:
                results['agent2_wins'] += 1
            else:
                results['ties'] += 1
        
        # Agent2 starts first for the other half
        for game_num in range(games_per_side):
            winner = self._play_single_game(agent2, agent1, games_per_side + game_num + 1)
            if winner == 1:  # Winner from agent2's perspective
                results['agent2_wins'] += 1
            elif winner == -1:  # Winner from agent2's perspective
                results['agent1_wins'] += 1
            else:
                results['ties'] += 1
        
        # Play remaining game if odd number of games
        if remaining_games > 0:
            winner = self._play_single_game(agent1, agent2, num_games)
            if winner == 1:
                results['agent1_wins'] += 1
            elif winner == -1:
                results['agent2_wins'] += 1
            else:
                results['ties'] += 1
        
        # Update statistics
        self._update_stats(agent1.name, results['agent1_wins'], 
                          results['agent2_wins'], results['ties'])
        self._update_stats(agent2.name, results['agent2_wins'], 
                          results['agent1_wins'], results['ties'])
        
        # Store match history
        self.match_history.append({
            'agent1': agent1.name,
            'agent2': agent2.name,
            'results': results.copy(),
            'timestamp': time.time()
        })
        
        if self.verbose:
            self._print_match_results(agent1.name, agent2.name, results)
        
        return results
    
    def _play_single_game(self, first_agent: BaseAgent, second_agent: BaseAgent, 
                          game_num: int) -> int:
        """
        Play a single game between two agents.
        
        Args:
            first_agent: Agent that plays first (player 1)
            second_agent: Agent that plays second (player -1)
            game_num: Game number for logging
            
        Returns:
            Winner: 1 if first_agent wins, -1 if second_agent wins, 0 if tie
        """
        env = GymnasiumConnectFour()
        
        # Set player IDs
        first_agent.set_player_id(1)
        second_agent.set_player_id(-1)
        
        # Reset environment
        board, info = env.reset()
        current_agent = first_agent
        
        moves = 0
        max_moves = 42  # 6x7 board
        
        while moves < max_moves:
            action_mask = info.get('action_mask', np.ones(7, dtype=np.int8))
            
            # Get action from current agent
            try:
                action = current_agent.select_action(board, action_mask)
                
                # Validate action
                if action_mask[action] == 0:
                    if self.verbose:
                        print(f"⚠️ Invalid move by {current_agent.name}: column {action}")
                    # Return win for the other agent
                    return -current_agent.player_id
                
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ Error from {current_agent.name}: {e}")
                # Return win for the other agent
                return -current_agent.player_id
            
            # Make move
            board, reward, terminated, truncated, info, _ = env.step(action)
            moves += 1
            
            if terminated:
                winner = info.get('winner', None)
                if winner is not None:
                    # Notify agents of game end
                    first_agent.game_over(board, winner)
                    second_agent.game_over(board, winner)
                    return winner
                else:
                    # Check tie
                    if info.get('reason') == 'Tie':
                        first_agent.game_over(board, 0)
                        second_agent.game_over(board, 0)
                        return 0
                    # Invalid move penalty
                    return -current_agent.player_id
            
            # Switch agents
            current_agent = second_agent if current_agent == first_agent else first_agent
        
        # Game exceeded maximum moves - tie
        first_agent.game_over(board, 0)
        second_agent.game_over(board, 0)
        return 0
    
    def run_round_robin(self, agents: List[BaseAgent], games_per_match: int = 10) -> Dict:
        """
        Run a round-robin tournament where every agent plays every other agent.
        
        Args:
            agents: List of agents to compete
            games_per_match: Number of games per head-to-head match
            
        Returns:
            Tournament results dictionary
        """
        if len(agents) < 2:
            raise ValueError("Need at least 2 agents for a tournament")
        
        print(f"\n🎯 Starting Round-Robin Tournament")
        print(f"Agents: {[agent.name for agent in agents]}")
        print(f"Games per match: {games_per_match}")
        print("=" * 50)
        
        # Clear previous stats
        self.agent_stats.clear()
        self.match_history.clear()
        
        start_time = time.time()
        
        # Play all pairs
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                self.play_match(agents[i], agents[j], games_per_match)
        
        tournament_time = time.time() - start_time
        
        # Calculate final rankings
        rankings = self._calculate_rankings()
        
        if self.verbose:
            self._print_tournament_results(rankings, tournament_time)
        
        return {
            'rankings': rankings,
            'agent_stats': dict(self.agent_stats),
            'match_history': self.match_history.copy(),
            'tournament_time': tournament_time
        }
    
    def _update_stats(self, agent_name: str, wins: int, losses: int, ties: int):
        """Update statistics for an agent."""
        stats = self.agent_stats[agent_name]
        stats['wins'] += wins
        stats['losses'] += losses
        stats['ties'] += ties
        stats['games'] += wins + losses + ties
        
        # Calculate win rate and points (win=1, tie=0.5, loss=0)
        if stats['games'] > 0:
            stats['win_rate'] = stats['wins'] / stats['games']
            stats['points'] = stats['wins'] + 0.5 * stats['ties']
    
    def _calculate_rankings(self) -> List[Dict]:
        """Calculate agent rankings based on points and win rate."""
        rankings = []
        
        for agent_name, stats in self.agent_stats.items():
            rankings.append({
                'agent': agent_name,
                'points': stats['points'],
                'win_rate': stats['win_rate'],
                'wins': stats['wins'],
                'losses': stats['losses'],
                'ties': stats['ties'],
                'games': stats['games']
            })
        
        # Sort by points, then by win rate
        rankings.sort(key=lambda x: (x['points'], x['win_rate']), reverse=True)
        
        # Add rank numbers
        for i, ranking in enumerate(rankings):
            ranking['rank'] = i + 1
        
        return rankings
    
    def _print_match_results(self, agent1_name: str, agent2_name: str, results: Dict):
        """Print results of a single match."""
        total_games = results['games']
        print(f"Results: {agent1_name} {results['agent1_wins']}-{results['agent2_wins']}-{results['ties']} {agent2_name}")
        
        if results['agent1_wins'] > results['agent2_wins']:
            print(f"🥇 {agent1_name} wins the match!")
        elif results['agent2_wins'] > results['agent1_wins']:
            print(f"🥇 {agent2_name} wins the match!")
        else:
            print("🤝 Match ends in a tie!")
    
    def _print_tournament_results(self, rankings: List[Dict], tournament_time: float):
        """Print final tournament results."""
        print("\n" + "=" * 60)
        print("🏆 TOURNAMENT RESULTS")
        print("=" * 60)
        
        print(f"{'Rank':<4} {'Agent':<20} {'Points':<8} {'Win%':<8} {'W-L-T':<12} {'Games':<6}")
        print("-" * 60)
        
        for ranking in rankings:
            win_rate_pct = ranking['win_rate'] * 100
            wlt = f"{ranking['wins']}-{ranking['losses']}-{ranking['ties']}"
            print(f"{ranking['rank']:<4} {ranking['agent']:<20} {ranking['points']:<8.1f} "
                  f"{win_rate_pct:<7.1f}% {wlt:<12} {ranking['games']:<6}")
        
        print(f"\nTournament completed in {tournament_time:.2f} seconds")
        print(f"Total matches: {len(self.match_history)}")
        
        # Print champion
        if rankings:
            champion = rankings[0]
            print(f"\n🎉 CHAMPION: {champion['agent']} with {champion['points']} points!")
    
    def save_results(self, filename: str):
        """Save tournament results to a file."""
        import json
        
        results = {
            'agent_stats': dict(self.agent_stats),
            'match_history': self.match_history,
            'rankings': self._calculate_rankings(),
            'timestamp': time.time()
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {filename}")


def quick_evaluate(agent1: BaseAgent, agent2: BaseAgent, num_games: int = 10) -> Dict:
    """
    Quick function to evaluate two agents against each other.
    
    Args:
        agent1: First agent
        agent2: Second agent
        num_games: Number of games to play
        
    Returns:
        Match results
    """
    tournament = Tournament(verbose=True)
    return tournament.play_match(agent1, agent2, num_games)