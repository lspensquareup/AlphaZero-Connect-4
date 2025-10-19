"""
Enhanced game visualizer for training games with agent information and statistics.

This module extends the existing ConnectFourGUI to show:
- Agent types and strategies
- Move analysis and thinking time
- Real-time game statistics
- Training context information
"""

import tkinter as tk
from tkinter import ttk
import threading
import time
import numpy as np
from typing import Dict, Any, Optional, Callable

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from connect4_gui import ConnectFourGUI
from connect4_env import GymnasiumConnectFour
from agents.base_agent import BaseAgent


class TrainingGameVisualizer(ConnectFourGUI):
    """
    Enhanced visualizer for training games with detailed agent information.
    """
    
    def __init__(self, env: GymnasiumConnectFour, agent1: BaseAgent, agent2: BaseAgent, 
                 game_callback: Optional[Callable] = None):
        """
        Initialize the training game visualizer.
        
        Args:
            env: Connect-4 environment
            agent1: First agent (player 1)
            agent2: Second agent (player -1)
            game_callback: Optional callback function when game ends
        """
        self.agent1 = agent1
        self.agent2 = agent2
        self.game_callback = game_callback
        self.move_history = []
        self.thinking_times = []
        self.current_thinking_time = 0
        self.game_start_time = time.time()
        
        # Initialize parent GUI
        super().__init__(env)
        
        # Override title and setup
        self.root.title(f"Training Game: {agent1.name} vs {agent2.name}")
        self.setup_agent_info()
        self.setup_analysis_panel()
        
        # Auto-start the game
        self.game_mode = "ai_vs_ai"
        self.mode_var.set("ai_vs_ai")
        self.start_game()
    
    def setup_agent_info(self):
        """Add agent information display."""
        # Agent info frame
        agent_frame = tk.Frame(self.root, bg=self.colors['background'])
        agent_frame.pack(pady=10)
        
        # Agent 1 info
        agent1_frame = tk.LabelFrame(agent_frame, text="Player 1 (Red)", 
                                    fg=self.colors['player1'], bg=self.colors['background'])
        agent1_frame.pack(side=tk.LEFT, padx=10, fill='both', expand=True)
        
        tk.Label(agent1_frame, text=f"Agent: {self.agent1.name}", 
                bg=self.colors['background'], fg=self.colors['text']).pack(pady=2)
        
        if hasattr(self.agent1, 'get_strategy_info'):
            strategy_info = self.agent1.get_strategy_info()
            for key, value in strategy_info.items():
                tk.Label(agent1_frame, text=f"{key}: {value}", 
                        bg=self.colors['background'], fg=self.colors['text'], 
                        font=('Arial', 9)).pack(pady=1)
        
        # Agent 2 info
        agent2_frame = tk.LabelFrame(agent_frame, text="Player 2 (Yellow)", 
                                    fg=self.colors['player2'], bg=self.colors['background'])
        agent2_frame.pack(side=tk.RIGHT, padx=10, fill='both', expand=True)
        
        tk.Label(agent2_frame, text=f"Agent: {self.agent2.name}", 
                bg=self.colors['background'], fg=self.colors['text']).pack(pady=2)
        
        if hasattr(self.agent2, 'get_strategy_info'):
            strategy_info = self.agent2.get_strategy_info()
            for key, value in strategy_info.items():
                tk.Label(agent2_frame, text=f"{key}: {value}", 
                        bg=self.colors['background'], fg=self.colors['text'], 
                        font=('Arial', 9)).pack(pady=1)
    
    def setup_analysis_panel(self):
        """Add analysis and statistics panel."""
        analysis_frame = tk.LabelFrame(self.root, text="Game Analysis", 
                                     bg=self.colors['background'], fg=self.colors['text'])
        analysis_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Game statistics
        stats_frame = tk.Frame(analysis_frame, bg=self.colors['background'])
        stats_frame.pack(side=tk.LEFT, fill='both', expand=True, padx=5)
        
        tk.Label(stats_frame, text="Game Statistics:", font=('Arial', 12, 'bold'),
                bg=self.colors['background'], fg=self.colors['text']).pack(anchor='w')
        
        self.stats_labels = {}
        stats_keys = ['Moves Played', 'Game Duration', 'Current Player', 'Last Move Time']
        
        for key in stats_keys:
            frame = tk.Frame(stats_frame, bg=self.colors['background'])
            frame.pack(fill='x', pady=1)
            
            tk.Label(frame, text=f"{key}:", bg=self.colors['background'], 
                    fg=self.colors['text'], font=('Arial', 10)).pack(side=tk.LEFT)
            
            self.stats_labels[key] = tk.Label(frame, text="0", bg=self.colors['background'], 
                                            fg=self.colors['highlight'], font=('Arial', 10, 'bold'))
            self.stats_labels[key].pack(side=tk.RIGHT)
        
        # Move history
        history_frame = tk.Frame(analysis_frame, bg=self.colors['background'])
        history_frame.pack(side=tk.RIGHT, fill='both', expand=True, padx=5)
        
        tk.Label(history_frame, text="Move History:", font=('Arial', 12, 'bold'),
                bg=self.colors['background'], fg=self.colors['text']).pack(anchor='w')
        
        self.history_text = tk.Text(history_frame, height=8, width=30, 
                                   bg='white', fg='black', font=('Courier', 9))
        history_scrollbar = tk.Scrollbar(history_frame, orient='vertical', 
                                       command=self.history_text.yview)
        self.history_text.configure(yscrollcommand=history_scrollbar.set)
        
        self.history_text.pack(side=tk.LEFT, fill='both', expand=True)
        history_scrollbar.pack(side=tk.RIGHT, fill='y')
        
        # Update statistics initially
        self.update_game_statistics()
    
    def make_ai_move(self):
        """Override to use our specific agents and track thinking time."""
        if not self.game_running or self.game_paused:
            return
        
        # Determine current agent
        current_agent = self.agent1 if self.env.current_player == 1 else self.agent2
        
        # Track thinking time
        start_time = time.time()
        
        # Get action from agent
        action = current_agent.select_action(self.env)
        
        # Calculate thinking time
        thinking_time = time.time() - start_time
        self.current_thinking_time = thinking_time
        self.thinking_times.append(thinking_time)
        
        # Record move
        self.move_history.append({
            'move_number': len(self.move_history) + 1,
            'player': self.env.current_player,
            'agent': current_agent.name,
            'column': action,
            'thinking_time': thinking_time,
            'board_state': self.env.board.copy()
        })
        
        # Make the move
        self.root.after(0, lambda: self.make_move(action))
        
        # Update statistics
        self.root.after(0, self.update_game_statistics)
    
    def run_ai_game(self):
        """Override to handle our specific AI game logic."""
        while self.game_running:
            if self.game_paused:
                time.sleep(0.1)
                continue
            
            # Add some delay for visualization
            time.sleep(max(0.5, self.animation_speed))
            
            if not self.game_running:
                break
            
            # Make AI move
            self.make_ai_move()
            
            # Brief pause after move
            time.sleep(0.2)
    
    def make_move(self, col):
        """Override to handle game completion callback."""
        success = super().make_move(col)
        
        if success and not self.game_running:  # Game ended
            # Calculate final statistics
            total_time = time.time() - self.game_start_time
            
            # Game result information
            game_result = {
                'winner': getattr(self, 'winner', None),
                'total_moves': len(self.move_history),
                'total_time': total_time,
                'move_history': self.move_history.copy(),
                'thinking_times': self.thinking_times.copy(),
                'agent1_name': self.agent1.name,
                'agent2_name': self.agent2.name
            }
            
            # Call callback if provided
            if self.game_callback:
                self.game_callback(game_result)
            
            # Show final analysis
            self.show_final_analysis(game_result)
    
    def update_game_statistics(self):
        """Update the real-time game statistics."""
        # Update move count
        self.stats_labels['Moves Played'].config(text=str(len(self.move_history)))
        
        # Update game duration
        duration = time.time() - self.game_start_time
        self.stats_labels['Game Duration'].config(text=f"{duration:.1f}s")
        
        # Update current player
        player_text = f"Player {self.env.current_player} ({'Red' if self.env.current_player == 1 else 'Yellow'})"
        self.stats_labels['Current Player'].config(text=player_text)
        
        # Update last move time
        if self.thinking_times:
            self.stats_labels['Last Move Time'].config(text=f"{self.current_thinking_time:.3f}s")
        
        # Update move history display
        self.update_move_history()
    
    def update_move_history(self):
        """Update the move history display."""
        self.history_text.delete(1.0, tk.END)
        
        for move in self.move_history[-15:]:  # Show last 15 moves
            player_symbol = "🔴" if move['player'] == 1 else "🟡"
            agent_short = move['agent'][:10] + "..." if len(move['agent']) > 13 else move['agent']
            
            move_text = f"{move['move_number']:2d}. {player_symbol} {agent_short}\n"
            move_text += f"    Col {move['column']}, {move['thinking_time']:.3f}s\n"
            
            self.history_text.insert(tk.END, move_text)
        
        self.history_text.see(tk.END)
    
    def show_final_analysis(self, game_result: Dict):
        """Show final game analysis in a popup window."""
        analysis_window = tk.Toplevel(self.root)
        analysis_window.title("Game Analysis")
        analysis_window.geometry("600x400")
        analysis_window.configure(bg=self.colors['background'])
        
        # Title
        title_text = f"Game Complete: {game_result['agent1_name']} vs {game_result['agent2_name']}"
        tk.Label(analysis_window, text=title_text, font=('Arial', 16, 'bold'),
                bg=self.colors['background'], fg=self.colors['text']).pack(pady=10)
        
        # Game summary
        summary_frame = tk.LabelFrame(analysis_window, text="Game Summary", 
                                    bg=self.colors['background'], fg=self.colors['text'])
        summary_frame.pack(fill='x', padx=20, pady=10)
        
        # Winner info
        winner_text = "Game tied!" if not game_result.get('winner') else f"Winner: Player {game_result['winner']}"
        tk.Label(summary_frame, text=winner_text, font=('Arial', 14, 'bold'),
                bg=self.colors['background'], fg=self.colors['highlight']).pack()
        
        # Statistics
        stats_text = f"""
Total Moves: {game_result['total_moves']}
Game Duration: {game_result['total_time']:.1f} seconds
Average Thinking Time: {np.mean(game_result['thinking_times']):.3f} seconds
        """
        
        tk.Label(summary_frame, text=stats_text, bg=self.colors['background'], 
                fg=self.colors['text'], justify='left').pack()
        
        # Thinking time analysis
        if game_result['thinking_times']:
            analysis_text = f"""
Fastest Move: {min(game_result['thinking_times']):.3f}s
Slowest Move: {max(game_result['thinking_times']):.3f}s
Total Thinking Time: {sum(game_result['thinking_times']):.1f}s
            """
            
            tk.Label(summary_frame, text=analysis_text, bg=self.colors['background'], 
                    fg=self.colors['text'], justify='left').pack()
        
        # Close button
        tk.Button(analysis_window, text="Close", command=analysis_window.destroy,
                 bg='white', fg='black', font=('Arial', 12)).pack(pady=20)


class GameSessionManager:
    """
    Manager for running multiple training games with visualization.
    """
    
    def __init__(self):
        self.session_stats = {
            'games_played': 0,
            'agent_wins': {},
            'total_moves': 0,
            'total_time': 0,
            'game_results': []
        }
    
    def play_visualized_game(self, agent1: BaseAgent, agent2: BaseAgent, 
                           auto_close: bool = True, speed: float = 1.0) -> Dict:
        """
        Play a single game with visualization.
        
        Args:
            agent1: First agent
            agent2: Second agent
            auto_close: Whether to auto-close after game ends
            speed: Game speed multiplier
            
        Returns:
            Game result dictionary
        """
        env = GymnasiumConnectFour()
        env.reset()
        
        game_result = {}
        
        def game_finished_callback(result):
            nonlocal game_result
            game_result = result
            self.session_stats['games_played'] += 1
            self.session_stats['game_results'].append(result)
            
            # Update agent win stats
            if result.get('winner'):
                winner_agent = agent1.name if result['winner'] == 1 else agent2.name
                if winner_agent not in self.session_stats['agent_wins']:
                    self.session_stats['agent_wins'][winner_agent] = 0
                self.session_stats['agent_wins'][winner_agent] += 1
            
            # Update totals
            self.session_stats['total_moves'] += result['total_moves']
            self.session_stats['total_time'] += result['total_time']
        
        # Create and run visualizer
        visualizer = TrainingGameVisualizer(env, agent1, agent2, game_finished_callback)
        visualizer.animation_speed = 1.0 / speed  # Convert speed to delay
        
        if auto_close:
            # Auto-close after game completion
            def check_game_complete():
                if not visualizer.game_running and game_result:
                    visualizer.root.after(3000, visualizer.root.destroy)  # Close after 3 seconds
                else:
                    visualizer.root.after(1000, check_game_complete)  # Check again in 1 second
            
            visualizer.root.after(1000, check_game_complete)
        
        visualizer.mainloop()
        
        return game_result
    
    def get_session_summary(self) -> str:
        """Get a summary of the current session."""
        if self.session_stats['games_played'] == 0:
            return "No games played in this session."
        
        avg_moves = self.session_stats['total_moves'] / self.session_stats['games_played']
        avg_time = self.session_stats['total_time'] / self.session_stats['games_played']
        
        summary = f"""Session Summary:
Games Played: {self.session_stats['games_played']}
Average Moves per Game: {avg_moves:.1f}
Average Game Duration: {avg_time:.1f}s

Agent Win Rates:"""
        
        for agent, wins in self.session_stats['agent_wins'].items():
            win_rate = (wins / self.session_stats['games_played']) * 100
            summary += f"\n  {agent}: {wins} wins ({win_rate:.1f}%)"
        
        return summary


if __name__ == "__main__":
    # Example usage
    from agents.minimax_agent import MinimaxAgent
    from agents.random_agent import RandomAgent
    
    # Create agents
    minimax_agent = MinimaxAgent(depth=4, name="Minimax-4")
    random_agent = RandomAgent()
    
    # Create session manager
    session = GameSessionManager()
    
    # Play a visualized game
    print("Starting visualized training game...")
    result = session.play_visualized_game(minimax_agent, random_agent, auto_close=False)
    
    print("\nSession Summary:")
    print(session.get_session_summary())