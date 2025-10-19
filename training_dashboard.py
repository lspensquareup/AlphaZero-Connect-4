"""
Training visualization and analytics dashboard for AlphaZero Connect-4.

This module provides comprehensive visualization for:
- Live training games with GUI
- Training metrics and loss charts
- Agent performance comparisons
- Tournament statistics
"""

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import json
import os
import sys
import threading
import time
from typing import List, Dict, Any, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from connect4_gui import ConnectFourGUI
from connect4_env import GymnasiumConnectFour
from training.trainer import Trainer, generate_minimax_training_data
from agents.minimax_agent import MinimaxAgent
from agents.random_agent import RandomAgent
from networks.policy_network import PolicyNetwork
from networks.value_network import ValueNetwork


class TrainingVisualizationDashboard:
    """
    Comprehensive dashboard for training visualization and analytics.
    """
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AlphaZero Connect-4 Training Dashboard")
        self.root.geometry("1400x900")
        self.root.configure(bg='#2c3e50')
        
        # Training data storage
        self.training_history = {
            'policy_losses': [],
            'value_losses': [],
            'epochs': [],
            'training_examples': [],
            'game_results': [],
            'agent_winrates': {}
        }
        
        # Current training session
        self.current_trainer = None
        self.is_training = False
        self.training_thread = None
        
        # Game visualization
        self.game_gui = None
        self.current_env = None
        
        self.setup_ui()
        self.load_existing_data()
    
    def setup_ui(self):
        """Set up the main dashboard UI."""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Tab 1: Training Control
        self.setup_training_tab()
        
        # Tab 2: Live Games
        self.setup_games_tab()
        
        # Tab 3: Analytics
        self.setup_analytics_tab()
        
        # Tab 4: Performance Comparison
        self.setup_comparison_tab()
    
    def setup_training_tab(self):
        """Set up the training control tab."""
        training_frame = ttk.Frame(self.notebook)
        self.notebook.add(training_frame, text="Training Control")
        
        # Control panel
        control_frame = ttk.LabelFrame(training_frame, text="Training Parameters")
        control_frame.pack(fill='x', padx=10, pady=5)
        
        # Parameters
        ttk.Label(control_frame, text="Number of Games:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.games_var = tk.IntVar(value=50)
        ttk.Entry(control_frame, textvariable=self.games_var, width=10).grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(control_frame, text="Minimax Depth:").grid(row=0, column=2, sticky='w', padx=5, pady=2)
        self.depth_var = tk.IntVar(value=4)
        ttk.Entry(control_frame, textvariable=self.depth_var, width=10).grid(row=0, column=3, padx=5, pady=2)
        
        ttk.Label(control_frame, text="Training Epochs:").grid(row=1, column=0, sticky='w', padx=5, pady=2)
        self.epochs_var = tk.IntVar(value=20)
        ttk.Entry(control_frame, textvariable=self.epochs_var, width=10).grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Label(control_frame, text="Batch Size:").grid(row=1, column=2, sticky='w', padx=5, pady=2)
        self.batch_var = tk.IntVar(value=16)
        ttk.Entry(control_frame, textvariable=self.batch_var, width=10).grid(row=1, column=3, padx=5, pady=2)
        
        # Visualization options
        viz_frame = ttk.LabelFrame(training_frame, text="Visualization Options")
        viz_frame.pack(fill='x', padx=10, pady=5)
        
        self.show_games_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(viz_frame, text="Show Training Games", variable=self.show_games_var).pack(side='left', padx=5)
        
        self.realtime_charts_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(viz_frame, text="Real-time Loss Charts", variable=self.realtime_charts_var).pack(side='left', padx=5)
        
        # Control buttons
        button_frame = ttk.Frame(training_frame)
        button_frame.pack(fill='x', padx=10, pady=10)
        
        self.start_training_btn = ttk.Button(button_frame, text="Start Training", command=self.start_training)
        self.start_training_btn.pack(side='left', padx=5)
        
        self.stop_training_btn = ttk.Button(button_frame, text="Stop Training", command=self.stop_training, state='disabled')
        self.stop_training_btn.pack(side='left', padx=5)
        
        ttk.Button(button_frame, text="Generate Sample Games", command=self.generate_sample_games).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Clear History", command=self.clear_history).pack(side='left', padx=5)
        
        # Progress and status
        self.progress_var = tk.StringVar(value="Ready to start training")
        ttk.Label(training_frame, textvariable=self.progress_var, font=('Arial', 12)).pack(pady=5)
        
        self.progress_bar = ttk.Progressbar(training_frame, length=600, mode='determinate')
        self.progress_bar.pack(pady=5)
        
        # Training log
        log_frame = ttk.LabelFrame(training_frame, text="Training Log")
        log_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.log_text = tk.Text(log_frame, height=15, wrap='word')
        log_scrollbar = ttk.Scrollbar(log_frame, orient='vertical', command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        
        self.log_text.pack(side='left', fill='both', expand=True)
        log_scrollbar.pack(side='right', fill='y')
    
    def setup_games_tab(self):
        """Set up the live games visualization tab."""
        games_frame = ttk.Frame(self.notebook)
        self.notebook.add(games_frame, text="Live Games")
        
        # Control panel
        game_control_frame = ttk.LabelFrame(games_frame, text="Game Controls")
        game_control_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(game_control_frame, text="New Game: Minimax vs Random", 
                  command=lambda: self.start_demo_game("minimax_random")).pack(side='left', padx=5)
        ttk.Button(game_control_frame, text="New Game: Trained vs Random", 
                  command=lambda: self.start_demo_game("trained_random")).pack(side='left', padx=5)
        ttk.Button(game_control_frame, text="New Game: Minimax vs Trained", 
                  command=lambda: self.start_demo_game("minimax_trained")).pack(side='left', padx=5)
        
        # Game display area
        self.game_display_frame = ttk.Frame(games_frame)
        self.game_display_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Game statistics
        stats_frame = ttk.LabelFrame(games_frame, text="Current Session Statistics")
        stats_frame.pack(fill='x', padx=10, pady=5)
        
        self.game_stats_text = tk.Text(stats_frame, height=8, wrap='word')
        stats_scrollbar = ttk.Scrollbar(stats_frame, orient='vertical', command=self.game_stats_text.yview)
        self.game_stats_text.configure(yscrollcommand=stats_scrollbar.set)
        
        self.game_stats_text.pack(side='left', fill='both', expand=True)
        stats_scrollbar.pack(side='right', fill='y')
    
    def setup_analytics_tab(self):
        """Set up the analytics and metrics tab."""
        analytics_frame = ttk.Frame(self.notebook)
        self.notebook.add(analytics_frame, text="Training Analytics")
        
        # Create matplotlib figures
        self.fig = Figure(figsize=(14, 8), facecolor='white')
        self.canvas = FigureCanvasTkAgg(self.fig, analytics_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=5)
        
        # Control buttons for analytics
        analytics_control_frame = ttk.Frame(analytics_frame)
        analytics_control_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(analytics_control_frame, text="Refresh Charts", command=self.update_analytics_charts).pack(side='left', padx=5)
        ttk.Button(analytics_control_frame, text="Export Data", command=self.export_training_data).pack(side='left', padx=5)
        ttk.Button(analytics_control_frame, text="Load Data", command=self.load_training_data).pack(side='left', padx=5)
        
        # Initialize charts
        self.setup_analytics_charts()
    
    def setup_comparison_tab(self):
        """Set up the agent performance comparison tab."""
        comparison_frame = ttk.Frame(self.notebook)
        self.notebook.add(comparison_frame, text="Agent Comparison")
        
        # Tournament controls
        tournament_frame = ttk.LabelFrame(comparison_frame, text="Tournament Setup")
        tournament_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(tournament_frame, text="Games per matchup:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.tournament_games_var = tk.IntVar(value=20)
        ttk.Entry(tournament_frame, textvariable=self.tournament_games_var, width=10).grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Button(tournament_frame, text="Run Tournament", command=self.run_tournament).grid(row=0, column=2, padx=10, pady=2)
        
        # Results display
        results_frame = ttk.LabelFrame(comparison_frame, text="Tournament Results")
        results_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.results_text = tk.Text(results_frame, wrap='word')
        results_scrollbar = ttk.Scrollbar(results_frame, orient='vertical', command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=results_scrollbar.set)
        
        self.results_text.pack(side='left', fill='both', expand=True)
        results_scrollbar.pack(side='right', fill='y')
    
    def setup_analytics_charts(self):
        """Set up the analytics charts."""
        self.fig.clear()
        
        # Create subplots
        self.ax1 = self.fig.add_subplot(2, 3, 1)  # Policy loss
        self.ax2 = self.fig.add_subplot(2, 3, 2)  # Value loss
        self.ax3 = self.fig.add_subplot(2, 3, 3)  # Training examples
        self.ax4 = self.fig.add_subplot(2, 3, 4)  # Win rates
        self.ax5 = self.fig.add_subplot(2, 3, 5)  # Game length distribution
        self.ax6 = self.fig.add_subplot(2, 3, 6)  # Loss comparison
        
        self.fig.tight_layout(pad=3.0)
        self.canvas.draw()
    
    def log_message(self, message: str):
        """Add a message to the training log."""
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
    
    def update_progress(self, current: int, total: int, message: str = ""):
        """Update the progress bar and status."""
        self.progress_bar['value'] = (current / total) * 100
        status = f"{message} ({current}/{total})" if message else f"Progress: {current}/{total}"
        self.progress_var.set(status)
        self.root.update_idletasks()
    
    def start_training(self):
        """Start the training process with visualization."""
        if self.is_training:
            return
        
        self.is_training = True
        self.start_training_btn.config(state='disabled')
        self.stop_training_btn.config(state='normal')
        
        # Start training in separate thread
        self.training_thread = threading.Thread(target=self._training_worker, daemon=True)
        self.training_thread.start()
    
    def stop_training(self):
        """Stop the training process."""
        self.is_training = False
        self.start_training_btn.config(state='normal')
        self.stop_training_btn.config(state='disabled')
        self.log_message("Training stopped by user")
    
    def _training_worker(self):
        """Worker function for training (runs in separate thread)."""
        try:
            self.log_message("Starting training session...")
            
            # Create trainer and networks
            trainer = Trainer()
            policy_net = PolicyNetwork()
            value_net = ValueNetwork()
            
            # Get parameters
            num_games = self.games_var.get()
            minimax_depth = self.depth_var.get()
            epochs = self.epochs_var.get()
            batch_size = self.batch_var.get()
            
            self.log_message(f"Parameters: {num_games} games, depth {minimax_depth}, {epochs} epochs, batch size {batch_size}")
            
            # Generate training data
            self.log_message("Generating training data with minimax games...")
            self.update_progress(0, num_games + epochs * 2, "Generating games")
            
            if self.show_games_var.get():
                # Show games as they're generated
                game_records = self._generate_games_with_visualization(num_games, minimax_depth)
            else:
                game_records = generate_minimax_training_data(num_games, minimax_depth)
            
            if not self.is_training:
                return
            
            # Convert to training data
            policy_data, value_data = trainer.generate_training_data_from_games(game_records)
            self.log_message(f"Generated {len(policy_data)} training examples")
            
            # Store game results
            self.training_history['game_results'].extend(game_records)
            self.training_history['training_examples'].append(len(policy_data))
            
            # Train policy network
            self.log_message("Training policy network...")
            self.update_progress(num_games, num_games + epochs * 2, "Training policy network")
            
            policy_results = trainer.train_policy_network(
                policy_net, policy_data, epochs, batch_size, 0.001
            )
            
            if not self.is_training:
                return
            
            # Train value network
            self.log_message("Training value network...")
            self.update_progress(num_games + epochs, num_games + epochs * 2, "Training value network")
            
            value_results = trainer.train_value_network(
                value_net, value_data, epochs, batch_size, 0.001
            )
            
            # Store results
            self.training_history['policy_losses'].extend(policy_results.get('epoch_losses', []))
            self.training_history['value_losses'].extend(value_results.get('epoch_losses', []))
            self.training_history['epochs'].extend(range(len(self.training_history['epochs']), 
                                                       len(self.training_history['epochs']) + epochs))
            
            self.log_message(f"Training complete! Policy loss: {policy_results['final_loss']:.6f}, Value loss: {value_results['final_loss']:.6f}")
            
            # Update charts if enabled
            if self.realtime_charts_var.get():
                self.root.after(0, self.update_analytics_charts)
            
            self.update_progress(num_games + epochs * 2, num_games + epochs * 2, "Training complete")
            
        except Exception as e:
            self.log_message(f"Training error: {str(e)}")
        finally:
            self.is_training = False
            self.root.after(0, lambda: [
                self.start_training_btn.config(state='normal'),
                self.stop_training_btn.config(state='disabled')
            ])
    
    def _generate_games_with_visualization(self, num_games: int, depth: int) -> List[Dict]:
        """Generate games with live visualization."""
        minimax_agent = MinimaxAgent(depth=depth)
        random_agent = RandomAgent()
        game_records = []
        
        for game_num in range(num_games):
            if not self.is_training:
                break
            
            self.log_message(f"Playing game {game_num + 1}/{num_games}")
            self.update_progress(game_num, num_games, f"Game {game_num + 1}")
            
            # Create environment for this game
            env = GymnasiumConnectFour()
            env.reset()
            
            # Show game in GUI if first few games
            if game_num < 5 and self.show_games_var.get():
                self.root.after(0, lambda: self._show_training_game(env, minimax_agent, random_agent))
                time.sleep(2)  # Brief pause to watch
            
            # Play the game
            game_record = self._play_single_game(env, minimax_agent, random_agent)
            game_records.append(game_record)
            
            time.sleep(0.1)  # Brief pause between games
        
        return game_records
    
    def _play_single_game(self, env, agent1, agent2) -> Dict:
        """Play a single game and return the record."""
        minimax_is_player1 = np.random.choice([True, False])
        
        board_states = []
        actions = []
        terminated = False
        winner = 0
        
        while not terminated:
            current_board = env.board.copy()
            
            # Choose action based on current player
            if (env.current_player == 1 and minimax_is_player1) or \
               (env.current_player == -1 and not minimax_is_player1):
                action = agent1.select_action(env)
            else:
                action = agent2.select_action(env)
            
            board_states.append(current_board)
            actions.append(action)
            
            # Make the move
            _, reward, terminated, _, info, _ = env.step(action)
            
            if terminated:
                if info.get("reason") == "Win":
                    winner = info.get("winner", 0)
                elif info.get("reason") == "Tie":
                    winner = 0
        
        return {
            'board_states': board_states,
            'actions': actions,
            'winner': winner,
            'minimax_is_player1': minimax_is_player1,
            'num_moves': len(actions)
        }
    
    def _show_training_game(self, env, agent1, agent2):
        """Show a training game in the GUI."""
        # This would integrate with your existing ConnectFourGUI
        # For now, just log the game
        self.log_message("Showing training game in GUI...")
    
    def generate_sample_games(self):
        """Generate a small set of sample games for testing."""
        self.log_message("Generating 10 sample games...")
        
        game_records = generate_minimax_training_data(10, 3)
        self.training_history['game_results'].extend(game_records)
        
        # Analyze results
        minimax_wins = sum(1 for g in game_records 
                          if (g['winner'] == 1 and g['minimax_is_player1']) or 
                             (g['winner'] == -1 and not g['minimax_is_player1']))
        
        self.log_message(f"Sample games complete. Minimax won {minimax_wins}/10 games")
        self.update_game_statistics()
    
    def update_analytics_charts(self):
        """Update all analytics charts."""
        self.fig.clear()
        self.setup_analytics_charts()
        
        # Policy loss chart
        if self.training_history['policy_losses']:
            self.ax1.plot(self.training_history['policy_losses'], 'b-', linewidth=2)
            self.ax1.set_title('Policy Network Loss')
            self.ax1.set_xlabel('Epoch')
            self.ax1.set_ylabel('Loss')
            self.ax1.grid(True, alpha=0.3)
        
        # Value loss chart
        if self.training_history['value_losses']:
            self.ax2.plot(self.training_history['value_losses'], 'r-', linewidth=2)
            self.ax2.set_title('Value Network Loss')
            self.ax2.set_xlabel('Epoch')
            self.ax2.set_ylabel('Loss')
            self.ax2.grid(True, alpha=0.3)
        
        # Training examples over time
        if self.training_history['training_examples']:
            self.ax3.plot(self.training_history['training_examples'], 'g-', marker='o')
            self.ax3.set_title('Training Examples per Session')
            self.ax3.set_xlabel('Training Session')
            self.ax3.set_ylabel('Examples')
            self.ax3.grid(True, alpha=0.3)
        
        # Win rates analysis
        if self.training_history['game_results']:
            self._plot_win_rates()
        
        # Game length distribution
        if self.training_history['game_results']:
            self._plot_game_lengths()
        
        # Combined loss comparison
        if self.training_history['policy_losses'] and self.training_history['value_losses']:
            min_len = min(len(self.training_history['policy_losses']), 
                         len(self.training_history['value_losses']))
            self.ax6.plot(self.training_history['policy_losses'][:min_len], 'b-', label='Policy', linewidth=2)
            self.ax6.plot(self.training_history['value_losses'][:min_len], 'r-', label='Value', linewidth=2)
            self.ax6.set_title('Training Loss Comparison')
            self.ax6.set_xlabel('Epoch')
            self.ax6.set_ylabel('Loss')
            self.ax6.legend()
            self.ax6.grid(True, alpha=0.3)
        
        self.fig.tight_layout(pad=3.0)
        self.canvas.draw()
    
    def _plot_win_rates(self):
        """Plot win rates for different agents."""
        games = self.training_history['game_results']
        
        minimax_wins = sum(1 for g in games 
                          if (g['winner'] == 1 and g['minimax_is_player1']) or 
                             (g['winner'] == -1 and not g['minimax_is_player1']))
        random_wins = sum(1 for g in games 
                         if (g['winner'] == 1 and not g['minimax_is_player1']) or 
                            (g['winner'] == -1 and g['minimax_is_player1']))
        ties = sum(1 for g in games if g['winner'] == 0)
        
        labels = ['Minimax', 'Random', 'Ties']
        values = [minimax_wins, random_wins, ties]
        colors = ['#3498db', '#e74c3c', '#95a5a6']
        
        self.ax4.pie(values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        self.ax4.set_title('Game Outcomes')
    
    def _plot_game_lengths(self):
        """Plot distribution of game lengths."""
        if not self.training_history['game_results']:
            return
        
        lengths = [g['num_moves'] for g in self.training_history['game_results']]
        self.ax5.hist(lengths, bins=20, alpha=0.7, color='#9b59b6', edgecolor='black')
        self.ax5.set_title('Game Length Distribution')
        self.ax5.set_xlabel('Number of Moves')
        self.ax5.set_ylabel('Frequency')
        self.ax5.grid(True, alpha=0.3)
    
    def update_game_statistics(self):
        """Update the game statistics display."""
        if not self.training_history['game_results']:
            return
        
        games = self.training_history['game_results']
        total_games = len(games)
        
        # Calculate statistics
        minimax_wins = sum(1 for g in games 
                          if (g['winner'] == 1 and g['minimax_is_player1']) or 
                             (g['winner'] == -1 and not g['minimax_is_player1']))
        random_wins = sum(1 for g in games 
                         if (g['winner'] == 1 and not g['minimax_is_player1']) or 
                            (g['winner'] == -1 and g['minimax_is_player1']))
        ties = sum(1 for g in games if g['winner'] == 0)
        
        avg_length = np.mean([g['num_moves'] for g in games])
        
        stats = f"""Game Statistics (Total: {total_games} games)
        
Minimax Agent:
  - Wins: {minimax_wins} ({100*minimax_wins/total_games:.1f}%)
  
Random Agent:
  - Wins: {random_wins} ({100*random_wins/total_games:.1f}%)
  
Ties: {ties} ({100*ties/total_games:.1f}%)

Average Game Length: {avg_length:.1f} moves

Training Examples Generated: {sum(self.training_history['training_examples'])}
"""
        
        self.game_stats_text.delete(1.0, tk.END)
        self.game_stats_text.insert(1.0, stats)
    
    def start_demo_game(self, game_type: str):
        """Start a demonstration game."""
        self.log_message(f"Starting demo game: {game_type}")
        # Implementation would integrate with your existing GUI
        pass
    
    def run_tournament(self):
        """Run a tournament between different agents."""
        self.log_message("Starting tournament...")
        # Implementation for agent tournaments
        pass
    
    def clear_history(self):
        """Clear training history."""
        self.training_history = {
            'policy_losses': [],
            'value_losses': [],
            'epochs': [],
            'training_examples': [],
            'game_results': [],
            'agent_winrates': {}
        }
        self.log_message("Training history cleared")
        self.update_analytics_charts()
    
    def load_existing_data(self):
        """Load existing training data from files."""
        try:
            if os.path.exists('models/training_session.json'):
                with open('models/training_session.json', 'r') as f:
                    data = json.load(f)
                    # Update history with loaded data
                    if 'training_examples' in data:
                        self.training_history['training_examples'].append(data['training_examples'])
                self.log_message("Loaded existing training session data")
        except Exception as e:
            self.log_message(f"Could not load existing data: {e}")
    
    def export_training_data(self):
        """Export training data to file."""
        try:
            with open('training_analytics.json', 'w') as f:
                json.dump(self.training_history, f, indent=2, default=str)
            self.log_message("Training data exported to training_analytics.json")
        except Exception as e:
            self.log_message(f"Export failed: {e}")
    
    def load_training_data(self):
        """Load training data from file."""
        try:
            with open('training_analytics.json', 'r') as f:
                self.training_history = json.load(f)
            self.log_message("Training data loaded from training_analytics.json")
            self.update_analytics_charts()
            self.update_game_statistics()
        except Exception as e:
            self.log_message(f"Load failed: {e}")
    
    def run(self):
        """Start the dashboard."""
        self.log_message("Training Dashboard initialized")
        self.log_message("Ready to start training with visualization")
        self.root.mainloop()


if __name__ == "__main__":
    dashboard = TrainingVisualizationDashboard()
    dashboard.run()