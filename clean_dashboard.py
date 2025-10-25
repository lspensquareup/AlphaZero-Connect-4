"""
Enhanced training dashboard with agent selection, iterative training, and evaluation tracking.
"""

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import os
import sys
import time
import torch

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from training.trainer import Trainer
from networks.policy_network import PolicyNetwork
from networks.value_network import ValueNetwork
from connect4_env import GymnasiumConnectFour
from agents.minimax_agent import MinimaxAgent
from agents.random_agent import RandomAgent
from agents.policy_agent import PolicyAgent
from agents.value_agent import ValueAgent
from agents.mcts_agent import MCTSAgent


class SimpleGameBoard:
    """Simple embedded Connect-4 board visualization."""
    
    def __init__(self, parent_frame):
        self.parent = parent_frame
        self.board = np.zeros((6, 7), dtype=int)
        self.cell_size = 50
        
        # Create canvas
        self.canvas = tk.Canvas(parent_frame, 
                               width=7 * self.cell_size + 20, 
                               height=6 * self.cell_size + 20,
                               bg='blue')
        self.canvas.pack(pady=10)
        
        self.draw_empty_board()
    
    def draw_empty_board(self):
        """Draw the empty Connect-4 board."""
        self.canvas.delete("all")
        
        # Draw board background
        self.canvas.create_rectangle(10, 10, 
                                   7 * self.cell_size + 10, 
                                   6 * self.cell_size + 10,
                                   fill='blue', outline='black', width=2)
        
        # Draw grid and holes
        for row in range(6):
            for col in range(7):
                x1 = col * self.cell_size + 15
                y1 = row * self.cell_size + 15
                x2 = x1 + self.cell_size - 10
                y2 = y1 + self.cell_size - 10
                
                # Draw hole (empty space)
                self.canvas.create_oval(x1, y1, x2, y2, 
                                      fill='white', outline='black', width=1)
    
    def update_board(self, board_state):
        """Update the board with current game state."""
        self.board = board_state.copy()
        self.draw_empty_board()
        
        # Draw pieces
        for row in range(6):
            for col in range(7):
                if self.board[row, col] != 0:
                    x1 = col * self.cell_size + 15
                    y1 = row * self.cell_size + 15
                    x2 = x1 + self.cell_size - 10
                    y2 = y1 + self.cell_size - 10
                    
                    color = 'red' if self.board[row, col] == 1 else 'yellow'
                    self.canvas.create_oval(x1, y1, x2, y2, 
                                          fill=color, outline='black', width=2)


class GameWindow:
    """Separate window for game visualization."""
    
    def __init__(self, parent):
        self.parent = parent
        self.window = tk.Toplevel(parent.root)
        self.window.title("Connect-4 Game Viewer")
        self.window.geometry("500x600")  # Smaller since controls are integrated
        
        # Handle window closing
        self.window.protocol("WM_DELETE_WINDOW", self.on_window_close)
        
        # Info frame
        info_frame = tk.Frame(self.window)
        info_frame.pack(pady=10)
        
        self.info_label = tk.Label(info_frame, text="Game Viewer", font=('Arial', 14, 'bold'))
        self.info_label.pack()
        
        self.players_label = tk.Label(info_frame, text="Players: - vs -", font=('Arial', 12))
        self.players_label.pack()
        
        self.status_label = tk.Label(info_frame, text="Status: Ready", font=('Arial', 10))
        self.status_label.pack()
        
        # Game board
        board_frame = tk.Frame(self.window)
        board_frame.pack(pady=10)
        
        self.game_board = SimpleGameBoard(board_frame)
        
        # Controls
        controls_frame = tk.Frame(self.window)
        controls_frame.pack(pady=10)
        
        # Top row of controls
        controls_top_frame = tk.Frame(controls_frame)
        controls_top_frame.pack(pady=5)
        
        self.step_button = tk.Button(controls_top_frame, text="▶ Next Move", 
                                   command=self.next_move, state='disabled')
        self.step_button.pack(side='left', padx=5)
        
        self.auto_button = tk.Button(controls_top_frame, text="⏯ Auto Play", 
                                   command=self.toggle_auto_play, state='disabled')
        self.auto_button.pack(side='left', padx=5)
        
        self.reset_button = tk.Button(controls_top_frame, text="🔄 Reset", 
                                    command=self.reset_game, state='disabled')
        self.reset_button.pack(side='left', padx=5)
        
        # Bottom row - Auto-advance control (integrated with game controls)
        controls_bottom_frame = tk.Frame(controls_frame)
        controls_bottom_frame.pack(pady=5)
        
        # Use a simple boolean instead of tkinter variable
        self.auto_advance_enabled = True
        
        # Auto-advance toggle button (same style as other controls)
        self.auto_advance_button = tk.Button(controls_bottom_frame, 
                                           text="✅ Auto-advance: ON", 
                                           command=self.toggle_auto_advance,
                                           font=('Arial', 9),
                                           bg='lightgreen',
                                           activebackground='green',
                                           relief='raised',
                                           bd=2)
        self.auto_advance_button.pack(side='left', padx=5)
        
        # Status label for auto-advance mode
        self.auto_status_label = tk.Label(controls_bottom_frame, 
                                        text="Auto-advance enabled", 
                                        font=('Arial', 9), 
                                        fg='green')
        self.auto_status_label.pack(side='left', padx=10)
        
        # Speed control
        speed_frame = tk.Frame(self.window, relief='ridge', bd=1)
        speed_frame.pack(pady=10, padx=10, fill='x')
        
        tk.Label(speed_frame, text="Playback Speed:", font=('Arial', 10, 'bold')).pack(pady=2)
        
        speed_control_frame = tk.Frame(speed_frame)
        speed_control_frame.pack(pady=5)
        
        tk.Label(speed_control_frame, text="Fast").pack(side='left', padx=5)
        self.speed_var = tk.DoubleVar(value=1.0)
        self.speed_scale = tk.Scale(speed_control_frame, from_=0.2, to=3.0, resolution=0.2,
                                  orient='horizontal', variable=self.speed_var, length=250)
        self.speed_scale.pack(side='left', padx=5)
        tk.Label(speed_control_frame, text="Slow").pack(side='left', padx=5)
        
        # Speed display
        self.speed_display = tk.Label(speed_frame, text="Speed: 1.0s per move", font=('Arial', 9))
        self.speed_display.pack(pady=2)
        
        # Update speed display when slider changes
        self.speed_var.trace('w', self.update_speed_display)
        
        # Game state
        self.moves = []
        self.current_move = 0
        self.auto_playing = False
        self.player1_name = ""
        self.player2_name = ""
        self.winner = 0
        
        # Auto-show game state
        self.auto_show_queue = []
        self.current_auto_game = 0
        self.auto_show_active = False
        self.game_complete = False  # Track if current game is complete
    
    def on_window_close(self):
        """Handle window closing event."""
        try:
            # Stop any ongoing auto-play
            self.auto_playing = False
            self.auto_show_active = False
            
            # Clear parent reference
            if self.parent:
                self.parent.game_window = None
            
            # Destroy the window
            self.window.destroy()
        except Exception as e:
            print(f"DEBUG: Error closing window: {e}")
    
    def update_speed_display(self, *args):
        """Update the speed display when slider changes."""
        speed = self.speed_var.get()
        self.speed_display.config(text=f"Speed: {speed}s per move")
    
    def test_toggle_state(self):
        """Test method to check if toggle is working."""
        try:
            print(f"DEBUG: Toggle state is: {self.auto_advance_enabled}")
            print(f"DEBUG: Toggle button exists: {hasattr(self, 'auto_advance_button')}")
            
            # Try to toggle programmatically to test
            old_state = self.auto_advance_enabled
            self.toggle_auto_advance()
            new_state = self.auto_advance_enabled
            print(f"DEBUG: After toggle, state changed from {old_state} to {new_state}")
            
            # Update display
            self.auto_status_label.config(text=f"DEBUG: Toggled from {old_state} to {new_state}", fg='blue')
            
        except Exception as e:
            print(f"DEBUG: Error in test: {e}")
            self.auto_status_label.config(text=f"DEBUG: Error - {e}", fg='red')
    
    def toggle_auto_advance(self):
        """Toggle the auto-advance setting."""
        try:
            self.auto_advance_enabled = not self.auto_advance_enabled
            print(f"DEBUG: Toggle clicked! New state: {self.auto_advance_enabled}")
            
            if self.auto_advance_enabled:
                self.auto_advance_button.config(text="✅ Auto-advance: ON", bg='lightgreen')
                self.auto_status_label.config(text="Auto-advance enabled", fg='green')
                
                # If we're in manual mode and game is complete, auto-continue
                if self.game_complete and self.auto_show_active:
                    print("DEBUG: Auto-continuing from manual mode")
                    self.manual_next_game()
            else:
                self.auto_advance_button.config(text="⏸️ Auto-advance: OFF", bg='lightcoral')
                self.auto_status_label.config(text="Manual advance mode", fg='red')
                
            # Update button text based on new state
            self.update_button_text()
            
            # Force window update
            self.window.update()
            
        except Exception as e:
            print(f"DEBUG: Error in toggle: {e}")
                
            # Force window update
            self.window.update()
            
        except Exception as e:
            print(f"DEBUG: Error in toggle: {e}")
    
    def on_auto_advance_changed(self):
        """Called when the auto-advance checkbox is toggled."""
        try:
            self.auto_advance_enabled = not self.auto_advance_enabled
            print(f"DEBUG: Toggle clicked! New state: {self.auto_advance_enabled}")
            
            if self.auto_advance_enabled:
                self.auto_advance_button.config(text="✅ Auto-advance: ON", bg='lightgreen')
                self.auto_status_label.config(text="Auto-advance enabled", fg='green')
                
                # If we're in the middle of a sequence and waiting, continue automatically
                if self.auto_show_active and self.game_complete:
                    print("DEBUG: Auto-continuing from manual mode")
                    self.manual_next_game()
            else:
                self.auto_advance_button.config(text="⏸️ Auto-advance: OFF", bg='lightcoral')
                self.auto_status_label.config(text="Manual advance mode", fg='orange')
                
            # Force window update
            self.window.update()
        except Exception as e:
            print(f"DEBUG: Error in checkbox callback: {e}")
    
    def manual_next_game(self):
        """Manually advance to the next game."""
        print(f"DEBUG: manual_next_game called")
        if self.auto_show_active and self.current_auto_game < len(self.auto_show_queue):
            self.current_auto_game += 1
            self.show_next_auto_game()
        else:
            self.auto_show_active = False
            self.step_button.config(state='normal')
            self.update_button_text()
    
    def show_game(self, moves, player1_name, player2_name, winner):
        """Display a game sequence."""
        self.moves = moves
        self.player1_name = player1_name
        self.player2_name = player2_name
        self.winner = winner
        self.current_move = 0
        self.auto_playing = False
        self.game_complete = False  # Reset game complete flag
        
        # Update UI
        self.players_label.config(text=f"Players: {player1_name} vs {player2_name}")
        self.status_label.config(text=f"Game loaded: {len(moves)} moves")
        
        # Enable controls
        self.step_button.config(state='normal')
        self.auto_button.config(state='normal', text="⏯ Auto Play")
        self.reset_button.config(state='normal')
        
        # Update button text
        self.update_button_text()
        
        # Reset board
        self.reset_game()
    
    def show_games_sequence(self, games_list):
        """Show multiple games in sequence automatically."""
        self.auto_show_queue = games_list.copy()
        self.current_auto_game = 0
        self.auto_show_active = True
        
        if self.auto_show_queue:
            self.show_next_auto_game()
    
    def show_next_auto_game(self):
        """Show the next game in the auto-show sequence."""
        if self.current_auto_game < len(self.auto_show_queue):
            game_data = self.auto_show_queue[self.current_auto_game]
            moves, player1_name, player2_name, winner = game_data
            
            # Update title to show game number
            game_num = self.current_auto_game + 1
            total_games = len(self.auto_show_queue)
            self.window.title(f"Connect-4 Game Viewer - Game {game_num}/{total_games}")
            
            # Disable step button while game is playing
            self.step_button.config(state='disabled')
            
            # Show the game
            self.show_game(moves, player1_name, player2_name, winner)
            
            # Start auto-play immediately
            self.auto_playing = True
            self.auto_button.config(text="⏸ Pause")
            self.next_move()
        else:
            # All games shown
            self.auto_show_active = False
            self.step_button.config(state='normal')
            self.window.title("Connect-4 Game Viewer - All Games Complete")
            self.status_label.config(text="All games displayed!")
    
    def on_game_complete(self):
        """Called when current game finishes playing."""
        print(f"DEBUG: on_game_complete called")
        if self.auto_show_active and self.current_auto_game < len(self.auto_show_queue):
            print(f"DEBUG: Auto-advance setting: {self.auto_advance_enabled}")
            
            if self.auto_advance_enabled:
                # Auto-advance is enabled, proceed automatically
                self.status_label.config(text="Game complete! Auto-advancing in 2 seconds...")
                self.current_auto_game += 1
                # Wait a bit before showing next game
                self.window.after(2000, self.show_next_auto_game)
            else:
                # Manual advance mode, update button and status
                remaining_games = len(self.auto_show_queue) - self.current_auto_game
                self.update_button_text()  # This will change to "Next Game"
                self.status_label.config(text=f"Game complete! {remaining_games} games remaining. Click 'Next Game' to continue.")
                print(f"DEBUG: Manual mode activated, {remaining_games} games remaining")
        else:
            # No more games or auto-show not active
            self.step_button.config(text="▶ Next Move")
    
    def next_move(self):
        """Show next move or advance to next game."""
        # Check if we should advance to next game instead of next move
        if self.game_complete and self.auto_show_active and not self.auto_advance_enabled:
            self.manual_next_game()
            return
            
        if self.current_move < len(self.moves):
            # Simulate the move
            env = GymnasiumConnectFour()
            env.reset()
            
            # Play moves up to current position
            for i in range(self.current_move + 1):
                if i < len(self.moves):
                    env.step(self.moves[i])
            
            # Update display
            self.game_board.update_board(env.board)
            self.current_move += 1
            
            # Update status
            current_player = self.player1_name if self.current_move % 2 == 1 else self.player2_name
            if self.current_move < len(self.moves):
                self.status_label.config(text=f"Move {self.current_move}: {current_player} played column {self.moves[self.current_move-1]}")
                self.update_button_text()
            else:
                winner_name = "Draw"
                if self.winner == 1:
                    winner_name = self.player1_name
                elif self.winner == -1:
                    winner_name = self.player2_name
                self.status_label.config(text=f"Game Over! Winner: {winner_name}")
                self.auto_playing = False
                self.auto_button.config(text="⏯ Auto Play")
                self.game_complete = True
                self.update_button_text()
                
                # Check if this was part of an auto-show sequence
                if self.auto_show_active:
                    self.on_game_complete()
        
        # Continue auto play if enabled
        if self.auto_playing and self.current_move < len(self.moves):
            # Use speed from slider (convert to milliseconds)
            delay_ms = int(self.speed_var.get() * 1000)
            self.window.after(delay_ms, self.next_move)
    
    def update_button_text(self):
        """Update the step button text based on context."""
        if self.game_complete and self.auto_show_active and not self.auto_advance_enabled:
            remaining_games = len(self.auto_show_queue) - self.current_auto_game
            if remaining_games > 0:
                self.step_button.config(text="➡️ Next Game")
            else:
                self.step_button.config(text="▶ Next Move")
        else:
            self.step_button.config(text="▶ Next Move")
    
    def toggle_auto_play(self):
        """Toggle automatic move playing."""
        self.auto_playing = not self.auto_playing
        if self.auto_playing:
            self.auto_button.config(text="⏸ Pause")
            self.next_move()
        else:
            self.auto_button.config(text="⏯ Auto Play")
    
    def reset_game(self):
        """Reset game to beginning."""
        self.current_move = 0
        self.auto_playing = False
        self.auto_button.config(text="⏯ Auto Play")
        self.game_board.draw_empty_board()
        self.status_label.config(text="Game reset")


class WorkingDashboard:
    """Enhanced training dashboard with agent selection and evaluation tracking."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AlphaZero Connect-4 Training Dashboard")
        self.root.geometry("1200x800")
        
        # Training data
        self.training_data = {
            'policy_losses': [],
            'value_losses': [],
            'training_examples': 0
        }
        
        # Evaluation tracking
        self.evaluation_data = {
            'iterations': [],
            'policy_win_rates': [],
            'value_win_rates': [],
            'policy_vs_minimax': [],
            'value_vs_minimax': [],
            'policy_vs_mcts': [],
            'value_vs_mcts': [],
            'policy_losses': [],
            'value_losses': []
        }
        
        # Trained networks storage
        self.trained_policy_net = None
        self.trained_value_net = None
        
        # Game window
        self.game_window = None
        
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the user interface."""
        # Control panel
        control_frame = ttk.LabelFrame(self.root, text="Training Controls")
        control_frame.pack(fill='x', padx=10, pady=10)
        
        # Parameters
        params_frame = ttk.Frame(control_frame)
        params_frame.pack(padx=10, pady=10)
        
        tk.Label(params_frame, text="Games:").grid(row=0, column=0, padx=5)
        self.games_var = tk.IntVar(value=15)
        tk.Entry(params_frame, textvariable=self.games_var, width=8).grid(row=0, column=1, padx=5)
        
        tk.Label(params_frame, text="Epochs:").grid(row=0, column=2, padx=5)
        self.epochs_var = tk.IntVar(value=10)
        tk.Entry(params_frame, textvariable=self.epochs_var, width=8).grid(row=0, column=3, padx=5)
        
        # Iterative training controls
        iter_frame = ttk.LabelFrame(control_frame, text="Iterative Training with Evaluation")
        iter_frame.pack(fill='x', padx=10, pady=5)
        
        iter_params_frame = ttk.Frame(iter_frame)
        iter_params_frame.pack(padx=10, pady=5)
        
        tk.Label(iter_params_frame, text="Iterations:").grid(row=0, column=0, padx=5)
        self.iterations_var = tk.IntVar(value=5)
        tk.Entry(iter_params_frame, textvariable=self.iterations_var, width=8).grid(row=0, column=1, padx=5)
        
        tk.Label(iter_params_frame, text="Eval Games:").grid(row=0, column=2, padx=5)
        self.eval_games_var = tk.IntVar(value=10)
        tk.Entry(iter_params_frame, textvariable=self.eval_games_var, width=8).grid(row=0, column=3, padx=5)
        
        tk.Label(iter_params_frame, text="Eval Every:").grid(row=0, column=4, padx=5)
        self.eval_frequency_var = tk.IntVar(value=1)
        tk.Entry(iter_params_frame, textvariable=self.eval_frequency_var, width=8).grid(row=0, column=5, padx=5)
        
        # Agent selection for AI games
        agent_frame = ttk.LabelFrame(control_frame, text="AI Game Agent Selection")
        agent_frame.pack(fill='x', padx=10, pady=5)
        
        agent_params_frame = ttk.Frame(agent_frame)
        agent_params_frame.pack(padx=10, pady=5)
        
        tk.Label(agent_params_frame, text="Player 1:").grid(row=0, column=0, padx=5)
        self.player1_var = tk.StringVar(value="Minimax")
        player1_combo = ttk.Combobox(agent_params_frame, textvariable=self.player1_var, 
                                   values=["Minimax", "Random", "MCTS", "Policy Network", "Value Network"], 
                                   width=15, state="readonly")
        player1_combo.grid(row=0, column=1, padx=5)
        
        tk.Label(agent_params_frame, text="Player 2:").grid(row=0, column=2, padx=5)
        self.player2_var = tk.StringVar(value="Random")
        player2_combo = ttk.Combobox(agent_params_frame, textvariable=self.player2_var,
                                   values=["Minimax", "Random", "MCTS", "Policy Network", "Value Network"],
                                   width=15, state="readonly")
        player2_combo.grid(row=0, column=3, padx=5)
        
        tk.Label(agent_params_frame, text="Games:").grid(row=0, column=4, padx=5)
        self.ai_games_var = tk.IntVar(value=3)
        tk.Entry(agent_params_frame, textvariable=self.ai_games_var, width=6).grid(row=0, column=5, padx=5)
        
        # Buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(pady=10)
        
        ttk.Button(button_frame, text="🎮 Single Demo", command=self.single_demo).pack(side='left', padx=5)
        ttk.Button(button_frame, text="🤖 AI Games", command=self.ai_games).pack(side='left', padx=5)
        ttk.Button(button_frame, text="🚀 Start Training", command=self.start_training).pack(side='left', padx=5)
        ttk.Button(button_frame, text="🔄 Iterative Training", command=self.start_iterative_training).pack(side='left', padx=5)
        ttk.Button(button_frame, text="📊 Update Charts", command=self.update_charts).pack(side='left', padx=5)
        ttk.Button(button_frame, text="🎬 Open Game Viewer", command=self.manual_open_game_window).pack(side='left', padx=5)
        
        # Status
        self.status_var = tk.StringVar(value="Ready to start")
        tk.Label(control_frame, textvariable=self.status_var, font=('Arial', 12)).pack(pady=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(control_frame, length=600, mode='determinate')
        self.progress.pack(pady=5)
        
        # Notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.setup_charts_tab()
        self.setup_evaluation_tab()
        self.setup_log_tab()
        self.setup_stats_tab()
    
    def setup_charts_tab(self):
        """Set up charts tab."""
        charts_frame = ttk.Frame(self.notebook)
        self.notebook.add(charts_frame, text="📈 Charts")
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(12, 8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, charts_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
        self.update_charts()
    
    def setup_evaluation_tab(self):
        """Set up evaluation metrics tab."""
        eval_frame = ttk.Frame(self.notebook)
        self.notebook.add(eval_frame, text="🎯 Evaluation")
        
        # Create matplotlib figure for evaluation charts
        self.eval_fig = Figure(figsize=(12, 8), dpi=100)
        self.eval_canvas = FigureCanvasTkAgg(self.eval_fig, eval_frame)
        self.eval_canvas.get_tk_widget().pack(fill='both', expand=True)
        
        self.update_evaluation_charts()
    
    def setup_log_tab(self):
        """Set up log tab."""
        log_frame = ttk.Frame(self.notebook)
        self.notebook.add(log_frame, text="📝 Log")
        
        self.log_text = tk.Text(log_frame, wrap='word', font=('Courier', 10))
        scrollbar = ttk.Scrollbar(log_frame, orient='vertical', command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        self.log_text.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        self.log("Dashboard ready!")
    
    def setup_stats_tab(self):
        """Set up statistics tab."""
        stats_frame = ttk.Frame(self.notebook)
        self.notebook.add(stats_frame, text="📊 Stats")
        
        self.stats_text = tk.Text(stats_frame, wrap='word', font=('Courier', 10))
        stats_scrollbar = ttk.Scrollbar(stats_frame, orient='vertical', command=self.stats_text.yview)
        self.stats_text.configure(yscrollcommand=stats_scrollbar.set)
        
        self.stats_text.pack(side='left', fill='both', expand=True)
        stats_scrollbar.pack(side='right', fill='y')
        
        self.update_stats()
    
    def log(self, message):
        """Add message to log."""
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.root.update()
    
    def update_status(self, message, progress=None):
        """Update status and progress."""
        self.status_var.set(message)
        if progress is not None:
            self.progress['value'] = progress
        self.root.update()
    
    def manual_open_game_window(self):
        """Open game viewer manually with demo game loaded."""
        self.manual_open = True
        self.open_game_window()
    
    def auto_load_demo(self):
        """Automatically load the demo game when game viewer is manually opened."""
        try:
            if self.game_window:
                # Valid Connect-4 game where red wins with 4 in a row horizontally
                moves = [2, 0, 3, 1, 4, 6, 5]  # Red: 2,3,4,5 (horizontal win) | Yellow: 0,1,6
                self.log("🎬 Loading demo game in viewer...")
                self.log("  Demo: Red wins with horizontal line (columns 2,3,4,5)")
                
                # Show the game but don't auto-start it
                self.game_window.show_game(moves, "Red Player (Demo)", "Yellow Player (Demo)", 1)
                self.log("✅ Demo game loaded! Use controls to play/pause.")
        except Exception as e:
            self.log(f"❌ Failed to load demo: {str(e)}")
    
    def open_game_window(self):
        """Open game visualization window."""
        # Check if game window exists and is still valid
        if self.game_window is None or not hasattr(self.game_window, 'window') or not self.game_window.window.winfo_exists():
            self.game_window = GameWindow(self)
            self.log("Game viewer opened")
            
            # Automatically load the demo game when manually opening the viewer
            if hasattr(self, 'manual_open') and self.manual_open:
                self.auto_load_demo()
                self.manual_open = False
        else:
            # Window exists, just bring it to front
            try:
                self.game_window.window.lift()
                self.game_window.window.focus_force()
                self.log("Game viewer brought to front")
            except tk.TclError:
                # Window was destroyed, create new one
                self.game_window = GameWindow(self)
                self.log("Game viewer recreated")
                
                # Auto-load demo for recreated window too
                if hasattr(self, 'manual_open') and self.manual_open:
                    self.auto_load_demo()
                    self.manual_open = False
    
    def single_demo(self):
        """Show a single demo game with a valid Connect-4 winning sequence."""
        try:
            self.log("🎮 Starting single demo game...")
            self.open_game_window()
            
            if self.game_window:
                # Valid Connect-4 game where red wins with 4 in a row horizontally
                moves = [2, 0, 3, 1, 4, 6, 5]  # Red: 2,3,4,5 (horizontal win) | Yellow: 0,1,6
                self.log("📺 Showing valid Connect-4 demo game...")
                self.log("  Red plays: columns 2, 3, 4, 5 (horizontal win)")
                self.log("  Yellow plays: columns 0, 1, 6")
                
                # Show single game and start auto-play immediately
                self.game_window.show_game(moves, "Red Player", "Yellow Player", 1)
                
                # Start auto-play automatically
                self.game_window.auto_playing = True
                self.game_window.auto_button.config(text="⏸ Pause")
                self.game_window.next_move()
                
                self.log("✅ Demo game started with auto-play!")
            else:
                self.log("❌ Could not open game window")
                
        except Exception as e:
            self.log(f"❌ Demo failed: {str(e)}")
    
    def create_agent(self, agent_type):
        """Create an agent based on the specified type."""
        try:
            if agent_type == "Minimax":
                return MinimaxAgent(depth=3)
            elif agent_type == "Random":
                return RandomAgent()
            elif agent_type == "MCTS":
                return MCTSAgent(simulations=1000, name="MCTS Agent")
            elif agent_type == "Policy Network":
                if self.trained_policy_net is None:
                    self.log("⚠️ No trained policy network available, using random agent")
                    return RandomAgent()
                agent = PolicyAgent(name="Policy Agent")
                agent.set_network(self.trained_policy_net)
                return agent
            elif agent_type == "Value Network":
                if self.trained_value_net is None:
                    self.log("⚠️ No trained value network available, using random agent")
                    return RandomAgent()
                agent = ValueAgent(name="Value Agent")
                agent.set_network(self.trained_value_net)
                return agent
            else:
                self.log(f"⚠️ Unknown agent type: {agent_type}, using random agent")
                return RandomAgent()
        except Exception as e:
            self.log(f"❌ Failed to create {agent_type} agent: {str(e)}")
            return RandomAgent()
    
    def get_agent_action(self, agent, env):
        """Get action from any agent type, handling different interfaces."""
        try:
            # Check if agent uses the base agent interface (board, action_mask)
            if hasattr(agent, 'select_action'):
                # Try to call with (board, action_mask) first (neural agents)
                try:
                    # Set player ID for neural network agents
                    if hasattr(agent, 'set_player_id'):
                        agent.set_player_id(env.current_player)
                    
                    board = env.board.copy()
                    action_mask = env._action_mask()
                    return agent.select_action(board, action_mask)
                except TypeError:
                    # Fallback to env interface (minimax, random agents)
                    return agent.select_action(env)
            else:
                self.log(f"⚠️ Agent {agent.name} has no select_action method")
                return 0
        except Exception as e:
            self.log(f"❌ Error getting action from {agent.name}: {str(e)}")
            return 0
    
    def ai_games(self):
        """Generate and show real AI vs AI games."""
        try:
            # Get selected agents and number of games
            player1_type = self.player1_var.get()
            player2_type = self.player2_var.get()
            num_games = self.ai_games_var.get()
            
            self.log("🤖 Starting real AI games...")
            self.log(f"  {player1_type} vs {player2_type} ({num_games} games)")
            
            # Create agents based on selection
            try:
                player1_agent = self.create_agent(player1_type)
                player2_agent = self.create_agent(player2_type)
                self.log(f"  ✓ Agents created: {player1_agent.name} vs {player2_agent.name}")
            except Exception as e:
                self.log(f"  ❌ Failed to create agents: {str(e)}")
                return
            
            self.open_game_window()
            
            if not self.game_window:
                self.log("❌ Could not open game window")
                return
            
            # Additional check to make sure window is valid
            try:
                if not self.game_window.window.winfo_exists():
                    self.log("❌ Game window is not valid")
                    self.game_window = None
                    return
            except (AttributeError, tk.TclError):
                self.log("❌ Game window error, recreating...")
                self.game_window = GameWindow(self)
                if not self.game_window:
                    self.log("❌ Failed to recreate game window")
                    return
            
            # Generate all games first
            self.log("🎯 Generating all games...")
            games_data = []
            wins = {player1_agent.name: 0, player2_agent.name: 0, "Tie": 0}
            
            for game_num in range(num_games):
                self.update_status(f"Generating game {game_num + 1}/{num_games}...", 
                                 int((game_num / num_games) * 100))
                self.log(f"🎯 Generating game {game_num + 1}/{num_games}...")
                
                try:
                    # Generate game with timeout protection
                    moves, p1_name, p2_name, winner = self.play_ai_game_safe(player1_agent, player2_agent, game_num + 1)
                    
                    if moves:
                        games_data.append((moves, p1_name, p2_name, winner))
                        
                        # Update win statistics
                        if winner == 1:
                            wins[p1_name] += 1
                            self.log(f"  🏆 Winner: {p1_name}")
                        elif winner == -1:
                            wins[p2_name] += 1
                            self.log(f"  🏆 Winner: {p2_name}")
                        else:
                            wins["Tie"] += 1
                            self.log(f"  🤝 Game ended in a tie")
                        
                        self.log(f"  ✅ Game {game_num + 1} generated ({len(moves)} moves)")
                    else:
                        self.log(f"❌ Game {game_num + 1} failed to generate")
                        
                except Exception as e:
                    self.log(f"❌ Error in game {game_num + 1}: {str(e)}")
            
            # Show final statistics
            self.log("� All games generated! Final Results:")
            for agent, count in wins.items():
                percentage = (count / num_games) * 100
                self.log(f"  {agent}: {count}/{num_games} ({percentage:.1f}%)")
            
            # Now show all games automatically in sequence
            if games_data:
                self.log("🎬 Starting automatic game playback...")
                self.update_status("Playing games automatically...", 100)
                self.game_window.show_games_sequence(games_data)
            else:
                self.log("❌ No games to display")
                self.update_status("No games generated", 0)
            
        except Exception as e:
            self.log(f"❌ AI games failed: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
    
    def play_ai_game_safe(self, player1_agent, player2_agent, game_num):
        """Play a single AI game with safety checks and timeout."""
        try:
            self.log(f"  🎮 Initializing game {game_num}...")
            
            # Create environment with error handling
            try:
                env = GymnasiumConnectFour()
                observation, info = env.reset()
                self.log(f"  ✓ Environment created for game {game_num}")
            except Exception as e:
                self.log(f"  ❌ Environment creation failed: {str(e)}")
                return [], "", "", 0
            
            # Use provided agents
            p1_name = player1_agent.name
            p2_name = player2_agent.name
            
            self.log(f"  🎯 Game setup: {p1_name} vs {p2_name}")
            
            moves = []
            move_count = 0
            terminated = False
            max_moves = 42  # Safety limit
            info = {}  # Initialize info to avoid scope issues
            reward = 0  # Initialize reward
            
            while not terminated and move_count < max_moves:
                try:
                    # Log current state
                    self.log(f"    Turn {move_count + 1}: Player {env.current_player}'s turn")
                    
                    # Agent selection with timeout protection
                    start_time = time.time()
                    
                    if (env.current_player == 1):
                        # Player 1's turn
                        self.log(f"    {player1_agent.name} thinking...")
                        action = self.get_agent_action(player1_agent, env)
                        agent_name = player1_agent.name
                    else:
                        # Player 2's turn
                        self.log(f"    {player2_agent.name} thinking...")
                        action = self.get_agent_action(player2_agent, env)
                        agent_name = player2_agent.name
                    
                    think_time = time.time() - start_time
                    self.log(f"    Move {move_count + 1}: {agent_name} plays column {action} (took {think_time:.2f}s)")
                    
                    # Validate action
                    if action < 0 or action >= 7:
                        self.log(f"    ❌ Invalid action: {action}")
                        break
                        
                    if env.board[0, action] != 0:
                        self.log(f"    ❌ Column {action} is full")
                        break
                    
                    moves.append(action)
                    
                    # Make move - fix the unpacking issue
                    try:
                        step_result = env.step(action)
                        if len(step_result) == 6:
                            # Custom environment returns 6 values including action
                            observation, reward, terminated, truncated, info, returned_action = step_result
                        elif len(step_result) == 5:
                            observation, reward, terminated, truncated, info = step_result
                        else:
                            # Handle different return format
                            observation, reward, terminated, info = step_result
                            truncated = False
                        
                        self.log(f"    Step result: reward={reward}, terminated={terminated}")
                        
                    except Exception as step_error:
                        self.log(f"    ❌ Step error: {str(step_error)}")
                        break
                    
                    move_count += 1
                    
                    # Check for timeout
                    if think_time > 10:
                        self.log(f"    ⚠️ Move took too long: {think_time:.2f}s")
                        
                except Exception as move_error:
                    self.log(f"    ❌ Move error: {str(move_error)}")
                    break
            
            # Determine winner
            winner = 0
            if 'winner' in info:
                winner = info['winner']
            elif terminated and reward != 0:
                winner = env.current_player * -1  # Previous player won
            
            winner_name = "Draw"
            if winner == 1:
                winner_name = p1_name
            elif winner == -1:
                winner_name = p2_name
                
            self.log(f"  🏆 Game {game_num} result: {winner_name} wins! ({len(moves)} moves)")
            
            return moves, p1_name, p2_name, winner
            
        except Exception as e:
            self.log(f"    ❌ Game {game_num} generation error: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
            return [], "", "", 0
    
    def start_training(self):
        """Start the training process."""
        try:
            # Get parameters
            num_games = self.games_var.get()
            epochs = self.epochs_var.get()
            
            self.log(f"🚀 Starting training: {num_games} games, {epochs} epochs")
            
            # Create trainer and networks
            self.update_status("Creating networks...", 5)
            trainer = Trainer(save_dir="models")
            policy_net = PolicyNetwork(hidden_size=128)
            value_net = ValueNetwork(hidden_size=128)
            
            self.log(f"✓ Networks created - Policy: {policy_net.get_num_parameters():,} params")
            
            # Generate training data
            self.update_status("Generating training data...", 20)
            self.log("🎮 Generating training games...")
            
            policy_data, value_data = trainer.generate_minimax_training_data(
                num_games=num_games,
                minimax_depth=4
            )
            
            self.training_data['training_examples'] = len(policy_data)
            self.log(f"✓ Generated {len(policy_data)} training examples")
            
            # Train policy network
            self.update_status("Training policy network...", 50)
            self.log("🧠 Training policy network...")
            
            policy_results = trainer.train_policy_network(
                policy_net, policy_data, epochs, batch_size=16, learning_rate=0.001
            )
            
            self.training_data['policy_losses'] = policy_results.get('epoch_losses', [])
            self.log(f"✓ Policy training complete - Final loss: {policy_results['final_loss']:.6f}")
            
            # Train value network
            self.update_status("Training value network...", 75)
            self.log("🧠 Training value network...")
            
            value_results = trainer.train_value_network(
                value_net, value_data, epochs, batch_size=16, learning_rate=0.001
            )
            
            self.training_data['value_losses'] = value_results.get('epoch_losses', [])
            self.log(f"✓ Value training complete - Final loss: {value_results['final_loss']:.6f}")
            
            # Test networks
            self.update_status("Testing networks...", 90)
            self.test_networks(policy_net, value_net)
            
            # Store trained networks for agent selection
            self.trained_policy_net = policy_net
            self.trained_value_net = value_net
            self.log("✓ Trained networks stored for AI games")
            
            # Complete
            self.update_status("Training complete!", 100)
            self.update_charts()
            self.update_stats()
            self.log("🎉 Training session complete!")
            self.log("🎮 You can now test trained networks in AI Games!")
            
        except Exception as e:
            self.log(f"❌ Training failed: {str(e)}")
            self.update_status("Training failed")
            import traceback
            self.log(traceback.format_exc())
    
    def start_iterative_training(self):
        """Start iterative training with evaluation tracking."""
        try:
            # Get parameters
            num_games = self.games_var.get()
            epochs = self.epochs_var.get()
            iterations = self.iterations_var.get()
            eval_games = self.eval_games_var.get()
            eval_frequency = self.eval_frequency_var.get()
            
            self.log(f"🔄 Starting iterative training: {iterations} iterations")
            self.log(f"  Parameters: {num_games} games, {epochs} epochs per iteration")
            self.log(f"  Evaluation: {eval_games} games every {eval_frequency} iterations")
            
            # Create trainer and networks (starts fresh each time for now)
            self.update_status("Creating networks...", 5)
            trainer = Trainer(save_dir="models")
            policy_net = PolicyNetwork(hidden_size=128)
            value_net = ValueNetwork(hidden_size=128)
            
            self.log(f"✓ Networks created - Policy: {policy_net.get_num_parameters():,} params")
            
            # Clear previous evaluation data
            self.evaluation_data = {
                'iterations': [],
                'policy_win_rates': [],
                'value_win_rates': [],
                'policy_vs_minimax': [],
                'value_vs_minimax': [],
                'policy_vs_mcts': [],
                'value_vs_mcts': [],
                'policy_losses': [],
                'value_losses': []
            }
            
            # Training loop
            for iteration in range(iterations):
                iteration_progress = int(10 + (iteration / iterations) * 80)
                self.update_status(f"Iteration {iteration + 1}/{iterations} - Generating data...", iteration_progress)
                self.log(f"🔄 === Iteration {iteration + 1}/{iterations} ===")
                
                # Generate training data
                self.log("🎮 Generating training games...")
                policy_data, value_data = trainer.generate_minimax_training_data(
                    num_games=num_games,
                    minimax_depth=4
                )
                self.log(f"✓ Generated {len(policy_data)} training examples")
                
                # Train policy network
                self.log("🧠 Training policy network...")
                policy_results = trainer.train_policy_network(
                    policy_net, policy_data, epochs, batch_size=16, learning_rate=0.001
                )
                self.log(f"✓ Policy training complete - Loss: {policy_results['final_loss']:.6f}")
                
                # Train value network
                self.log("🧠 Training value network...")
                value_results = trainer.train_value_network(
                    value_net, value_data, epochs, batch_size=16, learning_rate=0.001
                )
                self.log(f"✓ Value training complete - Loss: {value_results['final_loss']:.6f}")
                
                # Evaluation phase
                if (iteration + 1) % eval_frequency == 0:
                    self.update_status(f"Iteration {iteration + 1}/{iterations} - Evaluating...", iteration_progress + 5)
                    self.log(f"🎯 Evaluating networks (iteration {iteration + 1})...")
                    
                    # Create agents for evaluation
                    policy_agent = PolicyAgent(name="Policy Agent")
                    policy_agent.set_network(policy_net.eval())
                    
                    value_agent = ValueAgent(name="Value Agent")
                    value_agent.set_network(value_net.eval())
                    
                    # Evaluate against random
                    policy_vs_random = trainer.evaluate_agent_vs_baseline(policy_agent, num_games=eval_games)
                    value_vs_random = trainer.evaluate_agent_vs_baseline(value_agent, num_games=eval_games)
                    
                    # Evaluate against minimax
                    minimax_agent = MinimaxAgent(depth=3)
                    policy_vs_minimax = trainer.evaluate_agent_vs_baseline(policy_agent, minimax_agent, num_games=eval_games)
                    value_vs_minimax = trainer.evaluate_agent_vs_baseline(value_agent, minimax_agent, num_games=eval_games)
                    
                    # Evaluate against MCTS
                    mcts_agent = MCTSAgent(simulations=500, name="MCTS Agent")  # Use fewer sims for faster eval
                    policy_vs_mcts = trainer.evaluate_agent_vs_baseline(policy_agent, mcts_agent, num_games=eval_games)
                    value_vs_mcts = trainer.evaluate_agent_vs_baseline(value_agent, mcts_agent, num_games=eval_games)
                    
                    # Store results
                    self.evaluation_data['iterations'].append(iteration + 1)
                    self.evaluation_data['policy_win_rates'].append(policy_vs_random['win_rate'])
                    self.evaluation_data['value_win_rates'].append(value_vs_random['win_rate'])
                    self.evaluation_data['policy_vs_minimax'].append(policy_vs_minimax['win_rate'])
                    self.evaluation_data['value_vs_minimax'].append(value_vs_minimax['win_rate'])
                    self.evaluation_data['policy_vs_mcts'].append(policy_vs_mcts['win_rate'])
                    self.evaluation_data['value_vs_mcts'].append(value_vs_mcts['win_rate'])
                    self.evaluation_data['policy_losses'].append(policy_results['final_loss'])
                    self.evaluation_data['value_losses'].append(value_results['final_loss'])
                    
                    self.log(f"📊 Policy vs Random: {policy_vs_random['win_rate']:.2%}")
                    self.log(f"📊 Value vs Random: {value_vs_random['win_rate']:.2%}")
                    self.log(f"📊 Policy vs Minimax: {policy_vs_minimax['win_rate']:.2%}")
                    self.log(f"📊 Value vs Minimax: {value_vs_minimax['win_rate']:.2%}")
                    self.log(f"📊 Policy vs MCTS: {policy_vs_mcts['win_rate']:.2%}")
                    self.log(f"📊 Value vs MCTS: {value_vs_mcts['win_rate']:.2%}")
                    
                    # Update evaluation charts
                    self.update_evaluation_charts()
            
            # Store final networks
            self.trained_policy_net = policy_net
            self.trained_value_net = value_net
            self.log("✓ Trained networks stored for AI games")
            
            # Complete
            self.update_status("Iterative training complete!", 100)
            self.update_charts()
            self.update_evaluation_charts()
            self.update_stats()
            self.log("🎉 Iterative training session complete!")
            self.log("🎮 Check the Evaluation tab to see performance metrics!")
            
        except Exception as e:
            self.log(f"❌ Iterative training failed: {str(e)}")
            self.update_status("Training failed")
            import traceback
            self.log(traceback.format_exc())
    
    def test_networks(self, policy_net, value_net):
        """Test the trained networks."""
        try:
            # Test on empty board
            empty_board = np.zeros((6, 7))
            board_tensor = torch.FloatTensor(empty_board).unsqueeze(0)
            
            # Policy test
            policy_net.eval()
            with torch.no_grad():
                policy_logits = policy_net(board_tensor)
                policy_probs = torch.softmax(policy_logits, dim=1).squeeze().numpy()
                best_action = np.argmax(policy_probs)
            
            # Value test
            value_net.eval()
            with torch.no_grad():
                value = value_net(board_tensor).item()
            
            self.log(f"🧪 Network test - Prefers column {best_action}, Value: {value:.3f}")
            
        except Exception as e:
            self.log(f"❌ Network test failed: {str(e)}")
    
    def update_charts(self):
        """Update training charts."""
        self.fig.clear()
        
        if not (self.training_data['policy_losses'] or self.training_data['value_losses']):
            ax = self.fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, 'No training data yet.\nClick "Start Training" to begin!',
                   ha='center', va='center', fontsize=16, transform=ax.transAxes)
            ax.set_title('AlphaZero Connect-4 Training')
        else:
            # Create subplots
            ax1 = self.fig.add_subplot(2, 1, 1)
            ax2 = self.fig.add_subplot(2, 1, 2)
            
            # Policy loss
            if self.training_data['policy_losses']:
                ax1.plot(self.training_data['policy_losses'], 'b-', linewidth=2)
                ax1.set_title('Policy Network Loss')
                ax1.set_ylabel('Loss')
                ax1.grid(True, alpha=0.3)
            
            # Value loss
            if self.training_data['value_losses']:
                ax2.plot(self.training_data['value_losses'], 'r-', linewidth=2)
                ax2.set_title('Value Network Loss')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Loss')
                ax2.grid(True, alpha=0.3)
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def update_evaluation_charts(self):
        """Update evaluation metrics charts."""
        self.eval_fig.clear()
        
        if not self.evaluation_data['iterations']:
            ax = self.eval_fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, 'No evaluation data yet.\nRun "Iterative Training" to begin!',
                   ha='center', va='center', fontsize=16, transform=ax.transAxes)
            ax.set_title('Evaluation Metrics')
        else:
            # Create subplots (3x3 layout to include loss curves)
            ax1 = self.eval_fig.add_subplot(3, 3, 1)
            ax2 = self.eval_fig.add_subplot(3, 3, 2)
            ax3 = self.eval_fig.add_subplot(3, 3, 3)
            ax4 = self.eval_fig.add_subplot(3, 3, 4)
            ax5 = self.eval_fig.add_subplot(3, 3, 5)
            ax6 = self.eval_fig.add_subplot(3, 3, 6)
            ax7 = self.eval_fig.add_subplot(3, 3, 7)
            ax8 = self.eval_fig.add_subplot(3, 3, 8)
            ax9 = self.eval_fig.add_subplot(3, 3, 9)
            
            iterations = self.evaluation_data['iterations']
            
            # Win rates vs Random
            ax1.plot(iterations, self.evaluation_data['policy_win_rates'], 'b-o', label='Policy', linewidth=2)
            ax1.plot(iterations, self.evaluation_data['value_win_rates'], 'r-s', label='Value', linewidth=2)
            ax1.set_title('Win Rate vs Random Agent')
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Win Rate')
            ax1.set_ylim(0, 1)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Win rates vs Minimax
            ax2.plot(iterations, self.evaluation_data['policy_vs_minimax'], 'b-o', label='Policy', linewidth=2)
            ax2.plot(iterations, self.evaluation_data['value_vs_minimax'], 'r-s', label='Value', linewidth=2)
            ax2.set_title('Win Rate vs Minimax Agent')
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Win Rate')
            ax2.set_ylim(0, 1)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Win rates vs MCTS
            ax3.plot(iterations, self.evaluation_data['policy_vs_mcts'], 'b-o', label='Policy', linewidth=2)
            ax3.plot(iterations, self.evaluation_data['value_vs_mcts'], 'r-s', label='Value', linewidth=2)
            ax3.set_title('Win Rate vs MCTS Agent')
            ax3.set_xlabel('Iteration')
            ax3.set_ylabel('Win Rate')
            ax3.set_ylim(0, 1)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Policy network progression (all opponents)
            ax4.plot(iterations, self.evaluation_data['policy_win_rates'], 'b-', label='vs Random', linewidth=2)
            ax4.plot(iterations, self.evaluation_data['policy_vs_minimax'], 'b--', label='vs Minimax', linewidth=2)
            ax4.plot(iterations, self.evaluation_data['policy_vs_mcts'], 'b:', label='vs MCTS', linewidth=2)
            ax4.set_title('Policy Network Performance')
            ax4.set_xlabel('Iteration')
            ax4.set_ylabel('Win Rate')
            ax4.set_ylim(0, 1)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # Value network progression (all opponents)
            ax5.plot(iterations, self.evaluation_data['value_win_rates'], 'r-', label='vs Random', linewidth=2)
            ax5.plot(iterations, self.evaluation_data['value_vs_minimax'], 'r--', label='vs Minimax', linewidth=2)
            ax5.plot(iterations, self.evaluation_data['value_vs_mcts'], 'r:', label='vs MCTS', linewidth=2)
            ax5.set_title('Value Network Performance')
            ax5.set_xlabel('Iteration')
            ax5.set_ylabel('Win Rate')
            ax5.set_ylim(0, 1)
            ax5.legend()
            ax5.grid(True, alpha=0.3)
            
            # Comparative performance summary
            if len(iterations) > 0:
                latest_idx = -1
                policy_random = self.evaluation_data['policy_win_rates'][latest_idx]
                policy_minimax = self.evaluation_data['policy_vs_minimax'][latest_idx]
                policy_mcts = self.evaluation_data['policy_vs_mcts'][latest_idx]
                value_random = self.evaluation_data['value_win_rates'][latest_idx]
                value_minimax = self.evaluation_data['value_vs_minimax'][latest_idx]
                value_mcts = self.evaluation_data['value_vs_mcts'][latest_idx]
                
                agents = ['vs Random', 'vs Minimax', 'vs MCTS']
                policy_scores = [policy_random, policy_minimax, policy_mcts]
                value_scores = [value_random, value_minimax, value_mcts]
                
                x = np.arange(len(agents))
                width = 0.35
                
                ax6.bar(x - width/2, policy_scores, width, label='Policy', color='blue', alpha=0.7)
                ax6.bar(x + width/2, value_scores, width, label='Value', color='red', alpha=0.7)
                ax6.set_title('Latest Performance Summary')
                ax6.set_ylabel('Win Rate')
                ax6.set_xticks(x)
                ax6.set_xticklabels(agents)
                ax6.set_ylim(0, 1)
                ax6.legend()
                ax6.grid(True, alpha=0.3)
            
            # Policy Loss Curve
            if self.evaluation_data['policy_losses']:
                ax7.plot(iterations, self.evaluation_data['policy_losses'], 'b-o', linewidth=2)
                ax7.set_title('Policy Network Loss')
                ax7.set_xlabel('Iteration')
                ax7.set_ylabel('Loss')
                ax7.grid(True, alpha=0.3)
                ax7.set_yscale('log')  # Log scale for better loss visualization
            
            # Value Loss Curve
            if self.evaluation_data['value_losses']:
                ax8.plot(iterations, self.evaluation_data['value_losses'], 'r-o', linewidth=2)
                ax8.set_title('Value Network Loss')
                ax8.set_xlabel('Iteration')
                ax8.set_ylabel('Loss')
                ax8.grid(True, alpha=0.3)
                ax8.set_yscale('log')  # Log scale for better loss visualization
            
            # Combined Loss Comparison
            if self.evaluation_data['policy_losses'] and self.evaluation_data['value_losses']:
                ax9.plot(iterations, self.evaluation_data['policy_losses'], 'b-o', label='Policy', linewidth=2)
                ax9.plot(iterations, self.evaluation_data['value_losses'], 'r-s', label='Value', linewidth=2)
                ax9.set_title('Combined Loss Comparison')
                ax9.set_xlabel('Iteration')
                ax9.set_ylabel('Loss')
                ax9.set_yscale('log')  # Log scale for better loss visualization
                ax9.legend()
                ax9.grid(True, alpha=0.3)
        
        self.eval_fig.tight_layout()
        self.eval_canvas.draw()
    
    def update_stats(self):
        """Update statistics display."""
        stats = "=== Training Statistics ===\n\n"
        
        if self.training_data['training_examples']:
            stats += f"Training Examples: {self.training_data['training_examples']}\n\n"
        
        if self.training_data['policy_losses']:
            final_policy = self.training_data['policy_losses'][-1]
            stats += f"Policy Network:\n"
            stats += f"  Final Loss: {final_policy:.6f}\n"
            stats += f"  Epochs: {len(self.training_data['policy_losses'])}\n\n"
        
        if self.training_data['value_losses']:
            final_value = self.training_data['value_losses'][-1]
            stats += f"Value Network:\n"
            stats += f"  Final Loss: {final_value:.6f}\n"
            stats += f"  Epochs: {len(self.training_data['value_losses'])}\n\n"
        
        # Add evaluation statistics
        if self.evaluation_data['iterations']:
            stats += "=== Evaluation Results ===\n\n"
            
            if self.evaluation_data['policy_win_rates']:
                latest_policy_random = self.evaluation_data['policy_win_rates'][-1]
                latest_policy_minimax = self.evaluation_data['policy_vs_minimax'][-1]
                stats += f"Policy Network (Latest):\n"
                stats += f"  vs Random: {latest_policy_random:.2%}\n"
                stats += f"  vs Minimax: {latest_policy_minimax:.2%}\n\n"
            
            if self.evaluation_data['value_win_rates']:
                latest_value_random = self.evaluation_data['value_win_rates'][-1]
                latest_value_minimax = self.evaluation_data['value_vs_minimax'][-1]
                stats += f"Value Network (Latest):\n"
                stats += f"  vs Random: {latest_value_random:.2%}\n"
                stats += f"  vs Minimax: {latest_value_minimax:.2%}\n\n"
            
            stats += f"Iterations Completed: {len(self.evaluation_data['iterations'])}\n\n"
        
        if not any(self.training_data.values()) and not self.evaluation_data['iterations']:
            stats += "No training data yet.\nStart training to see statistics!"
        
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, stats)
    
    def run(self):
        """Run the dashboard."""
        self.root.mainloop()


if __name__ == "__main__":
    dashboard = WorkingDashboard()
    dashboard.run()