import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import random
import numpy as np

class ConnectFourGUI:
    def __init__(self, env):
        self.env = env
        self.root = tk.Tk()
        self.root.title("Connect Four - Interactive Game")
        self.root.configure(bg='#2c3e50')
        
        # Game state
        self.game_mode = "human_vs_human"  # "human_vs_human", "human_vs_ai", "ai_vs_ai"
        self.game_running = False
        self.game_paused = False
        self.animation_speed = 0.1  # seconds between AI moves
        
        # Colors
        self.colors = {
            'background': '#34495e',
            'board': '#3498db',
            'player1': '#e74c3c',
            'player2': '#f1c40f',
            'empty': '#ecf0f1',
            'highlight': '#2ecc71',
            'text': '#ffffff',
            'button_bg': '#5dade2',
            'button_fg': '#2c3e50'
        }
        
        self.setup_ui()
        self.cell_size = 60
        self.draw_board()
        
    def setup_ui(self):
        # Main frame
        main_frame = tk.Frame(self.root, bg=self.colors['background'])
        main_frame.pack(padx=20, pady=20)
        
        # Title
        title_label = tk.Label(main_frame, text="Connect Four", 
                              font=('Arial', 24, 'bold'), 
                              fg=self.colors['text'], 
                              bg=self.colors['background'])
        title_label.pack(pady=(0, 20))
        
        # Control panel
        control_frame = tk.Frame(main_frame, bg=self.colors['background'])
        control_frame.pack(pady=(0, 20))
        
        # Game mode selection
        mode_frame = tk.Frame(control_frame, bg=self.colors['background'])
        mode_frame.pack(side=tk.LEFT, padx=(0, 20))
        
        tk.Label(mode_frame, text="Game Mode:", 
                font=('Arial', 12, 'bold'), 
                fg=self.colors['text'], 
                bg=self.colors['background']).pack()
        
        self.mode_var = tk.StringVar(value=self.game_mode)
        modes = [("Human vs Human", "human_vs_human"), 
                ("Human vs AI", "human_vs_ai"), 
                ("AI vs AI", "ai_vs_ai")]
        
        for text, mode in modes:
            tk.Radiobutton(mode_frame, text=text, variable=self.mode_var, 
                          value=mode, fg=self.colors['text'], 
                          bg=self.colors['background'], 
                          selectcolor=self.colors['board'],
                          command=self.change_mode).pack(anchor=tk.W)
        
        # Control buttons
        button_frame = tk.Frame(control_frame, bg=self.colors['background'])
        button_frame.pack(side=tk.LEFT)
        
        self.start_button = tk.Button(button_frame, text="Start Game", 
                 command=self.start_game, 
                 bg='white', 
                 fg='black', font=('Arial', 10, 'bold'),
                 relief='flat', borderwidth=1)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.pause_button = tk.Button(button_frame, text="Pause", 
                 command=self.toggle_pause, 
                 bg='white', 
                 fg='black', font=('Arial', 10, 'bold'),
                 relief='flat', borderwidth=1,
                 state=tk.DISABLED)
        self.pause_button.pack(side=tk.LEFT, padx=5)
        
        self.reset_button = tk.Button(button_frame, text="Reset", 
                 command=self.reset_game, 
                 bg='white', 
                 fg='black', font=('Arial', 10, 'bold'),
                 relief='flat', borderwidth=1)
        self.reset_button.pack(side=tk.LEFT, padx=5)
        
        # Speed control for AI games
        speed_frame = tk.Frame(control_frame, bg=self.colors['background'])
        speed_frame.pack(side=tk.LEFT, padx=(20, 0))
        
        tk.Label(speed_frame, text="AI Speed:", 
                font=('Arial', 10, 'bold'), 
                fg=self.colors['text'], 
                bg=self.colors['background']).pack()
        
        self.speed_var = tk.DoubleVar(value=self.animation_speed)
        speed_scale = tk.Scale(speed_frame, from_=0.1, to=2.0, resolution=0.1,
                              orient=tk.HORIZONTAL, variable=self.speed_var,
                              bg=self.colors['background'], fg=self.colors['text'],
                              highlightbackground=self.colors['background'],
                              command=self.update_speed)
        speed_scale.pack()
        
        # Game canvas
        self.canvas = tk.Canvas(main_frame, width=420, height=360, 
                               bg=self.colors['board'], highlightthickness=2,
                               highlightbackground=self.colors['text'])
        self.canvas.pack(pady=(0, 20))
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Motion>", self.on_hover)
        
        # Status display
        self.status_label = tk.Label(main_frame, text="Ready to play!", 
                                    font=('Arial', 14, 'bold'), 
                                    fg=self.colors['text'], 
                                    bg=self.colors['background'])
        self.status_label.pack()
        
        # Current player indicator
        self.player_label = tk.Label(main_frame, text="Player 1's Turn", 
                                    font=('Arial', 12), 
                                    fg=self.colors['player1'], 
                                    bg=self.colors['background'])
        self.player_label.pack(pady=(5, 0))

    def change_mode(self):
        self.game_mode = self.mode_var.get()
        if not self.game_running:
            self.reset_game()
    
    def update_speed(self, value):
        self.animation_speed = float(value)
    
    def start_game(self):
        if not self.game_running:
            self.game_running = True
            self.game_paused = False
            self.start_button.config(state=tk.DISABLED)
            self.pause_button.config(state=tk.NORMAL)
            self.update_status("Game started!")
            
            if self.game_mode == "ai_vs_ai":
                threading.Thread(target=self.run_ai_game, daemon=True).start()
            elif self.game_mode == "human_vs_ai" and self.env.current_player == -1:
                threading.Thread(target=self.make_ai_move, daemon=True).start()
    
    def toggle_pause(self):
        self.game_paused = not self.game_paused
        self.pause_button.config(text="Resume" if self.game_paused else "Pause")
        self.update_status("Game paused" if self.game_paused else "Game resumed")
    
    def reset_game(self):
        self.game_running = False
        self.game_paused = False
        self.env.reset()
        self.draw_board()
        self.start_button.config(state=tk.NORMAL)
        self.pause_button.config(state=tk.DISABLED, text="Pause")
        self.update_status("Game reset - Ready to play!")
        self.update_player_display()
    
    def update_status(self, message):
        self.status_label.config(text=message)
        self.root.update()
    
    def update_player_display(self):
        if self.env.current_player == 1:
            self.player_label.config(text="Player 1's Turn (Red)", 
                                   fg=self.colors['player1'])
        else:
            self.player_label.config(text="Player 2's Turn (Yellow)", 
                                   fg=self.colors['player2'])

    def draw_board(self):
        self.canvas.delete("all")
        
        # Draw board background
        self.canvas.create_rectangle(0, 0, 420, 360, fill=self.colors['board'], outline="")
        
        # Draw grid and pieces
        for r in range(6):
            for c in range(7):
                x0 = c * self.cell_size + 10
                y0 = r * self.cell_size + 10
                x1 = x0 + self.cell_size - 20
                y1 = y0 + self.cell_size - 20
                
                cell = self.env.board[r, c]
                if cell == 1:
                    color = self.colors['player1']
                elif cell == -1:
                    color = self.colors['player2']
                else:
                    color = self.colors['empty']
                
                # Create circle for piece
                self.canvas.create_oval(x0, y0, x1, y1, fill=color, 
                                      outline='#2c3e50', width=2, 
                                      tags=f"piece_{r}_{c}")
        
        # Draw column numbers
        for c in range(7):
            x = c * self.cell_size + self.cell_size // 2 + 10
            self.canvas.create_text(x, 370, text=str(c), fill=self.colors['text'], 
                                  font=('Arial', 12, 'bold'), tags="column_label")
        
        self.root.update()

    def on_hover(self, event):
        if not self.game_running or self.game_paused:
            return
            
        # Show preview of where piece would go
        col = (event.x - 10) // self.cell_size
        if 0 <= col < 7 and self.env.board[0, col] == 0:
            # Find the row where piece would land
            for row in range(5, -1, -1):
                if self.env.board[row, col] == 0:
                    # Highlight this position
                    x0 = col * self.cell_size + 10
                    y0 = row * self.cell_size + 10
                    x1 = x0 + self.cell_size - 20
                    y1 = y0 + self.cell_size - 20
                    
                    # Remove previous highlight
                    self.canvas.delete("highlight")
                    
                    # Add highlight ring
                    self.canvas.create_oval(x0-3, y0-3, x1+3, y1+3, 
                                          outline=self.colors['highlight'], 
                                          width=3, tags="highlight")
                    break
    
    def on_click(self, event):
        if not self.game_running or self.game_paused:
            if not self.game_running:
                self.update_status("Please click 'Start Game' first!")
            elif self.game_paused:
                self.update_status("Game is paused. Click 'Resume' to continue.")
            return
            
        if self.game_mode == "ai_vs_ai":
            self.update_status("AI vs AI mode - no human interaction allowed")
            return  # No human interaction in AI vs AI
        
        if self.game_mode == "human_vs_ai" and self.env.current_player == -1:
            self.update_status("Wait for AI's turn...")
            return  # Wait for AI turn
        
        col = (event.x - 10) // self.cell_size
        if 0 <= col < 7:
            self.make_move(col)
        else:
            self.update_status("Click on a valid column (0-6)")
    
    def make_move(self, col):
        """Make a move in the specified column"""
        if not self.game_running or self.game_paused:
            return False
            
        # Check if move is legal
        if self.env.board[0, col] != 0:
            self.update_status("Column is full! Choose another column.")
            return False
        
        # Make the move
        obs, reward, terminated, truncated, info, _ = self.env.step(col)
        
        # Update display
        self.draw_board()
        
        # Check game end conditions
        if terminated or truncated:
            self.game_running = False
            self.start_button.config(state=tk.NORMAL)
            self.pause_button.config(state=tk.DISABLED)
            
            reason = info.get('reason', 'Unknown')
            
            if reason == 'Win':
                winner_player = info.get('winner', 1)
                winner = "Player 1 (Red)" if winner_player == 1 else "Player 2 (Yellow)"
                self.update_status(f"🎉 {winner} wins!")
                self.root.after(100, lambda: messagebox.showinfo("Game Over", f"{winner} wins!"))
            elif reason == 'Tie':
                self.update_status("🤝 It's a tie!")
                self.root.after(100, lambda: messagebox.showinfo("Game Over", "It's a tie!"))
            elif reason == 'Illegal move':
                self.update_status("❌ Illegal move! Game over.")
                self.root.after(100, lambda: messagebox.showerror("Illegal Move", "Illegal move!"))
        else:
            # Game continues - update player display and handle AI turns
            self.update_player_display()
            
            if self.game_mode == "human_vs_ai" and self.env.current_player == -1:
                threading.Thread(target=self.make_ai_move, daemon=True).start()
        
        return True
    
    def animate_piece_drop(self, col):
        # Simple animation - could be enhanced
        self.root.update()
        time.sleep(0.2)  # Brief pause for visual effect
    
    def make_ai_move(self):
        time.sleep(self.animation_speed)
        
        if not self.game_running or self.game_paused:
            return
        
        # Simple random AI - could be enhanced with actual AI logic
        action_mask = self.env._action_mask()
        legal_actions = np.where(action_mask)[0]
        
        if len(legal_actions) > 0:
            action = random.choice(legal_actions)
            self.root.after(0, lambda: self.make_move(action))
    
    def run_ai_game(self):
        while self.game_running:
            if self.game_paused:
                time.sleep(0.1)
                continue
                
            time.sleep(self.animation_speed)
            
            if not self.game_running:
                break
                
            # Make AI move
            action_mask = self.env._action_mask()
            legal_actions = np.where(action_mask)[0]
            
            if len(legal_actions) > 0:
                action = random.choice(legal_actions)
                self.root.after(0, lambda a=action: self.make_move(a))
            
            time.sleep(self.animation_speed)

    def mainloop(self):
        self.root.mainloop()