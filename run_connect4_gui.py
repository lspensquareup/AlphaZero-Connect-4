# run_connect4_gui.py - Enhanced GUI version
import sys
import os

# Add the current directory to Python path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from connect4_env import GymnasiumConnectFour as ConnectFourEnv
from connect4_gui import ConnectFourGUI

def main():
    """Launch the Connect Four game with enhanced GUI"""
    print("🎮 Starting Connect Four with Enhanced GUI...")
    print("Features:")
    print("  - Human vs Human gameplay")
    print("  - Human vs AI gameplay") 
    print("  - AI vs AI gameplay")
    print("  - Adjustable AI speed")
    print("  - Game controls (start, pause, reset)")
    print("  - Visual feedback and animations")
    print("\nLaunching GUI...")
    
    # Create the game environment
    env = ConnectFourEnv()
    
    # Create and run the GUI
    gui = ConnectFourGUI(env)
    gui.mainloop()
    
    print("Thanks for playing Connect Four! 🎉")

if __name__ == "__main__":
    main()