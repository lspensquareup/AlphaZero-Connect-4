# AlphaZero Connect-4 Visualization & Analytics

This project now includes comprehensive visualization and analytics tools for training and evaluating Connect-4 agents.

## Features

### 🎮 Live Game Visualization
- **Real-time game display** with enhanced GUI showing agent information
- **Agent strategy details** displayed during gameplay
- **Move analysis** with thinking times and decision tracking
- **Game statistics** updated in real-time during play

### 📊 Training Analytics Dashboard
- **Interactive training control** with customizable parameters
- **Real-time loss charts** for policy and value networks
- **Training progress tracking** with progress bars and logs
- **Performance metrics** including win rates and game statistics

### 📈 Comprehensive Metrics
- **Win rate analysis** for different agents
- **Game length distributions** and statistical analysis
- **Training data quality metrics**
- **Agent comparison tournaments**

## Quick Start

### 1. Launch Training Dashboard
```bash
conda activate chess-gym
python training_dashboard.py
```

This opens a comprehensive dashboard with tabs for:
- **Training Control**: Start/stop training with customizable parameters
- **Live Games**: Watch training games in real-time
- **Analytics**: View training metrics and loss charts
- **Agent Comparison**: Run tournaments between different agents

### 2. Visualize Individual Games
```bash
python training_game_visualizer.py
```

Watch a single game with detailed agent information and move analysis.

### 3. Run Analytics Demo
```bash
python test_visualization.py
```

Generate sample data and see detailed analytics output.

## Visualization Components

### Training Dashboard (`training_dashboard.py`)
The main analytics interface providing:

#### Training Control Panel
- **Game Generation**: Configure number of games and minimax depth
- **Network Training**: Set epochs, batch size, and learning rate
- **Visualization Options**: Enable/disable live games and real-time charts
- **Progress Tracking**: Real-time progress bars and detailed logging

#### Live Game Display
- Integration with existing ConnectFourGUI
- Agent information panels showing strategy details
- Real-time game statistics and move history

#### Analytics Charts
- **Policy Network Loss**: Training loss over epochs
- **Value Network Loss**: Evaluation accuracy improvement
- **Training Examples**: Data volume per session
- **Win Rate Analysis**: Agent performance comparison
- **Game Length Distribution**: Statistical analysis of game complexity
- **Combined Loss Comparison**: Side-by-side network performance

#### Agent Comparison
- Tournament system for evaluating different agents
- Head-to-head matchup statistics
- Performance ranking and win rate analysis

### Game Visualizer (`training_game_visualizer.py`)
Enhanced game display featuring:

#### Agent Information Display
- **Strategy Details**: Agent type, depth, evaluation method
- **Performance Stats**: Thinking time, move quality
- **Real-time Status**: Current player, game progress

#### Game Analysis Panel
- **Move History**: Complete game record with timestamps
- **Statistics Tracking**: Game duration, move count, thinking times
- **Performance Metrics**: Agent comparison during gameplay

#### Post-Game Analysis
- **Final Results**: Winner, game length, total time
- **Thinking Time Analysis**: Fastest/slowest moves, patterns
- **Strategic Insights**: Move quality and decision patterns

### Session Management
The `GameSessionManager` class provides:
- **Multi-game tracking**: Statistics across multiple games
- **Agent performance monitoring**: Win rates and comparative analysis
- **Data collection**: Automated metrics gathering for analysis

## Sample Analytics Output

```
Game Outcomes (10 games):
  Minimax wins: 5 (50.0%)
  Random wins: 5 (50.0%)
  Ties: 0 (0.0%)

Game Length Analysis:
  Shortest game: 9 moves
  Longest game: 23 moves
  Average game: 15.0 moves

Training Data Generated:
  Total position examples: 150
  Examples per game: 15.0
```

## Integration with Existing Training

The visualization system integrates seamlessly with the existing training pipeline:

### Training Data Generation
```python
from training_dashboard import TrainingVisualizationDashboard

# Launch dashboard
dashboard = TrainingVisualizationDashboard()
dashboard.run()
```

### Individual Game Analysis
```python
from training_game_visualizer import GameSessionManager
from agents.minimax_agent import MinimaxAgent
from agents.random_agent import RandomAgent

# Create agents
minimax = MinimaxAgent(depth=4)
random = RandomAgent()

# Play visualized game
session = GameSessionManager()
result = session.play_visualized_game(minimax, random)
```

### Batch Analytics
```python
from training.trainer import generate_minimax_training_data

# Generate training data
games = generate_minimax_training_data(num_games=50, minimax_depth=4)

# Analyze results
minimax_wins = sum(1 for g in games if g['minimax_wins'])
print(f"Minimax win rate: {100*minimax_wins/len(games):.1f}%")
```

## Customization Options

### Dashboard Configuration
- **Training Parameters**: Adjustable game count, depth, epochs, batch size
- **Visualization Speed**: Control game playback speed
- **Chart Updates**: Real-time or manual chart refreshing
- **Data Export**: Save training metrics to JSON files

### Game Display Options
- **Agent Information**: Toggle strategy details display
- **Move History**: Configurable history length
- **Statistics Panel**: Customizable metrics display
- **Analysis Depth**: Detailed or summary view

### Analytics Configuration
- **Chart Types**: Multiple visualization options
- **Metrics Selection**: Choose which statistics to track
- **Export Formats**: JSON, CSV data export
- **Session Persistence**: Save/load analytics sessions

## Performance Benefits

### Training Insights
- **Visual feedback** helps identify training issues early
- **Real-time metrics** allow for parameter adjustment during training
- **Game visualization** reveals agent behavior patterns
- **Statistical analysis** guides hyperparameter tuning

### Development Benefits
- **Debugging aid** for agent implementation
- **Performance comparison** between different approaches
- **Quality assurance** for training data generation
- **Progress monitoring** for long training sessions

## Technical Requirements

### Dependencies
- All existing project dependencies (PyTorch, Gymnasium, NumPy)
- Matplotlib for charting (already installed in chess-gym environment)
- Tkinter for GUI (included with Python)

### Performance Considerations
- **GUI responsiveness**: Game visualization runs in separate threads
- **Memory management**: Large training sessions handled efficiently
- **Real-time updates**: Non-blocking chart updates
- **Data persistence**: Automatic saving of session data

## Usage Examples

### Complete Training Session with Visualization
1. Launch `python training_dashboard.py`
2. Set parameters (50 games, depth 4, 20 epochs)
3. Enable "Show Training Games" and "Real-time Charts"
4. Click "Start Training"
5. Watch games play out in real-time
6. Monitor loss charts as networks train
7. Export results for further analysis

### Agent Development Workflow
1. Implement new agent class
2. Use `training_game_visualizer.py` to test against existing agents
3. Run tournament comparison in dashboard
4. Analyze performance metrics
5. Iterate on agent design based on insights

### Research and Analysis
1. Generate large datasets with `generate_minimax_training_data()`
2. Use dashboard analytics to examine patterns
3. Export data for external analysis tools
4. Compare different training approaches
5. Document insights for future improvements

The visualization system transforms the AlphaZero Connect-4 project from a training pipeline into a comprehensive analysis and development environment!