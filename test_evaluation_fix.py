#!/usr/bin/env python3
"""
Simple test of the evaluation system with the trainer.
"""

from training.trainer import Trainer
from agents.random_agent import RandomAgent
from agents.minimax_agent import MinimaxAgent

def test_evaluation_system():
    """Test that the evaluation system now works correctly."""
    
    print("🔍 Testing fixed evaluation system...")
    
    # Create trainer
    trainer = Trainer(save_dir='./debug_models')
    
    # Create agents
    random_agent = RandomAgent("Random")
    minimax_agent = MinimaxAgent(depth=2, name="Minimax")
    
    # Test evaluation
    print("\n🎯 Testing Random vs Minimax (10 games each direction)")
    
    # Random vs Minimax
    results1 = trainer.evaluate_agent_vs_baseline(random_agent, minimax_agent, num_games=10)
    print(f"Random vs Minimax: {results1['win_rate']:.2%} win rate")
    print(f"Detailed: {results1['wins']}W-{results1['losses']}L-{results1['ties']}T")
    
    # Minimax vs Random (reversed)
    results2 = trainer.evaluate_agent_vs_baseline(minimax_agent, random_agent, num_games=10)
    print(f"Minimax vs Random: {results2['win_rate']:.2%} win rate")
    print(f"Detailed: {results2['wins']}W-{results2['losses']}L-{results2['ties']}T")
    
    # Expected results:
    # - Random should have low win rate against Minimax
    # - Minimax should have high win rate against Random
    # - Neither should be exactly 50%
    
    if abs(results1['win_rate'] - 0.5) < 0.1 and abs(results2['win_rate'] - 0.5) < 0.1:
        print("⚠️  Still getting ~50% win rates - potential remaining issues")
    else:
        print("✅ Win rates are now meaningful and different from 50%!")
    
    print(f"\n📊 Summary:")
    print(f"Random win rate vs Minimax: {results1['win_rate']:.1%}")
    print(f"Minimax win rate vs Random: {results2['win_rate']:.1%}")
    
    # Should be roughly inverse (Random low, Minimax high)
    if results1['win_rate'] < 0.4 and results2['win_rate'] > 0.6:
        print("✅ EXCELLENT: Results make sense (Random < 40%, Minimax > 60%)")
    elif results1['win_rate'] > 0.6 and results2['win_rate'] < 0.4:
        print("✅ GOOD: Results inverted but consistent")
    else:
        print("🤔 Results are unexpected - needs more investigation")

if __name__ == "__main__":
    test_evaluation_system()