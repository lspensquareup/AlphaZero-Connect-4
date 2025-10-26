#!/usr/bin/env python3
"""
Test multi-depth minimax evaluation to verify the fix works.
"""

from training.trainer import Trainer
from agents.policy_agent import PolicyAgent
from agents.minimax_agent import MinimaxAgent
from networks.policy_network import PolicyNetwork

def test_multi_depth_evaluation():
    """Test that we can evaluate against multiple minimax depths."""
    print("🔍 Testing multi-depth minimax evaluation...")
    
    # Create trainer and agent
    trainer = Trainer(save_dir='./debug_models')
    
    # Create a policy agent with a simple network
    policy_agent = PolicyAgent("TestPolicy")
    policy_agent.set_network(PolicyNetwork())
    
    # Test against different minimax depths
    depths_to_test = [2, 3, 4]
    results = {}
    
    for depth in depths_to_test:
        print(f"\n🎯 Testing vs Minimax depth {depth}...")
        try:
            minimax_agent = MinimaxAgent(depth=depth, name=f"Minimax-{depth}")
            result = trainer.evaluate_agent_vs_baseline(policy_agent, minimax_agent, num_games=5)
            results[depth] = result['win_rate']
            print(f"  ✅ PolicyAgent vs Minimax-{depth}: {result['win_rate']:.1%}")
            print(f"     Details: {result['wins']}W-{result['losses']}L-{result['ties']}T")
        except Exception as e:
            print(f"  ❌ Error with depth {depth}: {e}")
            results[depth] = None
    
    print(f"\n📊 Results Summary:")
    all_same = True
    last_rate = None
    
    for depth in depths_to_test:
        rate = results[depth]
        if rate is not None:
            print(f"  Minimax depth {depth}: {rate:.1%}")
            if last_rate is not None and abs(rate - last_rate) > 0.01:  # Allow 1% tolerance
                all_same = False
            last_rate = rate
        else:
            print(f"  Minimax depth {depth}: ERROR")
    
    if all_same and last_rate is not None:
        print(f"\n⚠️  All depths show same win rate ({last_rate:.1%}) - may indicate an issue")
    else:
        print(f"\n✅ Different win rates across depths - evaluation working correctly!")
    
    return results

if __name__ == "__main__":
    test_multi_depth_evaluation()