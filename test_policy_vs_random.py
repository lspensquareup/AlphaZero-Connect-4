#!/usr/bin/env python3
"""
Test PolicyAgent vs RandomAgent to see if we get realistic win rates.
"""

from training.trainer import Trainer
from agents.policy_agent import PolicyAgent  
from agents.random_agent import RandomAgent
from networks.policy_network import PolicyNetwork

def test_policy_vs_random():
    """Test PolicyAgent against RandomAgent."""
    print("🔍 Testing PolicyAgent vs RandomAgent...")
    
    # Create trainer
    trainer = Trainer(save_dir='./debug_models')
    
    # Create agents
    policy_agent = PolicyAgent("PolicyAgent")
    policy_agent.set_network(PolicyNetwork())  # Untrained network
    
    random_agent = RandomAgent("Random")
    
    print("\n🎯 Testing PolicyAgent vs Random (10 games)")
    
    # Test evaluation
    results = trainer.evaluate_agent_vs_baseline(policy_agent, random_agent, num_games=10)
    print(f"PolicyAgent vs Random: {results['win_rate']:.2%} win rate")
    print(f"Detailed: {results['wins']}W-{results['losses']}L-{results['ties']}T")
    
    if results['win_rate'] > 0:
        print("✅ FIXED! PolicyAgent now has non-zero win rate")
    else:
        print("❌ Still broken - PolicyAgent has 0% win rate")
        
    # Test reverse
    print(f"\n🎯 Testing Random vs PolicyAgent (10 games)")
    results2 = trainer.evaluate_agent_vs_baseline(random_agent, policy_agent, num_games=10)
    print(f"Random vs PolicyAgent: {results2['win_rate']:.2%} win rate")
    print(f"Detailed: {results2['wins']}W-{results2['losses']}L-{results2['ties']}T")
    
    print(f"\n📊 Summary:")
    print(f"PolicyAgent win rate: {results['win_rate']:.1%}")
    print(f"Random win rate: {results2['win_rate']:.1%}")
    
    if results['win_rate'] > 0 and results2['win_rate'] > 0:
        print("✅ Both agents can win - evaluation system working!")
    else:
        print("❌ One agent never wins - still have issues")

if __name__ == "__main__":
    test_policy_vs_random()