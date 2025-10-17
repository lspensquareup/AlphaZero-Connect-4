#!/usr/bin/env python3
"""
Tournament runner for Connect-4 AI agents.

This script sets up and runs tournaments between different AI agents,
including neural network agents and rule-based agents.
"""

import sys
import os
import argparse
from typing import List

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import agents
from agents.random_agent import RandomAgent
from agents.policy_agent import PolicyAgent
from agents.value_agent import ValueAgent
from connect4_ai import MinimaxAgent, GreedyAgent, CenterAgent

# Import training utilities
from training.tournament import Tournament
from networks.policy_network import PolicyNetwork
from networks.value_network import ValueNetwork


def create_baseline_agents() -> List:
    """Create a list of baseline (non-neural) agents."""
    agents = [
        RandomAgent("Random Baseline"),
        CenterAgent("Center Preference"),
        GreedyAgent("Greedy Look-ahead"),
        MinimaxAgent("Minimax (depth=3)", depth=3),
        MinimaxAgent("Minimax (depth=5)", depth=5),
    ]
    return agents


def create_neural_agents() -> List:
    """Create neural network agents (if models exist)."""
    agents = []
    models_dir = "models"
    
    # Look for saved policy networks
    if os.path.exists(models_dir):
        policy_files = [f for f in os.listdir(models_dir) if f.startswith("policy_network") and f.endswith(".pth")]
        value_files = [f for f in os.listdir(models_dir) if f.startswith("value_network") and f.endswith(".pth")]
        
        # Create policy agents
        for i, policy_file in enumerate(policy_files[:3]):  # Limit to 3 to avoid too many
            agent = PolicyAgent(f"Policy Net {i+1}")
            try:
                agent.load_network(os.path.join(models_dir, policy_file))
                agents.append(agent)
                print(f"Loaded policy agent: {policy_file}")
            except Exception as e:
                print(f"Failed to load {policy_file}: {e}")
        
        # Create value agents
        for i, value_file in enumerate(value_files[:2]):  # Limit to 2
            agent = ValueAgent(f"Value Net {i+1}")
            try:
                agent.load_network(os.path.join(models_dir, value_file))
                agents.append(agent)
                print(f"Loaded value agent: {value_file}")
            except Exception as e:
                print(f"Failed to load {value_file}: {e}")
    
    # If no trained models, create untrained networks for testing
    if not agents:
        print("No trained models found. Creating untrained neural agents for testing...")
        
        # Untrained policy agent
        policy_agent = PolicyAgent("Untrained Policy")
        policy_agent.set_network(PolicyNetwork())
        agents.append(policy_agent)
        
        # Untrained value agent
        value_agent = ValueAgent("Untrained Value")
        value_agent.set_network(ValueNetwork())
        agents.append(value_agent)
    
    return agents


def run_baseline_tournament(games_per_match: int = 20):
    """Run tournament with only baseline agents."""
    print("🚀 Running Baseline Agent Tournament")
    print("=" * 50)
    
    agents = create_baseline_agents()
    tournament = Tournament(verbose=True)
    
    results = tournament.run_round_robin(agents, games_per_match)
    
    # Save results
    tournament.save_results("data/baseline_tournament_results.json")
    
    return results


def run_neural_vs_baseline(games_per_match: int = 20):
    """Run tournament with neural networks vs baseline agents."""
    print("🧠 Running Neural vs Baseline Tournament")
    print("=" * 50)
    
    baseline_agents = create_baseline_agents()
    neural_agents = create_neural_agents()
    
    if not neural_agents:
        print("No neural agents available. Run training first or check models directory.")
        return None
    
    all_agents = baseline_agents + neural_agents
    tournament = Tournament(verbose=True)
    
    results = tournament.run_round_robin(all_agents, games_per_match)
    
    # Save results
    tournament.save_results("data/neural_vs_baseline_results.json")
    
    return results


def run_head_to_head(agent1_name: str, agent2_name: str, num_games: int = 50):
    """Run head-to-head match between two specific agents."""
    print(f"⚔️ Head-to-Head: {agent1_name} vs {agent2_name}")
    print("=" * 50)
    
    # Create all available agents
    all_agents = create_baseline_agents() + create_neural_agents()
    
    # Find requested agents
    agent1 = None
    agent2 = None
    
    for agent in all_agents:
        if agent1_name.lower() in agent.name.lower():
            agent1 = agent
        if agent2_name.lower() in agent.name.lower():
            agent2 = agent
    
    if agent1 is None:
        print(f"Agent not found: {agent1_name}")
        print(f"Available agents: {[agent.name for agent in all_agents]}")
        return None
    
    if agent2 is None:
        print(f"Agent not found: {agent2_name}")
        print(f"Available agents: {[agent.name for agent in all_agents]}")
        return None
    
    tournament = Tournament(verbose=True)
    results = tournament.play_match(agent1, agent2, num_games)
    
    return results


def run_quick_test():
    """Run a quick test with a few agents."""
    print("🔬 Quick Test Tournament")
    print("=" * 30)
    
    agents = [
        RandomAgent("Random"),
        GreedyAgent("Greedy"),
        MinimaxAgent("Minimax", depth=3)
    ]
    
    tournament = Tournament(verbose=True)
    results = tournament.run_round_robin(agents, games_per_match=6)
    
    return results


def create_sample_networks():
    """Create and save sample neural networks for testing."""
    print("🏗️ Creating sample neural networks...")
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    try:
        import torch
        
        # Create sample policy network
        policy_net = PolicyNetwork()
        torch.save(policy_net.state_dict(), "models/sample_policy_network.pth")
        print("✅ Sample policy network saved")
        
        # Create sample value network
        value_net = ValueNetwork()
        torch.save(value_net.state_dict(), "models/sample_value_network.pth")
        print("✅ Sample value network saved")
        
        print("Sample networks created successfully!")
        
    except ImportError:
        print("⚠️ PyTorch not available. Cannot create sample networks.")
        print("Install PyTorch to use neural network agents:")
        print("pip install torch torchvision")


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(description="Connect-4 AI Tournament Runner")
    parser.add_argument("--mode", choices=["baseline", "neural", "head2head", "quick", "create-networks"], 
                       default="quick", help="Tournament mode")
    parser.add_argument("--games", type=int, default=20, 
                       help="Number of games per match")
    parser.add_argument("--agent1", type=str, help="First agent name for head-to-head")
    parser.add_argument("--agent2", type=str, help="Second agent name for head-to-head")
    
    args = parser.parse_args()
    
    if args.mode == "baseline":
        run_baseline_tournament(args.games)
    elif args.mode == "neural":
        run_neural_vs_baseline(args.games)
    elif args.mode == "head2head":
        if args.agent1 and args.agent2:
            run_head_to_head(args.agent1, args.agent2, args.games)
        else:
            print("Head-to-head mode requires --agent1 and --agent2 arguments")
    elif args.mode == "quick":
        run_quick_test()
    elif args.mode == "create-networks":
        create_sample_networks()
    
    print("\n🎯 Tournament completed!")


if __name__ == "__main__":
    main()