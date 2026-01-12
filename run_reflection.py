#!/usr/bin/env python3
"""
Quick Start Script for Universal Self-Reflection System

Run this script to perform a daily self-reflection for your agent.
"""

import sys
from pathlib import Path

# Import reflection integration
from reflection_integration import AIProphetAgent, run_daily_reflection

def main():
    print("🔮 Universal Self-Reflection System")
    print("=" * 60)
    
        # Create agent instance
    agent = AIProphetAgent()
    
    # Run daily reflection
    result = run_daily_reflection(agent)
    
    print("\n✅ Reflection complete!")
    print(f"📊 Results saved to: data/reflection/")
    
    return result

if __name__ == '__main__':
    main()
