#!/usr/bin/env python3
"""Simple test script for robco environments."""

import jax
import jax.numpy as jp
from mujoco_playground._src.robco import load, get_default_config


def test_robco_reach():
    """Test the RobcoReach environment."""
    print("Testing RobcoReach environment...")
    
    # Load environment
    config = get_default_config("RobcoReach")
    env = load("RobcoReach", config)
    
    print(f"Environment loaded successfully!")
    print(f"Action size: {env.action_size}")
    print(f"Observation size: {env.observation_size}")
    print(f"Number of joints: {env.num_joints}")
    
    # Test reset
    rng = jax.random.PRNGKey(42)
    state = env.reset(rng)
    
    print(f"Reset successful!")
    print(f"Observation shape: {state.obs.shape}")
    print(f"Initial reward: {state.reward}")
    print(f"Target position: {state.info['target_pos']}")
    print(f"Distance to target: {state.info['distance_to_target']}")
    
    # Test step
    action = jp.zeros(env.action_size)  # Zero action
    new_state = env.step(state, action)
    
    print(f"Step successful!")
    print(f"New reward: {new_state.reward}")
    print(f"New distance: {new_state.info['distance_to_target']}")
    print(f"Success: {new_state.info['success']}")
    
    return True


def test_robco_arm():
    """Test the basic RobcoArm environment."""
    print("\nTesting RobcoArm environment...")
    
    # Load environment
    config = get_default_config("RobcoArm")
    env = load("RobcoArm", config)
    
    print(f"Environment loaded successfully!")
    print(f"Action size: {env.action_size}")
    print(f"Number of joints: {env.num_joints}")
    
    return True


if __name__ == "__main__":
    try:
        test_robco_reach()
        test_robco_arm()
        print("\n✅ All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()