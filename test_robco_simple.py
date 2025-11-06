#!/usr/bin/env python3

import sys
import os
sys.path.append('/home/duckoid/Downloads/mujoco_playground')

# Set EGL for headless rendering
os.environ['MUJOCO_GL'] = 'egl'

import jax
import jax.numpy as jp
from mujoco_playground._src import robco

def test_robco_arm():
    print("Testing RobcoArm environment...")
    
    try:
        # Load environment
        env = robco.load("RobcoArm")
        print(f"✓ Environment loaded successfully")
        print(f"  Action size: {env.action_size}")
        print(f"  Number of joints: {env.num_joints}")
        
        # Reset environment
        rng = jax.random.PRNGKey(0)
        state = env.reset(rng)
        print(f"✓ Environment reset successfully")
        print(f"  Observation shape: {state.obs.shape}")
        print(f"  Initial reward: {state.reward}")
        
        # Test joint positions
        joint_pos = env.get_joint_positions(state.data)
        print(f"✓ Joint positions: {joint_pos}")
        
        # Test end effector position
        ee_pos = env.get_end_effector_position(state.data)
        print(f"✓ End effector position: {ee_pos}")
        
        # Take a few steps
        for step in range(3):
            rng, action_rng = jax.random.split(rng)
            action = jax.random.uniform(action_rng, (env.action_size,), minval=-0.1, maxval=0.1)
            state = env.step(state, action)
            print(f"✓ Step {step + 1}: reward = {state.reward:.4f}")
        
        print("\n🎉 All tests passed! RobcoArm is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_robco_arm()
    sys.exit(0 if success else 1)