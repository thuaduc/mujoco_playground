#!/usr/bin/env python3
"""Simple test script for robco environments."""

import warnings

# Suppress overflow warning from MuJoCo MJX collision detection
# This occurs when MJX uses jp.finfo(float).max with float32 arrays
warnings.filterwarnings("ignore", message="overflow encountered in cast")

from time import time

import jax
import jax.numpy as jp

from mujoco_playground._src.robco import get_default_config
from mujoco_playground._src.robco import load


def test_robco_reach():
  """Test the RobcoReach environment."""
  print("Testing RobcoArm environment...")

  # Load environment
  config = get_default_config("RobcoArm")
  env = load("RobcoArm", config)

  print(f"Environment loaded successfully!")
  print(f"Action size: {env.action_size}")
  print(f"Observation size: {env.observation_size}")
  print(f"Number of joints: {env.num_joints}")

  # Measure jit compile time
  start_time = time()
  jit_reset = jax.jit(env.reset)
  jit_step = jax.jit(env.step)
  end_time = time()
  print(f"JIT compilation time: {end_time - start_time:.4f} seconds")

  # Test reset
  rng = jax.random.PRNGKey(42)
  state = jit_reset(rng)

  print(f"Reset successful!")
  print(f"Observation shape: {state.obs.shape}")
  print(f"Initial reward: {state.reward}")

  # Measure first step time
  start_time = time()
  action = jp.zeros(env.action_size)  # Zero action
  new_state = jit_step(state, action)
  end_time = time()
  print(f"First step time (including JIT): {end_time - start_time:.4f} seconds")

  print(f"Step successful!")
  print(f"New reward: {new_state.reward}")

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
