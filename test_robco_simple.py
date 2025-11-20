import os
import sys
import warnings

sys.path.append("/home/duckoid/Downloads/mujoco_playground")

# Suppress overflow warning from MuJoCo MJX collision detection
# This occurs when MJX uses jp.finfo(float).max with float32 arrays
warnings.filterwarnings("ignore", message="overflow encountered in cast")

import jax
import jax.numpy as jp

from mujoco_playground import registry


def test_robco_arm():
  print("Testing RobcoArm environment...")

  try:
    # Load environment
    env = registry.load("RobcoArm")
    env_cfg = registry.get_default_config("RobcoArm")

    print(f"✓ Environment loaded successfully")
    print(f"  Action size: {env.action_size}")
    print(f"  Number of joints: {env.num_joints}")

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    # Reset environment
    state = jit_reset(jax.random.PRNGKey(0))
    rollout = [state]

    f = 0.5
    for i in range(100):
      action = []
      for j in range(env.action_size):
        action.append(
            jp.sin(
                state.data.time * 2 * jp.pi * f
                + j * 2 * jp.pi / env.action_size
            )
        )
      action = jp.array(action)
      state = jit_step(state, action)
      print(f"Step {i}: reward={state.reward}, done={state.done}")

  except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()
    return False


if __name__ == "__main__":
  success = test_robco_arm()
  sys.exit(0 if success else 1)
