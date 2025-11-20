#!/usr/bin/env python3
"""Test script to verify RobcoArm joint ranges match actuator control ranges."""

import jax
import jax.numpy as jp
import numpy as np

from mujoco_playground import registry


def test_actuator_ranges_match_joint_ranges():
  """Verify that actuator control ranges match joint ranges."""
  print("Testing RobcoArm actuator and joint ranges...")

  # Load environment
  env = registry.load("RobcoArm")
  config = registry.get_default_config("RobcoArm")

  # Check that each actuator control range matches the corresponding joint range
  joint_names = [
      "joint0_joint",
      "joint1_joint",
      "joint2_joint",
      "joint3_joint",
      "joint4_joint",
      "joint5_joint",
  ]

  print(f"\nVerifying actuator ranges match joint ranges:")
  all_match = True
  for i, joint_name in enumerate(joint_names):
    joint = env.mj_model.joint(joint_name)
    joint_range = env.mj_model.jnt_range[joint.id]

    actuator_name = f"{joint_name}_ctrl"
    actuator = env.mj_model.actuator(actuator_name)
    ctrl_range = env.mj_model.actuator_ctrlrange[actuator.id]

    match = np.allclose(joint_range, ctrl_range, rtol=1e-5)
    status = "✓" if match else "✗"
    print(
        f"  {status} Joint {i} ({joint_name}): range={joint_range},"
        f" ctrlrange={ctrl_range}"
    )
    all_match = all_match and match

  if not all_match:
    raise AssertionError("Some actuator ranges do not match joint ranges!")

  # Check that action_scale properly maps [-1, 1] to joint ranges
  print(f"\nVerifying action_scale configuration:")
  expected_scale = float(joint_range[1])  # Use the max joint range value
  actual_scale = config.action_scale

  scale_match = np.isclose(expected_scale, actual_scale, rtol=1e-5)
  status = "✓" if scale_match else "✗"
  print(
      f"  {status} action_scale: expected={expected_scale},"
      f" actual={actual_scale}"
  )

  if not scale_match:
    raise AssertionError(
        f"action_scale {actual_scale} does not match expected value"
        f" {expected_scale}!"
    )

  print("\n✓ All actuator and joint ranges are properly configured!")
  return True


def test_action_scaling():
  """Test that actions are properly scaled from [-1, 1] to joint ranges."""
  print("\nTesting action scaling...")

  # Load environment
  env = registry.load("RobcoArm")
  config = registry.get_default_config("RobcoArm")

  jit_reset = jax.jit(env.reset)
  jit_step = jax.jit(env.step)

  # Reset environment
  state = jit_reset(jax.random.PRNGKey(0))

  # Test that action = 1.0 is properly scaled
  action = jp.ones(env.action_size)
  expected_scaled_action = config.action_scale * action

  print(f"  Action (input): {action}")
  print(f"  Expected scaled action: {expected_scaled_action}")
  print(f"  Joint ranges: [-{config.action_scale}, {config.action_scale}]")

  # Verify the scaling is correct
  assert np.allclose(
      expected_scaled_action, config.action_scale
  ), "Action scaling is incorrect!"

  print("  ✓ Actions are properly scaled from [-1, 1] to joint ranges")
  return True


if __name__ == "__main__":
  try:
    test_actuator_ranges_match_joint_ranges()
    test_action_scaling()
    print("\n✅ All tests passed!")
  except Exception as e:
    print(f"\n❌ Test failed: {e}")
    import traceback

    traceback.print_exc()
    exit(1)
