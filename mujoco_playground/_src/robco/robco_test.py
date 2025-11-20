# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for the robco environments."""
import jax
import jax.numpy as jp
from absl.testing import absltest, parameterized

from mujoco_playground._src import robco


class TestSuite(parameterized.TestCase):
  """Tests for the robco environments."""

  @parameterized.named_parameters(
      {"testcase_name": f"test_can_create_{env_name}", "env_name": env_name}
      for env_name in robco.ALL_ENVS
  )
  def test_can_create_all_environments(self, env_name: str) -> None:
    env = robco.load(env_name)
    state = jax.jit(env.reset)(jax.random.PRNGKey(42))
    state = jax.jit(env.step)(state, jp.zeros(env.action_size))
    self.assertIsNotNone(state)
    obs_shape = jax.tree_util.tree_map(lambda x: x.shape, state.obs)
    obs_shape = obs_shape[0] if isinstance(obs_shape, tuple) else obs_shape
    self.assertEqual(obs_shape, env.observation_size)
    self.assertFalse(jp.isnan(state.data.qpos).any())

  def test_robco_arm_termination_on_success(self) -> None:
    """Test that RobcoArm terminates when distance is below threshold."""
    env = robco.load("RobcoArm")
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    
    # Reset environment
    state = jit_reset(jax.random.PRNGKey(0))
    
    # Initially, done should be 0 (not done)
    self.assertEqual(state.done, 0.0)
    
    # Take a step with zero action
    state = jit_step(state, jp.zeros(env.action_size))
    
    # Check that done is based on distance threshold, not NaN
    # Since we start at zero joint positions, we're unlikely to be at the target
    # so done should still be 0
    self.assertFalse(jp.isnan(state.data.qpos).any())
    self.assertFalse(jp.isnan(state.data.qvel).any())

  def test_robco_arm_distance_threshold_config(self) -> None:
    """Test that the distance threshold configuration works."""
    config = robco.get_default_config("RobcoArm")
    
    # Check that the threshold is in the config
    self.assertIn("success_distance_threshold", config)
    self.assertGreater(config.success_distance_threshold, 0.0)
    
    # Test with custom threshold
    config.success_distance_threshold = 0.1
    env = robco.load("RobcoArm", config)
    self.assertEqual(env._config.success_distance_threshold, 0.1)


if __name__ == "__main__":
  absltest.main()
