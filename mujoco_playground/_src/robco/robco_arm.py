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
"""Basic Robco arm environment implementation."""

from typing import Any, Dict, Optional, Union

import jax
from ml_collections import config_dict
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground._src.robco.base import RobcoArmBase


def default_config() -> config_dict.ConfigDict:
  """Default configuration for basic RobcoArm environments."""
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.002,
      episode_length=1000,
      action_repeat=1,
      vision=False,
      impl="jax",
      action_scale=4.7124,  # Scale actions from [-1, 1] to joint range [-4.7124, 4.7124]
      nconmax=10,  # maximum number of contacts
      njmax=2,  # maximum number of constraints
      success_distance_threshold=0.05,  # Distance threshold for task success (meters)
  )


class RobcoArm(RobcoArmBase):
  """Basic Robco arm environment."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ) -> None:
    xml_path = mjx_env.ROOT_PATH / "robco" / "xmls" / "robco_arm.xml"
    super().__init__(xml_path, config, config_overrides)
    # Assume the object is named 'target' in the XML
    self.target_body_name = "target"

  def reset(self, rng: jax.Array) -> mjx_env.State:
    """Reset the environment."""
    # Simple reset - just initialize with zero joint positions
    qpos = jax.numpy.zeros(self._mjx_model.nq)
    qvel = jax.numpy.zeros(self._mjx_model.nv)

    data = mjx.make_data(
        self._mjx_model,
        impl=self.mjx_model.impl.value,
        nconmax=self._config.nconmax,
        njmax=self._config.njmax,
    )
    data = data.replace(qpos=qpos, qvel=qvel)

    # Forward pass to compute derived quantities (xpos, xquat, etc.)
    data = mjx.forward(self._mjx_model, data)

    # Create observation
    obs = self._get_obs(data)

    metrics = {}
    info = {"rng": rng}

    reward, done = jax.numpy.zeros(2)

    return mjx_env.State(
        data=data,
        obs=obs,
        reward=reward,
        done=done,
        metrics=metrics,
        info=info,
    )

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    """Step the environment."""
    # Apply action scaling
    action = action * self._config.action_scale

    data = mjx_env.step(
        self._mjx_model, state.data, action, n_substeps=self.n_substeps
    )
    obs = self._get_obs(data)

    # Get end effector  and target position
    ee_pos = self.get_end_effector_position(data)
    target_pos = data.xpos[self._mj_model.body(self.target_body_name).id]

    # Compute distance between end effector and target
    distance = jax.numpy.linalg.norm(ee_pos - target_pos)

    # Reward: negative L2 distance between end effector and target
    reward = -distance

    # Episode terminates when distance is below threshold (task success)
    done = distance < self._config.success_distance_threshold
    done = done.astype(float)

    return mjx_env.State(
        data=data,
        obs=obs,
        reward=reward,
        done=done,
        metrics=state.metrics,
        info=state.info,
    )

  def _get_obs(self, data: mjx.Data) -> jax.Array:
    """Get observation from mjx data."""
    # Joint positions and velocities
    joint_pos = self.get_joint_positions(data)
    joint_vel = self.get_joint_velocities(data)

    # End effector position
    ee_pos = self.get_end_effector_position(data)

    return jax.numpy.concatenate([joint_pos, joint_vel, ee_pos])
