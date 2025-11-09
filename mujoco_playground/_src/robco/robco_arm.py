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
        action_scale=1.0,
        nconmax=10,   # Very small for fast compilation
        njmax=2,      # Minimal constraints
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

        # Create observation
        obs = self._get_obs(data)

        return mjx_env.State(
            data=data,
            obs=obs,
            reward=jax.numpy.array(0.0),
            done=jax.numpy.array(False),
            metrics={},
            info={},
        )

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        """Step the environment."""
        # Apply action scaling
        action = action * self._config.action_scale

        data = mjx_env.step(self._mjx_model, state.data, action, n_substeps=1)
        obs = self._get_obs(data)

        # Simple reward - just negative sum of squared joint velocities (encourages stillness)
        reward = -jax.numpy.sum(jax.numpy.square(data.qvel)) * 0.01

        return state.replace(
            data=data, obs=obs, reward=reward, done=jax.numpy.array(False)
        )

    def _get_obs(self, data: mjx.Data) -> jax.Array:
        """Get observation from mjx data."""
        # Joint positions and velocities
        joint_pos = self.get_joint_positions(data)
        joint_vel = self.get_joint_velocities(data)

        # End effector position
        ee_pos = self.get_end_effector_position(data)

        return jax.numpy.concatenate([joint_pos, joint_vel, ee_pos])
