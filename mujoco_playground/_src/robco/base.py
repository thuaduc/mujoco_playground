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
"""Base class for Robco arm environments."""

from typing import Any, Dict, Optional, Union

from etils import epath
import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env


# Joint names for the Robco arm
JOINT_NAMES = [
    "joint0_joint",
    "joint1_joint", 
    "joint2_joint",
    "joint3_joint",
    "joint4_joint",
    "joint5_joint",
]

# Actuator names for the Robco arm
ACTUATOR_NAMES = [
    "joint0_joint_ctrl",
    "joint1_joint_ctrl",
    "joint2_joint_ctrl", 
    "joint3_joint_ctrl",
    "joint4_joint_ctrl",
    "joint5_joint_ctrl",
]


def get_assets() -> Dict[str, bytes]:
  """Get assets for Robco arm environments."""
  assets = {}
  robco_path = mjx_env.ROOT_PATH / "robco" / "xmls"
  mjx_env.update_assets(assets, robco_path, "*.xml")
  mjx_env.update_assets(assets, robco_path / "urdf", "*.stl")
  mjx_env.update_assets(assets, robco_path / "urdf", "*.obj")
  return assets


class RobcoArmBase(mjx_env.MjxEnv):
  """Base class for Robco Arm environments."""

  def __init__(
      self,
      xml_path: str,
      config: config_dict.ConfigDict,
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ) -> None:
    super().__init__(config, config_overrides)
    self._model_assets = get_assets()
    self._mj_model = mujoco.MjModel.from_xml_string(
        epath.Path(xml_path).read_text(), assets=self._model_assets
    )
    self._mj_model.opt.timestep = self._config.sim_dt

    # Set visualization properties
    self._mj_model.vis.global_.offwidth = 3840
    self._mj_model.vis.global_.offheight = 2160

    self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
    self._xml_path = xml_path
    # Precompute joint qpos/qvel indices (each hinge joint has width 1).
    self._joint_qpos_indices = [
        self._mj_model.jnt_qposadr[self._mj_model.joint(j_name).id]
        for j_name in JOINT_NAMES
    ]
    self._joint_qvel_indices = [
        self._mj_model.jnt_dofadr[self._mj_model.joint(j_name).id]
        for j_name in JOINT_NAMES
    ]

    # Precompute end-effector body id with fallback logic once (avoid Python in jitted step).
    self._ee_body_id = self._compute_ee_body_id()

  # Joint and actuator utilities

  def get_joint_positions(self, data: mjx.Data) -> jax.Array:
    """Get positions of all arm joints."""
    return data.qpos[jp.array(self._joint_qpos_indices)]

  def get_joint_velocities(self, data: mjx.Data) -> jax.Array:
    """Get velocities of all arm joints."""
    return data.qvel[jp.array(self._joint_qvel_indices)]

  def get_end_effector_position(self, data: mjx.Data) -> jax.Array:
    """Get end effector position using joint 5 position."""
    return data.xpos[self._ee_body_id]

  def get_end_effector_orientation(self, data: mjx.Data) -> jax.Array:
    """Get end effector orientation using joint 5 orientation."""
    return data.xquat[self._ee_body_id]

  def get_end_effector_velocity(self, data: mjx.Data) -> jax.Array:
    """Get end effector linear velocity using joint 5."""
    return mjx_env.get_body_linvel(self._mjx_model, data, self._ee_body_id)

  def _compute_ee_body_id(self) -> int:
    """Resolve a stable end-effector body id with fallbacks."""
    # Try common body naming conventions.
    for name in ["joint5", "link5", "joint5_distal"]:
      try:
        return self._mj_model.body(name).id
      except Exception:  # pylint: disable=broad-except
        pass
    # Fallback: use body owning the last joint.
    try:
      jnt_id = self._mj_model.joint("joint5_joint").id
      body_id = self._mj_model.jnt_bodyid[jnt_id]
      return int(body_id)
    except Exception:  # pylint: disable=broad-except
      # Final fallback to world (0) to avoid crashes; will give meaningless EE data.
      return 0

  # Accessors

  @property
  def xml_path(self) -> str:
    return self._xml_path

  @property
  def action_size(self) -> int:
    return self._mjx_model.nu

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self._mjx_model

  @property
  def num_joints(self) -> int:
    return len(JOINT_NAMES)


def uniform_quat(rng: jax.Array) -> jax.Array:
  """Generate a random quaternion from a uniform distribution."""
  u, v, w = jax.random.uniform(rng, (3,))
  return jp.array([
      jp.sqrt(1 - u) * jp.sin(2 * jp.pi * v),
      jp.sqrt(1 - u) * jp.cos(2 * jp.pi * v),
      jp.sqrt(u) * jp.sin(2 * jp.pi * w),
      jp.sqrt(u) * jp.cos(2 * jp.pi * w),
  ])


def normalize_angle(angle: jax.Array) -> jax.Array:
  """Normalize angle to [-pi, pi]."""
  return jp.mod(angle + jp.pi, 2 * jp.pi) - jp.pi
