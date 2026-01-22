from typing import Any, Dict, Optional, Union

from etils import epath
import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env

_XML_PATH = mjx_env.ROOT_PATH / "robco" / "xmls"

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
  assets = {}
  mjx_env.update_assets(assets, _XML_PATH, "*.xml")
  
  # Add assets with proper subdirectory paths
  assets_path = _XML_PATH / "assets"
  for asset_file in assets_path.glob("*"):
    if asset_file.is_file():
      # Include the assets/ prefix in the key to match XML file references
      assets[f"assets/{asset_file.name}"] = asset_file.read_bytes()
  
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

    self._ee_body_id = self._mj_model.body("joint5_distal").id

  # Joint and actuator utilities

  def get_joint_positions(self, data: mjx.Data) -> jax.Array:
    """Get positions of all arm joints."""
    return data.qpos

  def get_joint_velocities(self, data: mjx.Data) -> jax.Array:
    """Get velocities of all arm joints."""
    return data.qvel

  def get_end_effector_position(self, data: mjx.Data) -> jax.Array:
    """Get end effector position using joint 5 position."""
    return data.xpos[self._ee_body_id]

  def get_reaching_point_position(self, data: mjx.Data) -> jax.Array:
    """Get target reaching point position."""
    target_body_id = self._mj_model.body(self.target_body_name).id
    return data.xpos[target_body_id]

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
