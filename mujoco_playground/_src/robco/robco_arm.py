from typing import Any, Dict, Optional, Union, Sequence

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground._src.robco.base import RobcoArmBase


def default_config() -> config_dict.ConfigDict:
  """Default configuration for basic RobcoArm environments."""
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.002,
      episode_length=150,
      action_repeat=1,
      vision=False,
      impl="jax",
      action_scale=4.7124,
      nconmax=32,
      njmax=8,
      success_distance_threshold=0.05,
      reward_config=config_dict.create(
          scales=config_dict.create(
              end_effector_target=1,
              ground_collision=2,
              self_collision=2,
              energy=0,
          ),
      ),
  )


class RobcoArm(RobcoArmBase):
  """Basic Robco arm environment (mjx-safe, jittable)."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, Sequence[Any]]]] = None,
  ) -> None:
    xml_path = mjx_env.ROOT_PATH / "robco" / "xmls" / "robco_arm.xml"
    super().__init__(xml_path, config, config_overrides)

    # name of target body in XML (must exist)
    self.target_body_name = "target"
    self._target_body_id = self._mj_model.body(self.target_body_name).id

    # store sensor *ids* (these come from model.sensor(name).id)
    self._self_collision_sensor_adrs = jp.array(
        [
            self._mj_model.sensor("joint0_joint4_found").adr,
            self._mj_model.sensor("joint1_joint4_found").adr,
        ],
        dtype=jp.int32,
    )

    self._ground_collision_sensor_adrs = jp.array(
        [
            self._mj_model.sensor("joint3_floor_found").adr,
            self._mj_model.sensor("joint4_floor_found").adr,
        ],
        dtype=jp.int32,
    )

  def reset(self, rng: jax.Array) -> mjx_env.State:
    """Reset the environment and return initial State (mjx Data + obs, metrics...)."""
    qpos = jp.zeros(self._mjx_model.nq)
    qvel = jp.zeros(self._mjx_model.nv)

    data = mjx.make_data(
        self._mjx_model,
        impl=self.mjx_model.impl.value,
        nconmax=self._config.nconmax,
        njmax=self._config.njmax,
    )
    data = data.replace(qpos=qpos, qvel=qvel)

    # forward to compute derived quantities
    data = mjx.forward(self._mjx_model, data)

    obs = self._get_obs(data)

    # metrics initialized as plain floats (or jax arrays if you prefer)
    metrics = {k: jp.array(0.0) for k in self._config.reward_config.scales.keys()}
    info = {"rng": rng}

    reward = jp.array(0.0)
    done = jp.array(0.0)

    return mjx_env.State(
        data=data,
        obs=obs,
        reward=reward,
        done=done,
        metrics=metrics,
        info=info,
    )

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    """Step the environment (jittable)."""
    # scale action (kept as JAX array)
    action = action * jp.array(self._config.action_scale)

    data = mjx_env.step(
        self._mjx_model, state.data, action, n_substeps=self.n_substeps
    )
    obs = self._get_obs(data)

    raw_rewards = self._get_reward(data)

    # apply reward scales (keep as jax arrays)
    rewards = {
        k: v * jp.array(self._config.reward_config.scales[k])
        for k, v in raw_rewards.items()
    }
    # scalar reward clipped
    reward = jp.clip(sum(rewards.values()), -1e4, 1e4)

    # compute termination condition in JAX
    ee_pos = self.get_end_effector_position(data)
    target_pos = data.xpos[self._target_body_id]
    distance = jp.linalg.norm(ee_pos - target_pos)
    
    # temporary disable done condition
    # done = (distance < jp.array(self._config.success_distance_threshold)).astype(jp.float32)
    done = jp.array(0.0)
    
    # create new metrics dict (avoid in-place mutation)
    metrics = {**state.metrics, **raw_rewards}

    return mjx_env.State(
        data=data,
        obs=obs,
        reward=reward,
        done=done,
        metrics=metrics,
        info=state.info,
    )

  def _get_obs(self, data: mjx.Data) -> jax.Array:
    """Construct observation from mjx data (joint pos + vel)."""
    joint_pos = self.get_joint_positions(data)
    joint_vel = self.get_joint_velocities(data)
    ee_pos = self.get_end_effector_position(data)
    target_pos = self.get_reaching_point_position(data)

    return jax.numpy.concatenate([joint_pos, joint_vel, ee_pos, target_pos])

  def get_reaching_point_position(self, data: mjx.Data) -> jax.Array:
    """Get target reaching point position."""
    return data.xpos[self._target_body_id]


  def _get_reward(self, data: mjx.Data) -> Dict[str, jax.Array]:
    """Return raw (unscaled) reward components as JAX arrays."""
    ee_pos = self.get_end_effector_position(data)
    target_pos = data.xpos[self._target_body_id]
    distance = jp.linalg.norm(ee_pos - target_pos)

    rewards = {
        "end_effector_target": self._cost_end_effector_target(distance),
        "ground_collision": self._cost_ground_collision(data),
        "self_collision": self._cost_self_collision(data),
        "energy": self._cost_energy(data.qvel, data.qfrc_actuator),
    }
    return rewards

  def _cost_end_effector_target(self, distance: jax.Array) -> jax.Array:
    """Cost (negative reward) for distance to target."""
    return -distance

  def _cost_ground_collision(self, data: mjx.Data) -> jax.Array:
    ground_collision_vals = (
        data.sensordata[self._ground_collision_sensor_adrs] > 0
    ).astype(jp.float32)
    return (
        -jp.sum(ground_collision_vals)
        if self._ground_collision_sensor_adrs.size > 0
        else jp.array(0.0, dtype=jp.float32)
    )

  def _cost_self_collision(self, data: mjx.Data) -> jax.Array:
    self_collision_vals = (
        data.sensordata[self._self_collision_sensor_adrs] > 0
    ).astype(jp.float32)
    return (
        -jp.sum(self_collision_vals)
        if self._self_collision_sensor_adrs.size > 0
        else jp.array(0.0, dtype=jp.float32)
    )

  def _cost_energy(
      self, qvel: jax.Array, qfrc_actuator: jax.Array
  ) -> jax.Array:
    """Penalize energy consumption (negative)."""
    return jp.tanh(-jp.sum(jp.abs(qvel) * jp.abs(qfrc_actuator)))
