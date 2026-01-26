from typing import Any, Dict, Optional, Union, Sequence

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground._src.robco.base import RobcoArmBase
import time

def default_config() -> config_dict.ConfigDict:
  """Default configuration for basic RobcoArm environments."""
  return config_dict.create(
      ctrl_dt=0.01,
      sim_dt=0.001,
      episode_length=500,
      action_repeat=1,
      vision=False,
      impl="jax",
      action_scale=4.7124,
      nconmax=4096,
      njmax=128,
      success_distance_threshold=0.05,
      max_action_delta=0.004,
      reward_config=config_dict.create(
          scales=config_dict.create(
              end_effector_target=1,
              ground_collision=0.5,
              self_collision=0.5,
              energy=0.0,
              action_rate=0.01,
          ),
      ),
  )


class RobcoArmPosition(RobcoArmBase):
  """Basic Robco arm environment (mjx-safe, jittable)."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, Sequence[Any]]]] = None,
  ) -> None:
    xml_path = mjx_env.ROOT_PATH / "robco" / "xmls" / "robco_arm_position.xml"
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

    data = mjx.forward(self._mjx_model, data)

    rng, rng_obs = jax.random.split(jax.random.PRNGKey(int(time.time())))
    obs = self._get_obs(data, rng_obs)

    # metrics initialized as plain floats (or jax arrays if you prefer)
    metrics = {k: jp.array(0.0) for k in self._config.reward_config.scales.keys()}
    info = {
        "rng": rng,
        "last_act": jp.zeros(self._mjx_model.nu),
    }

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
    # Apply delta clamping to unscaled action
    action_delta = action - state.info["last_act"]
    action_delta = jp.clip(
        action_delta,
        -jp.array(self._config.max_action_delta),
        jp.array(self._config.max_action_delta)
    )
    action_unscaled = state.info["last_act"] + action_delta
    
    # Scale action for simulation
    action_scaled = action_unscaled * jp.array(self._config.action_scale)

    data = mjx_env.step(
        self._mjx_model, state.data, action_scaled, n_substeps=self.n_substeps
    )
    
    rng, rng_obs = jax.random.split(state.info["rng"])
    obs = self._get_obs(data, rng_obs)

    raw_rewards = self._get_reward(data, action_unscaled, state.info)

    # apply reward scales (keep as jax arrays)
    rewards = {
        k: v * jp.array(self._config.reward_config.scales[k])
        for k, v in raw_rewards.items()
    }
    # scalar reward clipped
    reward = jp.clip(sum(rewards.values()), -1e4, 1e4)

    # compute termination condition in JAX
    # ee_pos = self.get_end_effector_position(data)
    # target_pos = data.xpos[self._target_body_id]
    # distance = jp.linalg.norm(ee_pos - target_pos)
    # done = (distance < jp.array(self._config.success_distance_threshold)).astype(jp.float32)
    done = jp.array(0.0)
    
    # create new metrics dict (avoid in-place mutation)
    metrics = {**state.metrics, **raw_rewards}
    
    # update last action and rng
    info = {**state.info, "last_act": action_unscaled, "rng": rng}

    return mjx_env.State(
        data=data,
        obs=obs,
        reward=reward,
        done=done,
        metrics=metrics,
        info=info,
    )

  def _get_obs(self, data: mjx.Data, rng: jax.Array) -> jax.Array:
    """Construct observation from mjx data (joint pos + vel)."""
    joint_pos = self.get_joint_positions(data)
    joint_vel = self.get_joint_velocities(data)
    
    # add random position to pos and vel. random uniform noise
    rng, rng_pos = jax.random.split(rng)
    joint_pos += jax.random.uniform(
        rng_pos, shape=joint_pos.shape, minval=-0.0001, maxval=0.0001
    )
    rng, rng_vel = jax.random.split(rng)
    joint_vel += jax.random.uniform(
        rng_vel, shape=joint_vel.shape, minval=-0.0001, maxval=0.0001
    )
    
    target_pos = self.get_reaching_point_position(data)

    return jax.numpy.concatenate([joint_pos, joint_vel, target_pos])

  def get_reaching_point_position(self, data: mjx.Data) -> jax.Array:
    """Get target reaching point position."""
    return data.xpos[self._target_body_id]

  def _get_reward(
      self, data: mjx.Data, action: jax.Array, info: Dict[str, Any]
  ) -> Dict[str, jax.Array]:
    """Return raw (unscaled) reward components as JAX arrays."""
    ee_pos = self.get_end_effector_position(data)
    target_pos = data.xpos[self._target_body_id]
    distance = jp.linalg.norm(ee_pos - target_pos)

    rewards = {
        "end_effector_target": self._cost_end_effector_target(distance),
        "ground_collision": self._cost_ground_collision(data),
        "self_collision": self._cost_self_collision(data),
        "energy": self._cost_energy(data.qvel, data.qfrc_actuator),
        "action_rate": self._cost_action_rate(action, info["last_act"]),
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
    energy = -jp.sum(jp.abs(qvel) * jp.abs(qfrc_actuator))
    return jp.nan_to_num(energy, nan=0.0, posinf=0.0, neginf=0.0)

  def _cost_action_rate(self, action: jax.Array, last_action: jax.Array) -> jax.Array:
    """Penalize rapid changes in action (negative)."""
    action_rate = -jp.sum(jp.square(action - last_action))
    return jp.nan_to_num(action_rate, nan=0.0, posinf=0.0, neginf=0.0)
