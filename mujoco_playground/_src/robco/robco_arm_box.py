import os
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
      ctrl_dt=0.05,
      sim_dt=0.005,
      episode_length=400,
      action_repeat=1,
      vision=False,
      impl="jax",
      action_scale=4.7124,
      nconmax=4096,
      njmax=128,
      box_on_target_distance_threshold=0.1,
      reward_config=config_dict.create(
          scales=config_dict.create(
              box_on_target=500.0,  # Add a large bonus for box being on target
              box_target_distance=10.0,  # Reward for distance from box to target
              distance_end_effector_box=0.0,  # Reward for EE getting/staying close to box
              alignment=70.0,  # Reward for aligning EE-Box-Target
              box_velocity=3.0,  # Reward for box velocity towards target

              return_to_neutral=40.0,  # Reward for returning to neutral pose after success
              
              ground_collision=1,
              self_collision=1,
              energy=0.001,
              control_cost=0.001,  # Add a small penalty for large actions
              action_rate=0.01,  # Penalty for big changes between actions (smoothness)
          ),
      ),
  )


class RobcoArmBox(RobcoArmBase):
  """Basic Robco arm environment (mjx-safe, jittable)."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, Sequence[Any]]]] = None,
  ) -> None:
    xml_path = mjx_env.ROOT_PATH / "robco" / "xmls" / "robco_arm_box.xml"
    super().__init__(xml_path, config, config_overrides)

    # Save this file to wandb code section if wandb is active
    try:
      import wandb
      if wandb.run:
        wandb.run.log_code(
            root=os.path.dirname(__file__),
            include_fn=lambda path: path.endswith(os.path.basename(__file__))
        )
    except (ImportError, AttributeError):
      pass

    # name of target body in XML (must exist)
    self.target_body_name = "target"
    self._target_body_id = self._mj_model.body(self.target_body_name).id

    # movable box object body id
    self.box_body_name = "box_object"
    self._box_body_id = self._mj_model.body(self.box_body_name).id

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
    rng, box_rng, target_rng = jax.random.split(rng, 3)

    # Initialize qpos and qvel to zeros
    #qpos = jp.zeros(self._mjx_model.nq)
    qpos = self._mjx_model.qpos0
    qvel = jp.zeros(self._mjx_model.nv)

    data = mjx.make_data(
        self._mjx_model,
        impl=self.mjx_model.impl.value,
        nconmax=self._config.nconmax,
        njmax=self._config.njmax,
    )

    # Randomize box and target positions
    # Use a circular region for more natural placement
    '''
    radius = jax.random.uniform(box_rng, minval=0.3, maxval=0.6)
    angle = jax.random.uniform(box_rng, minval=-jp.pi, maxval=jp.pi)
    box_xy = jp.array([radius * jp.cos(angle), radius * jp.sin(angle)])
    box_z = 0.1  # Box half-height, so it rests on the ground
    qpos = qpos.at[self.num_joints:self.num_joints+3].set(jp.array([1, 1, box_z]))
    qpos = qpos.at[self.num_joints+3:self.num_joints+7].set(jp.array([1.0, 0.0, 0.0, 0.0]))  
    # Randomize target to be on the same height as the box center
    # Place target in a different circular region
    radius_target = jax.random.uniform(target_rng, minval=0.4, maxval=0.7)
    angle_target = jax.random.uniform(target_rng, minval=-jp.pi, maxval=jp.pi)
    target_xy = jp.array([radius_target * jp.cos(angle_target), radius_target * jp.sin(angle_target)])
    target_pos = jp.array([1, 0, box_z])
    '''
    
    # Update data with new qpos and target position using .replace()
    #data = data.replace(qpos=qpos, qvel=qvel, xpos=data.xpos.at[self._target_body_id].set(target_pos))
    data = data.replace(qpos=qpos, qvel=qvel)

    # forward to compute derived quantities
    data = mjx.forward(self._mjx_model, data)

    obs = self._get_obs(data)

    # metrics initialized as plain floats (or jax arrays if you prefer)
    metrics = {k: jp.array(0.0, dtype=jp.float32) for k in self._config.reward_config.scales.keys()}
    info = {"rng": rng, "last_action": jp.zeros(self._mjx_model.nu, dtype=jp.float32)}

    reward = jp.array(0.0, dtype=jp.float32)
    done = jp.array(0.0, dtype=jp.float32)

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
    action = action * jp.array(self._config.action_scale, dtype=jp.float32)

    data = mjx_env.step(
        self._mjx_model, state.data, action, n_substeps=self.n_substeps
    )
    obs = self._get_obs(data)

    raw_rewards = self._get_reward(data, action, state.info)

    # apply reward scales (keep as jax arrays)
    rewards = {
        k: v * jp.array(self._config.reward_config.scales[k], dtype=jp.float32)
        for k, v in raw_rewards.items()
    }
    # scalar reward clipped
    reward = jp.clip(sum(rewards.values()), -1e4, 1e4)

    # compute termination condition in JAX / currently disabled
    box_pos = self.get_box_position(data)
    target_pos = self.get_reaching_point_position(data)
    distance = jp.linalg.norm(box_pos - target_pos)
    done = jp.array(0.0)
    
    # create new metrics dict (avoid in-place mutation)
    metrics = {**state.metrics, **raw_rewards}

    # Update info with current action for the next step
    info = {**state.info, "last_action": action}

    return mjx_env.State(
        data=data,
        obs=obs,
        reward=reward,
        done=done,
        metrics=metrics,
        info=info,
    )

  def _get_obs(self, data: mjx.Data) -> jax.Array:
    """Construct observation from mjx data (joint pos + vel)."""
    joint_pos = self.get_joint_positions(data) # contains the box data
    joint_vel = self.get_joint_velocities(data)
    target_pos = self.get_target_point_position(data)

    return jax.numpy.concatenate([joint_pos, joint_vel, target_pos])

  def _get_reward(self, data: mjx.Data, action: jax.Array, info: Dict[str, Any]) -> Dict[str, jax.Array]:
    """Return raw (unscaled) reward components as JAX arrays."""
    box_pos = self.get_box_position(data)
    target_pos = self.get_target_point_position(data)
    ee_pos = self.get_end_effector_position(data)

    rewards = {
        "box_on_target": self._reward_box_on_target(box_pos, target_pos),
        "box_target_distance": self._reward_box_target_distance(box_pos, target_pos, ee_pos),
        "distance_end_effector_box": self._reward_distance_end_effector_box(ee_pos, box_pos),
        "alignment": self._reward_alignment(box_pos, target_pos, ee_pos),
        "box_velocity": self._reward_box_velocity(data, box_pos, target_pos, ee_pos),
        
        "return_to_neutral": self._reward_return_to_neutral(data, box_pos, target_pos),
        
        "ground_collision": self._cost_ground_collision(data),
        "self_collision": self._cost_self_collision(data),
        "energy": self._cost_energy(data.qvel, data.qfrc_actuator),
        "control_cost": self._cost_control(action),
        "action_rate": self._cost_action_rate(action, info),
    }
    return rewards  
  
  #####################################
  # Utility functions

  def get_target_point_position(self, data: mjx.Data) -> jax.Array:
    """Get target point position."""
    return data.xpos[self._target_body_id]

  def get_box_position(self, data: mjx.Data) -> jax.Array:
    """Get movable box object position."""
    return data.xpos[self._box_body_id]  

  def get_box_orientation(self, data: mjx.Data) -> jax.Array:
    """Get movable box object orientation."""
    return data.xquat[self._box_body_id]

  def get_box_velocity(self, data: mjx.Data) -> jax.Array:
    """Get movable box object linear velocity."""
    return data.cvel[self._box_body_id, 3:]

  def _is_success(self, box_pos: jax.Array, target_pos: jax.Array) -> jax.Array:
    """Check if box is within threshold distance of target."""
    box_target_dist = jp.linalg.norm(box_pos[:2] - target_pos[:2])
    return box_target_dist < self._config.box_on_target_distance_threshold  

  def _get_alignment_dist(self, box_pos: jax.Array, target_pos: jax.Array, ee_pos: jax.Array) -> jax.Array:
    # Use 2D vector to ensure push point is at same height as box
    vec_box_target = target_pos[:2] - box_pos[:2]
    box_target_dir = vec_box_target / (jp.linalg.norm(vec_box_target) + 1e-6)
    box_target_dir = jp.concatenate([box_target_dir, jp.array([0.0], dtype=jp.float32)])

    # Define a "pushing segment" 0 to 15cm behind the box
    vec_box_ee = ee_pos - box_pos
    proj = jp.dot(vec_box_ee, box_target_dir)
    # Clamp projection to be between -15cm (behind) and 0cm (at box center)
    proj_clamped = jp.clip(proj, -0.15, 0.0)
    closest_point = box_pos + proj_clamped * box_target_dir
    return jp.linalg.norm(ee_pos - closest_point)

  #####################################
  # Positive reward component functions

  def _reward_box_on_target(self, box_pos: jax.Array, target_pos: jax.Array) -> jax.Array:
    """Reward for box being on target."""
    return self._is_success(box_pos, target_pos).astype(jp.float32)
  
  def _reward_box_target_distance(
      self, box_pos: jax.Array, target_pos: jax.Array, ee_pos: jax.Array
  ) -> jax.Array:
    """Reward for distance from box to target."""
    box_target_dist = jp.linalg.norm(box_pos[:2] - target_pos[:2])
    reward = 1.0 / (1.0 + 1.0 * box_target_dist)
    #reward = 1.0 - box_target_dist  # Linear reward

    # Gate reward by alignment: if robot is not aligned, distance reward is suppressed.
    dist_ee_push = self._get_alignment_dist(box_pos, target_pos, ee_pos)
    is_aligned = dist_ee_push < 0.05
    # If box is on target, give max reward (2.0) regardless of alignment/distance
    is_success = self._is_success(box_pos, target_pos)
    
    # If not aligned, give 0 reward to force realignment
    return jp.where(is_success, 2.0, jp.where(is_aligned, reward * 2.0, 0.0))
  
  def _reward_distance_end_effector_box(self, ee_pos: jax.Array, box_pos: jax.Array) -> jax.Array:
    """Reward for distance from end-effector to the box."""
    ee_box_dist = jp.linalg.norm(ee_pos - box_pos)
    return 1.0 / (1.0 + 5.0 * ee_box_dist)

  def _reward_alignment(self, box_pos: jax.Array, target_pos: jax.Array, ee_pos: jax.Array) -> jax.Array:
    """Alignment reward: encourage EE to be behind the box relative to target"""
    dist_ee_push = self._get_alignment_dist(box_pos, target_pos, ee_pos)
    reward = 1.0 / (1.0 + 2.0 * dist_ee_push)
    is_aligned = dist_ee_push < 0.05
    
    # If box is on target, we consider it "perfectly aligned" (max reward) to allow robot to leave
    is_success = self._is_success(box_pos, target_pos)
    return jp.where(is_success, 2.0, jp.where(is_aligned, reward * 2.0, reward))

  def _reward_box_velocity(
      self, data: mjx.Data, box_pos: jax.Array, target_pos: jax.Array, ee_pos: jax.Array
  ) -> jax.Array:
    """Reward for box moving towards target."""
    box_vel = self.get_box_velocity(data)
    vec_box_target = target_pos[:2] - box_pos[:2]
    dist = jp.linalg.norm(vec_box_target)
    # Normalized direction
    dir_box_target = vec_box_target / (dist + 1e-6)
    # Project velocity onto direction (2D)
    vel_proj = jp.dot(box_vel[:2], dir_box_target)
    vel_proj = jp.minimum(vel_proj, self._config.target_push_speed)
    
    # Gate reward by alignment: if robot is not aligned, velocity reward is suppressed.
    dist_ee_push = self._get_alignment_dist(box_pos, target_pos, ee_pos)
    alignment_gate = 1.0 / (1.0 + 5.0 * dist_ee_push)

    # Scale reward by distance (tanh) AND alignment
    return vel_proj * jp.tanh(5.0 * dist) * alignment_gate

  def _reward_return_to_neutral(self, data: mjx.Data, box_pos: jax.Array, target_pos: jax.Array) -> jax.Array:
    """Reward for returning to neutral configuration, active only when box is on target."""
    is_success = self._is_success(box_pos, target_pos)
    
    current_qpos = data.qpos[:self.num_joints]
    neutral_qpos = self._mjx_model.qpos0[:self.num_joints]
    dist_to_neutral = jp.linalg.norm(current_qpos - neutral_qpos)
    # return jp.where(is_success, 1.0 / (1.0 + 2.0 * dist_to_neutral), 0.0)
    return jp.where(is_success, 1.0 - 0.5 * dist_to_neutral, 0.0)

  #####################################
  # Negative reward component functions
  
  def _cost_ground_collision(self, data: mjx.Data) -> jax.Array:
    ground_collision_vals = data.sensordata[self._ground_collision_sensor_adrs]
    return (
        -jp.sum(jp.square(ground_collision_vals))
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

  def _cost_energy(self, qvel: jax.Array, qfrc_actuator: jax.Array) -> jax.Array:
    """Penalize energy consumption (negative)."""
    return -jp.sum(jp.abs(qvel) * jp.abs(qfrc_actuator))

  def _cost_control(self, action: jax.Array) -> jax.Array:
    """Penalize large actions to encourage smoothness."""
    return -jp.sum(jp.square(action))

  def _cost_action_rate(self, action: jax.Array, info: Dict[str, Any]) -> jax.Array:
    """Penalize large changes in action to encourage smoothness."""
    return -jp.sum(jp.square(action - info["last_action"]))