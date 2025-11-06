import abc
from etils import epath
import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
from mujoco_playground._src import mjx_env
from typing import Any, Dict, Optional, Union, Mapping, Sequence, Callable, List, Tuple
import numpy as np


Observation = Union[jax.Array, Mapping[str, jax.Array]]

class RobcoEnv(mjx_env.MjxEnv):
    # typed attributes
    _model_assets: Dict[str, bytes]
    _mj_model: mujoco.MjModel
    _mjx_model: mjx.Model
    _xml_path: str
    renderer: mujoco.Renderer
    _obj_body: int
    _gripper_site: int
    _obj_qposadr: int
    _action_scale: jax.Array

    @classmethod
    def default_config(cls) -> config_dict.ConfigDict:
        return config_dict.create(
            impl='jax',
            sim_dt=0.002,
            ctrl_dt=0.02,
            episode_length=1000,
        )

    def __init__(
        self,
        xml_path: str = 'robco_robot_with_gripper.xml',
        config: Optional[config_dict.ConfigDict] = None,
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ) -> None:
        if config is None:
            config = self.default_config()
        super().__init__(config, config_overrides)

        self._model_assets = {}
        # Use from_xml_path instead of from_xml_string so that <include> directives
        # (e.g. base.xml defining material 'red') are correctly resolved by MuJoCo.
        # from_xml_string would not have a filesystem context and thus included assets
        # like materials and additional bodies would be missing, causing ValueErrors.
        self._mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self._mj_model.opt.timestep = self.sim_dt
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
        self._xml_path = xml_path
        self.renderer = mujoco.Renderer(self._mj_model)

        # Get body and site IDs
        self._obj_body = self._mj_model.body('cube_red').id
        self._gripper_site = self._mj_model.site('joint0_site').id
        
        # Get qpos indices for object (free joint has 7 dofs)
        obj_jnt_id = self._mj_model.body(self._obj_body).jntadr[0]  # assuming one joint per body
        self._obj_qposadr = self._mj_model.jnt_qposadr[obj_jnt_id]
        
        self._action_scale = jp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])  # scale gripper less

    def reset(self, rng: jax.Array) -> mjx_env.State:
        state_data: mjx.Data = mjx.make_data(self._mjx_model)
        
        # Randomize object position
        obj_pos = jp.array([0.3, 0.3, 0.025]) + jax.random.uniform(rng, (3,), minval=-0.1, maxval=0.1)
        
        # Randomize target position
        target_pos = jp.array([0.5, 0.5, 0.05]) + jax.random.uniform(rng, (3,), minval=-0.1, maxval=0.1)
        info: Dict[str, Any] = {'target_pos': target_pos}
        
        # Set object position in qpos (free joint: 3 pos + 4 quat)
        obj_qpos = jp.concatenate([obj_pos, jp.array([1.0, 0.0, 0.0, 0.0])])  # quaternion for no rotation
        state_data = state_data.replace(qpos=state_data.qpos.at[self._obj_qposadr:self._obj_qposadr+7].set(obj_qpos))
        
        state_data = mjx.forward(self._mjx_model, state_data)
        
        obs = self._get_obs(state_data, info)

        # Initialize reward/done/metrics to match mjx_env.State signature.
        reward = jp.array(0.0)
        done = jp.array(False)
        metrics: Dict[str, jax.Array] = {}

        return mjx_env.State(
            data=state_data, obs=obs, reward=reward, done=done, metrics=metrics, info=info
        )

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        state_data: mjx.Data = state.data
        info: Dict[str, Any] = state.info

        # Validate and apply action to ctrl. Mujoco JAX does not expose mjx.set_ctrl; instead
        # we directly replace the ctrl field on the immutable mjx.Data structure.
        if action.shape[-1] != self.action_size:
            raise ValueError(f"Action shape {action.shape} does not match expected ({self.action_size},)")

        # Clip actions to [-1, 1] before scaling (typical convention) then scale per-element.
        scaled_action = jp.clip(action, -1.0, 1.0) * self._action_scale

        # Replace ctrl; mjx.Data is a Frozen dataclass-like object supporting .replace.
        state_data = state_data.replace(ctrl=scaled_action)

        # Step
        state_data = mjx.step(self._mjx_model, state_data)

        # Compute reward
        reward = self._get_reward(state_data, info)

        # Check done
        done = self._is_done(state_data, info)

        obs = self._get_obs(state_data, info)

        metrics: Dict[str, jax.Array] = {}

        return mjx_env.State(
            data=state_data, obs=obs, reward=reward, done=done, metrics=metrics, info=info
        )

    def _get_obs(self, data: mjx.Data, info: Dict[str, Any]) -> Observation:
        # Observations: robot qpos (actuated), qvel, object pos, target pos, gripper pos
        obs = jp.concatenate([
            jp.array(data.qpos[7:14]),  # actuated joints: floating is 0-6, joints 7-12, gripper 13
            jp.array(data.qvel[6:13]),  # corresponding vel
            jp.array(data.xpos[self._obj_body]),
            info['target_pos'],
            jp.array(data.site_xpos[self._gripper_site]),
        ])
        return obs

    def _get_reward(self, data: mjx.Data, info: Dict[str, Any]) -> jax.Array:
        obj_pos = jp.array(data.xpos[self._obj_body])
        target_pos = info['target_pos']
        gripper_pos = jp.array(data.site_xpos[self._gripper_site])
        
        # Distance from gripper to object
        gripper_obj_dist = jp.linalg.norm(gripper_pos - obj_pos)
        
        # Distance from object to target
        obj_target_dist = jp.linalg.norm(obj_pos - target_pos)
        
        # Reward for close to object
        reward_gripper_obj = 1 - jp.tanh(5 * gripper_obj_dist)
        
        # Reward for object close to target
        reward_obj_target = 1 - jp.tanh(5 * obj_target_dist)
        
        # Bonus for lifting object
        lifted = obj_pos[2] > 0.1
        reward_lift = lifted * 0.5
        
        # Success bonus
        success = obj_target_dist < 0.05
        reward_success = success * 1.0
        
        total_reward = reward_gripper_obj + reward_obj_target + reward_lift + reward_success

        return total_reward

    def _is_done(self, data: mjx.Data, info: Dict[str, Any]) -> jax.Array:
        obj_pos = jp.array(data.xpos[self._obj_body])
        target_pos = info['target_pos']
        
        # Done if object is at target or out of bounds
        success = jp.linalg.norm(obj_pos - target_pos) < 0.05
        out_of_bounds = jp.any(jp.abs(obj_pos) > 1.0) | (obj_pos[2] < 0.0)
        
        done = success | out_of_bounds

        return done
    
    def render(self, state: mjx_env.State) -> np.ndarray:
        data = mjx.get_data(self._mjx_model, state.data)
        self.renderer.update_scene(data)
        return self.renderer.render()

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