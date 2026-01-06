"""RL config for RobCo envs."""

from typing import Optional
from ml_collections import config_dict
from mujoco_playground._src.robco import get_default_config


def robco_ppo_config(
	env_name: str, impl: Optional[str] = None
) -> config_dict.ConfigDict:
	"""Returns tuned PPO config for the RobCo environment."""
	env_config = get_default_config(env_name)

	rl_config = config_dict.create(
		num_timesteps=100_000,
		num_evals=100,
		reward_scaling=1.0,
		episode_length=env_config.episode_length,
		normalize_observations=True,
		action_repeat=1,
		unroll_length=20,
		num_minibatches=32,
		num_updates_per_batch=4,
		discounting=0.97,
		learning_rate=3e-4,
		entropy_cost=1e-2,
	)
	return rl_config


def robco_sac_config(
	env_name: str, impl: Optional[str] = None
) -> config_dict.ConfigDict:
	"""Returns tuned SAC config for the RobCo environment."""
	env_config = get_default_config(env_name)

	rl_config = config_dict.create(
		num_timesteps=100_000,
		num_evals=100,
		reward_scaling=1.0,
		episode_length=env_config.episode_length,
		normalize_observations=False,
		action_repeat=1,
		discounting=1.00,
		learning_rate=1e-3,
		num_envs=64,
		batch_size=256,
		grad_updates_per_step=4,
		max_replay_size=10_000,
		min_replay_size=1_000,
		tau=0.005,
		network_factory=config_dict.create(
			hidden_layer_sizes=(128, 64),
		),
	)
	return rl_config
