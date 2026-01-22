"""Brax SAC training script for RobcoArm environment.

This script uses Brax's fully JAX-based SAC implementation for much faster training
with vectorized environments. It provides consistent logging with the PyTorch version.

Usage:
    # Train with wandb logging (requires wandb login)
    python learning/train_robco_brax_sac.py --track
    
    # Train without wandb logging
    python learning/train_robco_brax_sac.py
    
    # Train for more steps with more environments
    python learning/train_robco_brax_sac.py --num-timesteps 1000000 --num-envs 256
"""
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import datetime
import functools
import json
import inspect
import time
import uuid

import jax
import jax.numpy as jp
import mediapy as media
import mujoco
from mujoco import mjx
import numpy as np
from brax.training.agents.sac import train as sac
from brax.training.agents.sac import networks as sac_networks
from etils import epath
from flax import serialization
from ml_collections import config_dict

from mujoco_playground import registry
from mujoco_playground import wrapper
from mujoco_playground.config import robco_params


WANDB_AVAILABLE = True
import wandb


# ============================================================================
# Configuration
# ============================================================================

def get_sac_config(env_name: str) -> config_dict.ConfigDict:
    """Returns SAC config tuned for RobcoArm."""
    return robco_params.robco_sac_config(env_name)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train RobcoArm with Brax SAC")
    parser.add_argument("--exp-name", type=str, default="robco_brax_sac", help="Experiment name")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--track", action="store_true", help="Track with wandb")
    parser.add_argument("--wandb-project-name", type=str, default="robco_brax_sac", help="Wandb project name")
    parser.add_argument("--wandb-entity", type=str, default="benjamin-ring-technical-university-of-munich", help="Wandb entity")
    
    # Environment
    parser.add_argument("--env-name", type=str, default="RobcoArm2", help="Environment name")
    parser.add_argument("--num-timesteps", type=int, default=None, help="Total training timesteps")
    parser.add_argument("--episode-length", type=int, default=None, help="Episode length (uses env default if not set)")
    
    # SAC hyperparameters
    parser.add_argument("--num-envs", type=int, default=None, help="Number of parallel environments")
    parser.add_argument("--num-eval-envs", type=int, default=32, help="Number of eval environments")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=None, help="Learning rate")
    parser.add_argument("--discounting", type=float, default=None, help="Discount factor (gamma)")
    parser.add_argument("--reward-scaling", type=float, default=None, help="Reward scaling")
    parser.add_argument("--grad-updates-per-step", type=int, default=None, help="Gradient updates per env step")
    parser.add_argument("--max-replay-size", type=int, default=None, help="Maximum replay buffer size")
    parser.add_argument("--min-replay-size", type=int, default=None, help="Minimum replay size before training")
    parser.add_argument("--normalize-observations", action="store_true", default=True, help="Normalize observations")
    
    # Logging
    parser.add_argument("--num-evals", type=int, default=None, help="Number of evaluations during training")
    parser.add_argument("--num-videos", type=int, default=1, help="Number of videos to record after training")
    parser.add_argument("--save-checkpoint", action="store_true", default=True, help="Save checkpoints")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Generate unique run name
    uid = uuid.uuid4().hex[:6]
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{args.exp_name}_{args.seed}_{uid}"
    
    print(f"=" * 60)
    print(f"Brax SAC Training for {args.env_name}")
    print(f"Run name: {run_name}")
    print(f"=" * 60)
    
    # Load environment config
    env_cfg = registry.get_default_config(args.env_name)
    sac_params = get_sac_config(args.env_name)
    
    # Override SAC params with command line args
    if args.num_timesteps:
        sac_params.num_timesteps = args.num_timesteps
    if args.episode_length:
        sac_params.episode_length = args.episode_length
    if args.num_envs:
        sac_params.num_envs = args.num_envs
    if args.batch_size:
        sac_params.batch_size = args.batch_size
    if args.learning_rate:
        sac_params.learning_rate = args.learning_rate
    if args.discounting:
        sac_params.discounting = args.discounting
    if args.reward_scaling:
        sac_params.reward_scaling = args.reward_scaling
    if args.grad_updates_per_step:
        sac_params.grad_updates_per_step = args.grad_updates_per_step
    if args.max_replay_size:
        sac_params.max_replay_size = args.max_replay_size
    if args.min_replay_size:
        sac_params.min_replay_size = args.min_replay_size
    if args.num_evals:
        sac_params.num_evals = args.num_evals
    sac_params.normalize_observations = args.normalize_observations
    
    print(f"\nEnvironment Config:\n{env_cfg}")
    print(f"\nSAC Training Parameters:\n{sac_params}")
    
    # Set up logging directory
    logdir = epath.Path("logs").resolve() / run_name
    logdir.mkdir(parents=True, exist_ok=True)
    print(f"\nLogs directory: {logdir}")
    
    # Set up checkpoint directory
    checkpoint_dir = epath.Path("checkpoints").resolve() / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint directory: {checkpoint_dir}")
    
    # Save config
    with open(checkpoint_dir / "config.json", "w", encoding="utf-8") as fp:
        config_to_save = {
            "env_config": env_cfg.to_dict(),
            "sac_params": dict(sac_params),
            "args": vars(args),
        }
        json.dump(config_to_save, fp, indent=4, default=str)
    
    # Initialize wandb
    if args.track:
        if not WANDB_AVAILABLE:
            raise ImportError("wandb is required for tracking. Install with: pip install wandb")
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config={
                "env_config": env_cfg.to_dict(),
                "sac_params": dict(sac_params),
                **vars(args),
            },
            name=run_name,
            save_code=False,
        )
    
    # Create environment
    env = registry.load(args.env_name, config=env_cfg)
    eval_env = registry.load(args.env_name, config=env_cfg)

    # Save the training script, environment source, and XML to WandB
    if args.track:
        files_to_save = [
            os.path.abspath(__file__),
            os.path.abspath(inspect.getfile(env.__class__)),
            os.path.abspath(str(env.xml_path)),
        ]
        for file_path in files_to_save:
            wandb.save(file_path, base_path=os.path.dirname(file_path), policy="now")
    
    print(f"\nObservation size: {env.observation_size}")
    print(f"Action size: {env.action_size}")
    
    # Training metrics tracking
    times = [time.monotonic()]
    training_metrics_history = []
    eval_metrics_history = []
    
    def progress_fn(num_steps, metrics):
        """Progress callback for logging during training."""
        times.append(time.monotonic())
        elapsed = times[-1] - times[1] if len(times) > 1 else times[-1] - times[0]
        sps = num_steps / elapsed if elapsed > 0 else 0
        
        # Store metrics
        metrics_with_step = {"global_step": num_steps, **metrics}
        
        # Print progress
        eval_reward = metrics.get("eval/episode_reward", 0)
        eval_reward_std = metrics.get("eval/episode_reward_std", 0)
        print(f"Step {num_steps:>8} | "
              f"Reward: {eval_reward:>8.2f} ± {eval_reward_std:.2f} | "
              f"SPS: {sps:>6.0f} | "
              f"Time: {elapsed:>6.1f}s / {elapsed/60:.1f}m")
        
        # Log detailed metrics
        if args.track:
            log_dict = {
                "global_step": num_steps,
                "episode/return": eval_reward,
                "episode/return_std": eval_reward_std,
                "training/sps": sps,
                "training/elapsed_time": elapsed,
            }
            
            # Add all metrics from Brax
            for key, value in metrics.items():
                # Convert to proper logging keys
                if key.startswith("eval/"):
                    log_dict[key] = value
                elif key.startswith("training/"):
                    log_dict[key] = value
                else:
                    log_dict[f"metrics/{key}"] = value
            
            wandb.log(log_dict, step=num_steps)
        
        # Store for later analysis
        eval_metrics_history.append(metrics_with_step)
    
    # Set up network factory
    network_factory = functools.partial(
        sac_networks.make_sac_networks,
        **sac_params.network_factory,
    )
    
    # Prepare training parameters
    training_params = dict(sac_params)
    del training_params["network_factory"]
    
    # Create the train function
    train_fn = functools.partial(
        sac.train,
        **training_params,
        network_factory=network_factory,
        seed=args.seed,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        progress_fn=progress_fn,
        num_eval_envs=args.num_eval_envs,
    )
    
    print(f"\n{'=' * 60}")
    print("Starting training...")
    print(f"{'=' * 60}\n")
    
    # Run training
    training_start = time.monotonic()
    make_inference_fn, params, metrics = train_fn(
        environment=env,
        eval_env=eval_env,
    )
    training_time = time.monotonic() - training_start
    
    print(f"\n{'=' * 60}")
    print(f"Training complete!")
    print(f"Total training time: {training_time:.1f}s / {training_time/60:.1f}m")
    print(f"Final eval reward: {metrics.get('eval/episode_reward', 0):.2f}")
    print(f"{'=' * 60}\n")
    
    # Save final checkpoint
    if args.save_checkpoint:
        print("Saving final checkpoint...")
        checkpoint_path = checkpoint_dir / "final_params.pkl"
        with open(checkpoint_path, "wb") as f:
            f.write(serialization.to_bytes(params))
        print(f"Checkpoint saved to: {checkpoint_path}")
        
        # Also save as a JAX-compatible format
        params_path = checkpoint_dir / "params.npy"
        flat_params = jax.tree_util.tree_leaves(params)
        np.save(params_path, np.array(flat_params, dtype=object), allow_pickle=True)
    
    # Generate evaluation videos
    if args.num_videos > 0:
        print(f"\nGenerating {args.num_videos} evaluation videos...")
        

        for video_idx in range(args.num_videos):
            print(f"  Recording video {video_idx + 1}/{args.num_videos}...")
            
            rng = jax.random.PRNGKey(args.seed + video_idx)

            # Re-create JIT functions for video generation
            jit_reset = jax.jit(eval_env.reset)
            jit_step = jax.jit(eval_env.step)
            inference_fn = make_inference_fn(params, deterministic=True)
            jit_inference_fn = jax.jit(inference_fn)

            state = jit_reset(rng)
            
            rollout = [state]
            total_reward = 0.0
            
            for step in range(sac_params.episode_length):
                act_rng, rng = jax.random.split(rng)
                action, _ = jit_inference_fn(state.obs, act_rng)
                state = jit_step(state, action)
                rollout.append(state)
                total_reward += float(state.reward)
                
                if state.done:
                    break
            
            print(f"    Episode reward: {total_reward:.2f}, length: {len(rollout)}")
            
            # Render video
            render_every = 2
            fps = 1.0 / eval_env.dt / render_every
            
            mj_model = eval_env.mj_model
            renderer = mujoco.Renderer(mj_model, height=480, width=640)
            
            frames = []
            for state in rollout[::render_every]:
                # Use mjx.get_data to get a fresh MjData object with the full state for rendering.
                mj_data = mjx.get_data(mj_model, state.data)
                mujoco.mj_forward(mj_model, mj_data)
                renderer.update_scene(mj_data)
                frames.append(renderer.render())
            
            # Save video locally
            video_path = logdir / f"eval_video_{video_idx}.mp4"
            media.write_video(str(video_path), frames, fps=fps)
            print(f"    Video saved to: {video_path}")
            
            # Log to wandb
            if args.track and len(frames) > 0:
                video_array = np.array(frames).transpose(0, 3, 1, 2)
                wandb.log({
                    f"video/eval_{video_idx}": wandb.Video(video_array, fps=int(fps), format="mp4"),
                    "video/episode_reward": total_reward,
                })
            
            renderer.close()
    
    # Save training history
    history_path = logdir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump({
            "eval_metrics": eval_metrics_history,
            "training_time": training_time,
            "final_reward": float(metrics.get("eval/episode_reward", 0)),
        }, f, indent=2, default=float)
    print(f"\nTraining history saved to: {history_path}")
    
    # Final logging
    if args.track:
        wandb.log({
            "final/training_time": training_time,
            "final/episode_reward": metrics.get("eval/episode_reward", 0),
        })
        wandb.finish()
    
    print(f"\n{'=' * 60}")
    print("All done!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
