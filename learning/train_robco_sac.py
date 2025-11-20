"""Simple SAC training script for RobcoArm environment.

Based on CleanRL's SAC implementation (@vwxyzjn/cleanrl).

Usage:
    # Train with wandb logging (requires wandb login)
    python learning/train_robco_sac.py --track
    
    # Train without wandb logging
    python learning/train_robco_sac.py
    
    # Train for more steps
    python learning/train_robco_sac.py --total-timesteps 500000
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["MUJOCO_GL"] = "egl"

import argparse
import collections
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

from mujoco_playground import registry

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# ============================================================================
# Gymnasium Wrapper for RobcoArm
# ============================================================================

class RobcoArmGymnasiumWrapper(gym.Env):
    """Gymnasium wrapper for RobcoArm environment."""

    def __init__(self, render_mode=None):
        super().__init__()
        self.env = registry.load("RobcoArm")
        self.render_mode = render_mode
        
        # Define action and observation spaces
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.env.action_size,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.env.observation_size,), dtype=np.float32
        )
        
        # JIT compiled functions
        self._jit_reset = jax.jit(self.env.reset)
        self._jit_step = jax.jit(self.env.step)
        
        self._state = None
        self._rng = jax.random.PRNGKey(0)
        self._step_count = 0

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = jax.random.PRNGKey(seed)
        
        self._rng, reset_rng = jax.random.split(self._rng)
        self._state = self._jit_reset(reset_rng)
        self._step_count = 0
        
        obs = np.array(self._state.obs, dtype=np.float32)
        info = {}
        return obs, info

    def step(self, action):
        action = jnp.array(action, dtype=jnp.float32)
        self._state = self._jit_step(self._state, action)
        self._step_count += 1
        
        obs = np.array(self._state.obs, dtype=np.float32)
        reward = float(self._state.reward)
        terminated = bool(self._state.done)
        truncated = self._step_count >= 1000  # Episode length
        info = {}
        
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            # Use mujoco to render the current state
            mj_data = mujoco.MjData(self.env.mj_model)
            mj_data.qpos[:] = np.array(self._state.data.qpos)
            mj_data.qvel[:] = np.array(self._state.data.qvel)
            mujoco.mj_forward(self.env.mj_model, mj_data)
            
            renderer = mujoco.Renderer(self.env.mj_model, height=480, width=640)
            renderer.update_scene(mj_data)
            return renderer.render()
        return None


# ============================================================================
# SAC Networks
# ============================================================================

LOG_STD_MAX = 2
LOG_STD_MIN = -5


class SoftQNetwork(nn.Module):
    """Soft Q-Network."""

    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    """SAC Actor network."""

    def __init__(self, obs_dim, action_dim, action_scale, action_bias):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, action_dim)
        self.fc_logstd = nn.Linear(256, action_dim)
        
        self.register_buffer("action_scale", torch.tensor(action_scale, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor(action_bias, dtype=torch.float32))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        # Clamp mean to prevent extreme values that can cause numerical instability
        mean = torch.clamp(mean, min=-20, max=20)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


# ============================================================================
# Replay Buffer
# ============================================================================

class ReplayBuffer:
    """Simple replay buffer."""

    def __init__(self, obs_dim, action_dim, buffer_size, device):
        self.obs = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((buffer_size,), dtype=np.float32)
        self.dones = np.zeros((buffer_size,), dtype=np.float32)
        
        self.ptr = 0
        self.size = 0
        self.max_size = buffer_size
        self.device = device

    def add(self, obs, next_obs, action, reward, done):
        self.obs[self.ptr] = obs
        self.next_obs[self.ptr] = next_obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        
        return (
            torch.from_numpy(self.obs[idxs]).to(self.device),
            torch.from_numpy(self.actions[idxs]).to(self.device),
            torch.from_numpy(self.next_obs[idxs]).to(self.device),
            torch.from_numpy(self.rewards[idxs]).to(self.device),
            torch.from_numpy(self.dones[idxs]).to(self.device),
        )


# ============================================================================
# Training Configuration
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="robco_sac", help="Experiment name")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--cuda", action="store_true", default=True, help="Use CUDA")
    parser.add_argument("--track", action="store_true", help="Track with wandb")
    parser.add_argument("--wandb-project-name", type=str, default="mujoco_playground", help="Wandb project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="Wandb entity")
    
    # Environment
    parser.add_argument("--total-timesteps", type=int, default=100000, help="Total training timesteps")
    
    # SAC hyperparameters
    parser.add_argument("--buffer-size", type=int, default=100000, help="Replay buffer size")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.005, help="Target network update rate")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--learning-starts", type=int, default=5000, help="Steps before learning starts")
    parser.add_argument("--policy-lr", type=float, default=3e-4, help="Policy learning rate")
    parser.add_argument("--q-lr", type=float, default=1e-3, help="Q-network learning rate")
    parser.add_argument("--policy-frequency", type=int, default=2, help="Policy update frequency")
    parser.add_argument("--target-network-frequency", type=int, default=1, help="Target network update frequency")
    parser.add_argument("--alpha", type=float, default=0.2, help="Entropy regularization coefficient")
    parser.add_argument("--autotune", action="store_true", default=True, help="Auto-tune alpha")
    
    # Logging
    parser.add_argument("--render-frequency", type=int, default=5000, help="Video rendering frequency")
    
    return parser.parse_args()


@dataclass
class Args:
    """Training arguments (deprecated, use parse_args instead)."""
    exp_name: str = "robco_sac"
    seed: int = 1
    cuda: bool = True
    track: bool = False  # Set to True to enable wandb logging
    wandb_project_name: str = "mujoco_playground"
    wandb_entity: str = None
    
    # Environment
    total_timesteps: int = 100000
    
    # SAC hyperparameters
    buffer_size: int = 100000
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    learning_starts: int = 5000
    policy_lr: float = 3e-4
    q_lr: float = 1e-3
    policy_frequency: int = 2
    target_network_frequency: int = 1
    alpha: float = 0.2
    autotune: bool = True
    
    # Logging
    render_frequency: int = 5000


def main():
    args = parse_args()
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    
    # Initialize wandb
    if args.track:
        if not WANDB_AVAILABLE:
            raise ImportError("wandb is required for tracking. Install it with: pip install wandb")
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),
            name=run_name,
            save_code=True,
        )
    
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")
    
    # Create environment
    env = RobcoArmGymnasiumWrapper(render_mode="rgb_array")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print(f"Observation dim: {obs_dim}, Action dim: {action_dim}")
    
    # Create networks
    action_scale = (env.action_space.high - env.action_space.low) / 2.0
    action_bias = (env.action_space.high + env.action_space.low) / 2.0
    
    print(f"action_scale: {action_scale} action_bias: {action_bias}")
    
    actor = Actor(obs_dim, action_dim, action_scale, action_bias).to(device)
    qf1 = SoftQNetwork(obs_dim, action_dim).to(device)
    qf2 = SoftQNetwork(obs_dim, action_dim).to(device)
    qf1_target = SoftQNetwork(obs_dim, action_dim).to(device)
    qf2_target = SoftQNetwork(obs_dim, action_dim).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(actor.parameters(), lr=args.policy_lr)
    
    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(env.action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        alpha_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha
    
    # Replay buffer
    replay_buffer = ReplayBuffer(obs_dim, action_dim, args.buffer_size, device)
    
    # Training loop
    obs, _ = env.reset(seed=args.seed)
    episode_return = 0
    episode_length = 0
    
    recent_returns = collections.deque(maxlen=100)
    recent_lengths = collections.deque(maxlen=100)

    for global_step in range(args.total_timesteps):
        # Collect experience
        if global_step < args.learning_starts:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action, _, _ = actor.get_action(torch.Tensor(obs).unsqueeze(0).to(device))
                action = action.cpu().numpy()[0]
        
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        episode_return += reward
        episode_length += 1
        
        # Store transition
        replay_buffer.add(obs, next_obs, action, reward, float(terminated))
        obs = next_obs
        
        # Episode end
        if done:
            recent_returns.append(episode_return)
            recent_lengths.append(episode_length)
            avg_return = np.mean(recent_returns)
            avg_length = np.mean(recent_lengths)

            if args.track:
                wandb.log({
                    "episode/return": episode_return,
                    "episode/length": episode_length,
                    "episode/avg_return": avg_return,
                    "episode/avg_length": avg_length,
                    "global_step": global_step,
                })
            print(f"Step {global_step}: episode_return={episode_return:.2f}, episode_length={episode_length}, avg_return={avg_return:.2f}")
            
            obs, _ = env.reset()
            episode_return = 0
            episode_length = 0
        
        # Training
        if global_step >= args.learning_starts:
            # Sample batch
            obs_batch, action_batch, next_obs_batch, reward_batch, done_batch = replay_buffer.sample(args.batch_size)
            
            # Update Q-functions
            with torch.no_grad():
                next_actions, next_log_pi, _ = actor.get_action(next_obs_batch)
                qf1_next_target = qf1_target(next_obs_batch, next_actions)
                qf2_next_target = qf2_target(next_obs_batch, next_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_log_pi
                next_q_value = reward_batch.unsqueeze(-1) + (1 - done_batch.unsqueeze(-1)) * args.gamma * min_qf_next_target
            
            qf1_a_values = qf1(obs_batch, action_batch)
            qf2_a_values = qf2(obs_batch, action_batch)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss
            
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()
            
            # Update policy
            if global_step % args.policy_frequency == 0:
                for _ in range(args.policy_frequency):
                    pi, log_pi, _ = actor.get_action(obs_batch)
                    qf1_pi = qf1(obs_batch, pi)
                    qf2_pi = qf2(obs_batch, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()
                    
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()
                    
                    # Update alpha
                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(obs_batch)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()
                        
                        alpha_optimizer.zero_grad()
                        alpha_loss.backward()
                        alpha_optimizer.step()
                        alpha = log_alpha.exp().item()
            
            # Update target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
            
            # Logging
            if global_step % 1000 == 0:
                if args.track:
                    wandb.log({
                        "losses/qf1_loss": qf1_loss.item(),
                        "losses/qf2_loss": qf2_loss.item(),
                        "losses/actor_loss": actor_loss.item(),
                        "alpha": alpha,
                        "global_step": global_step,
                    })
        
        # Render video periodically
        if global_step > 0 and global_step % args.render_frequency == 0:
            print(f"Rendering video at step {global_step}...")
            frames = []
            obs, _ = env.reset()
            for _ in range(200):
                with torch.no_grad():
                    action, _, _ = actor.get_action(torch.Tensor(obs).unsqueeze(0).to(device))
                    action = action.cpu().numpy()[0]
                obs, _, terminated, truncated, _ = env.step(action)
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
                if terminated or truncated:
                    break
            
            if frames and args.track:
                # Convert to video and log to wandb
                video = np.array(frames).transpose(0, 3, 1, 2)
                wandb.log({
                    "video": wandb.Video(video, fps=30, format="mp4"),
                    "global_step": global_step,
                })
            
            # Reset environment after rendering
            obs, _ = env.reset()
    
    env.close()
    if args.track:
        wandb.finish()
    print("Training complete!")


if __name__ == "__main__":
    main()
