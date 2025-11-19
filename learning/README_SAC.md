# RobcoArm SAC Training

A simple, clean implementation of Soft Actor-Critic (SAC) for training the RobcoArm environment.

## Overview

This script trains the RobcoArm robotic arm environment using the SAC algorithm from CleanRL (@vwxyzjn/cleanrl). The implementation is kept simple and clean, with minimal complexity.

## Features

- **Simple SAC implementation**: Based on CleanRL's clean and well-tested SAC algorithm
- **Wandb logging**: Optional logging of training metrics to Weights & Biases
- **Video rendering**: Automatically renders videos every 5000 steps (configurable)
- **Gymnasium wrapper**: Clean wrapper to make RobcoArm compatible with standard RL libraries

## Usage

### Basic Training (without wandb)

```bash
python learning/train_robco_sac.py
```

### Training with Wandb Logging

First, make sure you're logged into wandb:
```bash
wandb login
```

Then run with the `--track` flag:
```bash
python learning/train_robco_sac.py --track
```

### Custom Training Parameters

```bash
# Train for 500k steps
python learning/train_robco_sac.py --total-timesteps 500000

# Change render frequency
python learning/train_robco_sac.py --render-frequency 10000

# Change learning rate
python learning/train_robco_sac.py --policy-lr 1e-4 --q-lr 5e-4
```

## Command Line Arguments

- `--exp-name`: Experiment name (default: "robco_sac")
- `--seed`: Random seed (default: 1)
- `--track`: Enable wandb logging
- `--total-timesteps`: Total training timesteps (default: 100000)
- `--render-frequency`: Steps between video rendering (default: 5000)
- `--learning-starts`: Steps before learning starts (default: 5000)
- `--batch-size`: Batch size for training (default: 256)
- `--policy-lr`: Policy learning rate (default: 3e-4)
- `--q-lr`: Q-network learning rate (default: 1e-3)

Run `python learning/train_robco_sac.py --help` for all available options.

## Logged Metrics

When using `--track`, the following metrics are logged to wandb:

- `episode/return`: Episode return (cumulative reward)
- `episode/length`: Episode length (number of steps)
- `losses/qf1_loss`: Q-function 1 loss
- `losses/qf2_loss`: Q-function 2 loss
- `losses/actor_loss`: Actor (policy) loss
- `alpha`: Entropy regularization coefficient
- `video`: Rendered videos of the agent's behavior (every `--render-frequency` steps)

## Implementation Details

- **Algorithm**: Soft Actor-Critic (SAC)
- **Network architecture**: 2-layer MLP with 256 hidden units
- **Replay buffer**: 100k transitions
- **Automatic entropy tuning**: Enabled by default
- **Target network updates**: Soft updates with τ=0.005

## Notes

- The script is designed to be simple and clean - it doesn't include advanced features like distributed training or complicated logging
- Video rendering requires the environment to be renderable (uses MuJoCo's renderer)
- Training on CPU will be slower than GPU due to JAX JIT compilation overhead
