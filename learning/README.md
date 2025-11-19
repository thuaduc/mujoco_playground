# Learning RL Agents

In this directory, we demonstrate learning RL agents from MuJoCo Playground environments using [Brax](https://github.com/google/brax), [RSL-RL](https://github.com/leggedrobotics/rsl_rl), and [CleanRL](https://github.com/vwxyzjn/cleanrl). We provide three entrypoints from the command line: `python train_jax_ppo.py`, `python train_rsl_rl.py`, and `python train_robco_sac.py`.

For more detailed tutorials on using MuJoCo Playground for RL, see:

1. Intro. to the Playground with DM Control Suite [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco_playground/blob/main/learning/notebooks/dm_control_suite.ipynb)
2. Locomotion Environments [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco_playground/blob/main/learning/notebooks/locomotion.ipynb)
3. Manipulation Environments [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco_playground/blob/main/learning/notebooks/manipulation.ipynb)
4. Training CartPole from Vision [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco_playground/blob/main/learning/notebooks/training_vision_1.ipynb)
5. Robotic Manipulation from Vision [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco_playground/blob/main/learning/notebooks/training_vision_2.ipynb)

## Training with brax PPO

To train with brax PPO, you can use the `train_jax_ppo.py` script. This script uses the brax PPO algorithm to train an agent on a given environment.

```bash
python train_jax_ppo.py --env_name=CartpoleBalance
```

To train a vision-based policy using pixel observations:
```bash
python train_jax_ppo.py --env_name=CartpoleBalance --vision
```

Use `python train_jax_ppo.py --help` to see possible options and usage. Logs and checkpoints are saved in `logs` directory.

## Training with RSL-RL

To train with RSL-RL, you can use the `train_rsl_rl.py` script. This script uses the RSL-RL algorithm to train an agent on a given environment.

```bash
python train_rsl_rl.py --env_name=LeapCubeReorient
```

To render the behaviour from the resulting policy:
```bash
python learning/train_rsl_rl.py --env_name LeapCubeReorient --play_only --load_run_name <run_name>
```

where `run_name` is the name of the run you want to load (will be printed in the console when the training run is started).

Logs and checkpoints are saved in `logs` directory.

## Training RobcoArm with SAC

For the RobcoArm environment, we provide a simple SAC (Soft Actor-Critic) implementation based on CleanRL:

```bash
# Train without wandb logging
python train_robco_sac.py

# Train with wandb logging
python train_robco_sac.py --track

# Train for more steps
python train_robco_sac.py --total-timesteps 500000
```

See [README_SAC.md](README_SAC.md) for more details and usage examples.
