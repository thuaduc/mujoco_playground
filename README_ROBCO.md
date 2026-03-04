# Robco Arm Environments

This module adds MuJoCo MJX environments for the **[RobCo Light Leo](https://www.rob.co/platform/robot/light-leo)** (`R005-A6-K3`) — a 6-DOF modular robotic arm with 1000 mm reach and 4.0 kg payload. All environments are JAX-compatible, fully jittable, and designed to be vectorized for massively parallel training with Brax (SAC or PPO).

## Robot Description

The **Light Leo** is a modular robot assembled in a 2-2-2 configuration:

| Module type | Module(s) |
|---|---|
| Base | 1× BD116-90 |
| Motors | 1× D116, 4× D86 |
| Links | 1× L116-500, 1× I86-400 |

**Key specs:**

| Property | Value |
|---|---|
| Degrees of freedom | 6 |
| Reach | 1000 mm |
| Payload | 4.0 kg |
| SKU | R005-A6-K3 |

The simulation uses the STL meshes of the BD116 proximal/distal segments (from `xmls/assets/`). In simulation the arm is driven by 6 revolute joints:

| Joint | Actuator name |
|---|---|
| `joint0_joint` | `joint0_joint_ctrl` |
| `joint1_joint` | `joint1_joint_ctrl` |
| `joint2_joint` | `joint2_joint_ctrl` |
| `joint3_joint` | `joint3_joint_ctrl` |
| `joint4_joint` | `joint4_joint_ctrl` |
| `joint5_joint` | `joint5_joint_ctrl` |

The end-effector is attached to the `joint5_distal` body.

3D meshes (STL) are stored in `xmls/assets/` and loaded at environment instantiation.

---

## File Structure

```
mujoco_playground/_src/robco/
├── base.py                    # RobcoArmBase: shared utilities and MjModel setup
├── robco_arm_position.py      # RobcoArmPosition: delta-position reaching
├── robco_position_hard.py     # RobcoPositionHard: reaching with randomized target
├── robco_arm_torque.py        # RobcoArmTorque: torque-controlled reaching
├── robco_arm_box.py           # RobcoArmBox: box pushing task
├── __init__.py                # Registry and env factory
└── xmls/
    ├── robco_arm_position.xml
    ├── robco_arm_torque.xml
    ├── robco_arm_box.xml
    └── assets/                # STL meshes for the BD116 arm
```

---

## Environments

### `RobcoArmPosition`

**File:** [robco_arm_position.py](robco_arm_position.py)  
**XML:** `xmls/robco_arm_position.xml` (timestep: 0.002 s, `implicitfast`)

Position-controlled reaching task. The agent outputs **joint position deltas**, which are clamped and accumulated to produce the target joint configuration. The target is a fixed body (`target`) placed in the scene.

**Observation:** joint positions + joint velocities + 3D target position (with small uniform noise).

**Action:** 6-dim delta joint position, scaled by `action_scale` (default: 4.7124 rad) and clamped per step by `max_action_delta` (default: 0.004 rad).

**Reward components:**

| Component | Default scale | Description |
|---|---|---|
| `end_effector_target` | 1.0 | Negative L2 distance from EE to target |
| `ground_collision` | 0.5 | Penalty per active ground-contact sensor |
| `self_collision` | 0.5 | Penalty per active self-contact sensor |
| `energy` | 0.0 | Penalty for joint power ($\|\dot{q}\| \cdot \|qfrc\|$) |
| `action_rate` | 0.01 | Penalty for large step-to-step action changes |

**Default config:**
```python
ctrl_dt = 0.01        # control timestep
sim_dt  = 0.001       # simulation timestep
episode_length = 500
success_distance_threshold = 0.05  # metres
```

---

### `RobcoPositionHard`

**File:** [robco_position_hard.py](robco_position_hard.py)  
**XML:** `xmls/robco_arm_position.xml` (same model as above)

Extends `RobcoArmPosition` with a **randomized target position** sampled uniformly inside a spherical shell at every episode reset. Suitable for learning generalizable reaching policies.

**Target sampling:**
- Direction: uniformly sampled on the unit sphere.
- Radius: uniformly sampled in `[target_min_norm, target_max_norm]` (default: 0.3 – 1.4 m).
- Supports passing a fixed `target_pos` at reset time for deterministic evaluation.

Reward components and observation are identical to `RobcoArmPosition`.

**Extra config parameters:**
```python
target_min_norm = 0.3   # minimum target distance from origin (metres)
target_max_norm = 1.4   # maximum target distance from origin (metres)
```

---

### `RobcoArmTorque`

**File:** [robco_arm_torque.py](robco_arm_torque.py)  
**XML:** `xmls/robco_arm_torque.xml` (timestep: 0.005 s, `implicitfast`)

Torque-controlled reaching task. The agent directly outputs joint torques (scaled by `action_scale`). Implements **action chunking** (`action_chunk_size`) for compatibility with chunk-based policies.

Compared to `RobcoArmPosition`, the reaching reward gives a **one-time binary bonus** when the target is first reached, then switches to a holding reward:

$$r_{\text{ee}} = \begin{cases} -d & \text{target not yet reached} \\ \text{bonus} & \text{first step reaching target} \\ \text{holding reward} & \text{target already reached} \end{cases}$$

A `target_reached` flag is tracked in `state.info`.

**Observation:** joint positions + joint velocities + 3D target position (with larger noise: ±0.001).

**Default config:**
```python
ctrl_dt = 0.01
sim_dt  = 0.001
episode_length = 300
action_scale   = 0.004
action_chunk_size = 1
success_distance_threshold = 0.05
```

**Reward scales (default):**

| Component | Default scale |
|---|---|
| `end_effector_target` | 1.0 |
| `ground_collision` | 0.5 |
| `self_collision` | 0.5 |
| `energy` | 0.0 |
| `action_rate` | 1.0 |

---

### `RobcoArmBox`

**File:** [robco_arm_box.py](robco_arm_box.py)  
**XML:** `xmls/robco_arm_box.xml` (timestep: 0.005 s)

Box-pushing / manipulation task. A **movable box** (`box_object`) must be pushed onto a fixed target marker (`target`). The reward is shaped to guide the arm towards the box, then the box towards the target, with velocity penalties for smooth interaction.

**Observation:** joint positions, joint velocities, end-effector position, box position, target position, relative vectors (EE→box, box→target).

**Reward components:**

| Component | Default scale | Description |
|---|---|---|
| `box_on_target` | 500.0 | Large sparse reward when box reaches target |
| `box_target_distance` | 10.0 | Dense reward for box–target proximity |
| `end_effector_box_distance` | 0.0 | Dense reward for EE–box proximity |
| `alignment` | 70.0 | Reward for collinear EE–box–target alignment |
| `end_effector_velocity` | 20.0 | Penalty for EE speed exceeding threshold |
| `box_velocity` | 0.0 | Penalty for box speed exceeding threshold |
| `ground_collision` | 1.0 | Ground contact penalty |
| `self_collision` | 1.0 | Self contact penalty |
| `energy` | 0.001 | Joint power penalty |
| `control_cost` | 0.001 | Penalty for large absolute actions |
| `action_rate` | 0.01 | Smoothness penalty |

**Key thresholds (default config):**
```python
box_on_target_distance_threshold = 0.1   # metres
alignment_distance_threshold     = 0.1   # metres
max_box_push_speed               = 0.1   # m/s
max_ee_speed                     = 0.05  # m/s
ctrl_dt = 0.05
episode_length = 400
```

---

## Collision Sensing

All environments use MuJoCo **contact sensors** registered in the XML to detect collisions without relying on expensive contact list parsing:

| Sensor | Type |
|---|---|
| `joint0_joint4_found` | Self-collision (links 0 ↔ 4) |
| `joint1_joint4_found` | Self-collision (links 1 ↔ 4) |
| `joint3_floor_found` | Ground collision (link 3) |
| `joint4_floor_found` | Ground collision (link 4) |

Collision costs are binary (`sensor > 0`) and summed.

---

## Registration & Usage

All environments are registered in [\_\_init\_\_.py](__init__.py) and accessible via `mujoco_playground.registry`:

```python
from mujoco_playground import registry

# List available Robco environments
print(registry.robco.ALL_ENVS)
# ('RobcoArmPosition', 'RobcoArmTorque', 'RobcoPositionHard', 'RobcoArmBox')

# Load an environment
env = registry.load("RobcoArmPosition")
print(f"Observation size: {env.observation_size}")
print(f"Action size:      {env.action_size}")

# JIT-compile and run
import jax
jit_reset = jax.jit(env.reset)
jit_step  = jax.jit(env.step)

state = jit_reset(jax.random.PRNGKey(0))
state = jit_step(state, jax.numpy.zeros(env.action_size))
```

Custom configs can be passed at instantiation:

```python
from mujoco_playground._src.robco.robco_arm_position import RobcoArmPosition, default_config

cfg = default_config()
cfg.reward_config.scales.action_rate = 0.05
cfg.episode_length = 1000
env = RobcoArmPosition(config=cfg)
```

---

## Training

Two training scripts are provided under `learning/`:

### SAC (recommended for sample efficiency)

```bash
conda activate playground

# Default (RobcoArmPosition, no wandb)
python learning/train_robco_brax_sac.py

# With wandb tracking and environment selection
python learning/train_robco_brax_sac.py \
    --track \
    --env-name RobcoPositionHard \
    --num-timesteps 2000000 \
    --num-envs 256
```

### PPO

```bash
conda activate playground

python learning/train_robco_brax_ppo.py \
    --track \
    --env-name RobcoArmPosition \
    --num-timesteps 5000000 \
    --num-envs 512
```

Both scripts support:
- `--env-name`: one of `RobcoArmPosition`, `RobcoArmTorque`, `RobcoPositionHard`, `RobcoArmBox`
- `--track`: enables wandb logging
- `--num-timesteps`, `--num-envs`, `--seed`

Trained checkpoints are saved to `checkpoints/` following the naming convention:  
`robco_brax_sac_<envname>_<seed>_<hash>/`

---

## Demo Notebook

[robco_environment_demo.ipynb](../../../robco_environment_demo.ipynb) demonstrates how to load an environment, run a sinusoidal open-loop rollout, render video frames, and plot reward components.
