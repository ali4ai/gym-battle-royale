# Gym Battle Royale (PUBG-like 2D Environment)

This repository provides a **single-player Gymnasium environment** that mimics the high-level battle royale loop from PUBG/Suroi-style games:

- Drop/spawn into a 2D map
- Loot weapons, ammo, armor, and healing items
- Fight AI enemies
- Survive a shrinking gas zone
- Win by being the last one alive

## Features

- **2D top-down combat** with movement + aim + firing
- **Inventory loop** (weapon slots, magazines, reserve ammo, medkits)
- **Loot interaction** (ammo/armor/medkits/weapons)
- **Enemy bots** with simple chase/strafe/shoot behavior
- **Shrinking safe zone** with increasing gas damage
- **Pygame-based visual rendering** with icons for loot/player/enemies
- **Fog-of-war style observation**: nearby area is clear, far area is blurred/darkened
- Gymnasium-compatible API (`reset`, `step`, observation/action spaces)

## Install

```bash
pip install -e .
```

## Quick start

```python
import gymnasium as gym
import gym_battle_royale  # registers env

env = gym.make(
    "BattleRoyale2D-v0",
    enemy_count=30,
    render_mode="rgb_array",
    render_size=720,
)
obs, info = env.reset(seed=42)

for _ in range(500):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    frame = env.render()  # np.ndarray (H, W, 3), dtype=uint8
    if terminated or truncated:
        break

print(info)
```

## Render modes

- `render_mode="ansi"` (default behavior): returns compact text state.
- `render_mode="rgb_array"`: returns a top-down RGB frame.
- `render_mode="human"`: opens a pygame window and displays frames in real time.

### What is drawn

- Player and aim line
- Enemies with health rings
- Loot icons by type (ammo, medkit, armor, weapon)
- Safe zone ring and gas overlay outside zone
- HP and armor HUD bars
- Observation spotlight (clear) with blurry/dark outer area


## Training with RSL-RL

A ready-to-use training entrypoint is included:

```bash
PYTHONPATH=src python -m gym_battle_royale.rsl_rl_training   --num-envs 64   --iterations 1000   --steps-per-env 24   --device cpu
```

What this script provides:

- Vectorized wrapper (`RslVecBattleRoyaleEnv`) over multiple Gym env instances
- Flattened observations for actor-critic networks
- Multi-discrete action decoding from concatenated logits
- PPO runner config for RSL-RL (`OnPolicyRunner`)

Notes:

- Requires `rsl_rl` and `torch` installed in your environment.
- Outputs logs/checkpoints to `runs/rsl_rl_battle_royale` by default.

## Demo: random-action video

Generate a random-policy gameplay video:

```bash
PYTHONPATH=src python -m gym_battle_royale.demo
```

This writes `battle_royale_random.mp4` in the project root.

## Action space

The environment uses:

```text
MultiDiscrete([9, 11, 11, 2, 2, 2, 2, 5])
```

- `0`: movement (stay / 8 directions)
- `1`: aim x bucket (maps to -1..1)
- `2`: aim y bucket (maps to -1..1)
- `3`: attack trigger
- `4`: interact/loot trigger
- `5`: reload trigger
- `6`: heal trigger
- `7`: switch weapon slot (none, slots 1..4)

This condenses mouse+keyboard control complexity into an RL-friendly discrete format while preserving the same gameplay ideas.

## Observation space

`Dict` with:

- `player`: normalized vector with position, aim, health, armor, ammo, etc.
- `enemies`: padded enemy table with per-enemy position, health, alive flag, distance
- `loot`: padded loot table with position, item type/value, distance
- `zone`: zone center/radius + normalized player distance to zone center

## Mapping from requested controls

Your requested keybind list includes many actions (WASD, mouse aim, attack, reload, heal, interact, swap guns, map toggles, emote wheel, pings, etc.).

This environment implements the **core gameplay controls relevant to reinforcement learning**:

- Movement
- Aim direction
- Attack
- Reload
- Interact/loot
- Heal
- Weapon-slot switch

UI-only actions (fullscreen map, minimap toggle, emotes, pings, settings) are intentionally omitted, since they do not affect core survival dynamics for a single-player training environment.

## Notes

- Designed for **single-player** training/evaluation.
- Visual output is generated as NumPy RGB frames and can also be displayed via pygame window mode.
- You can extend this base with richer bot behavior, recoil, ballistics, and terrain/obstacles.
