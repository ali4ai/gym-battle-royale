from __future__ import annotations

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

import gym_battle_royale  # noqa: F401


def write_video(path: str, frames: list[np.ndarray], fps: int) -> None:
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise ImportError(
            "imageio is required to write demo videos. Install with `pip install -e .`."
        ) from exc

    with imageio.get_writer(path, fps=fps, codec="libx264", quality=7) as writer:
        for frame in frames:
            writer.append_data(frame)


def run_demo(
    steps: int = 5000,
    out_video: str = "battle_royale_random.mp4",
    fps: int = 15,
) -> None:
    env = gym.make(
        "BattleRoyale2D-v0",
        enemy_count=1,
        loot_count=60,
        render_mode="rgb_array",
        render_size=720,
    )
    obs, info = env.reset(seed=7)
    total_reward = 0.0
    frames: list[np.ndarray] = []
    model = PPO.load("runs/sb3_battle_royale_enemy_v2/best_model/best_model.zip")

    for _ in range(steps):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        frame = env.render()
        if isinstance(frame, np.ndarray):
            frames.append(frame)
        if terminated or truncated:
            break
        print(reward)
    env.close()

    if frames:
        write_video(out_video, frames, fps=fps)

    print(
        "step={step} kills={kills} health={health:.1f} enemies_left={alive_enemies} "
        "zone_radius={zone_radius:.1f}".format(**info)
    )
    print(f"total_reward={total_reward:.2f} frames={len(frames)} video={out_video}")


if __name__ == "__main__":
    run_demo()
