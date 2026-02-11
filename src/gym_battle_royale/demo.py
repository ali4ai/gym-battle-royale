from __future__ import annotations

import gymnasium as gym

import gym_battle_royale  # noqa: F401


def save_ppm(path: str, frame) -> None:
    h, w, _ = frame.shape
    with open(path, "wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
        f.write(frame.tobytes())


def run_demo(steps: int = 200, out_image: str = "battle_royale_frame.ppm") -> None:
    env = gym.make(
        "BattleRoyale2D-v0",
        enemy_count=18,
        loot_count=45,
        render_mode="rgb_array",
    )
    obs, info = env.reset(seed=7)
    total_reward = 0.0
    last_frame = None

    for _ in range(steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        last_frame = env.render()
        if terminated or truncated:
            break

    if last_frame is not None:
        save_ppm(out_image, last_frame)

    print(
        "step={step} kills={kills} health={health:.1f} enemies_left={alive_enemies} "
        "zone_radius={zone_radius:.1f}".format(**info)
    )
    print(f"total_reward={total_reward:.2f} saved_frame={out_image}")


if __name__ == "__main__":
    run_demo()
