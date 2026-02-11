import gymnasium as gym
import gym_battle_royale  # noqa: F401


def run_demo(steps: int = 200) -> None:
    env = gym.make("BattleRoyale2D-v0", enemy_count=18, loot_count=45)
    obs, info = env.reset(seed=7)
    total_reward = 0.0
    for _ in range(steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    print(env.render())
    print(f"total_reward={total_reward:.2f} info={info}")


if __name__ == "__main__":
    run_demo()
