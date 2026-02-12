from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np

import gym_battle_royale  # noqa: F401 - registers env


@dataclass
class TrainConfig:
    num_envs: int = 8
    enemy_count: int = 1
    loot_count: int = 50
    max_steps: int = 2500
    total_timesteps: int = 1_000_000
    eval_episodes: int = 5
    seed: int = 1
    device: str = "cuda"
    log_dir: str = "runs/sb3_battle_royale"
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 256
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    clip_range: float = 0.2


def make_env(cfg: TrainConfig, rank: int):
    def _thunk():
        env = gym.make(
            "BattleRoyale2D-v0",
            enemy_count=cfg.enemy_count,
            loot_count=cfg.loot_count,
            max_steps=cfg.max_steps,
            render_mode=None,
            seed=cfg.seed + rank,
        )
        return env

    return _thunk


def train(cfg: TrainConfig) -> Path:
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError as exc:
        raise ImportError(
            "stable-baselines3 is required for training. Install with `pip install 'gym-battle-royale[train]'`."
        ) from exc

    log_dir = Path(cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    env = DummyVecEnv([make_env(cfg, i) for i in range(cfg.num_envs)])

    eval_env = DummyVecEnv([make_env(cfg, 10_000)])

    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        verbose=1,
        tensorboard_log=str(log_dir / "tb"),
        learning_rate=cfg.learning_rate,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        ent_coef=cfg.ent_coef,
        clip_range=cfg.clip_range,
        device=cfg.device,
        seed=cfg.seed,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(10_000 // cfg.num_envs, 1),
        save_path=str(log_dir / "checkpoints"),
        name_prefix="ppo_battle_royale",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(log_dir / "best_model"),
        log_path=str(log_dir / "eval"),
        eval_freq=max(20_000 // cfg.num_envs, 1),
        n_eval_episodes=cfg.eval_episodes,
        deterministic=True,
        render=False,
    )

    model.learn(total_timesteps=cfg.total_timesteps, callback=[checkpoint_cb, eval_cb])

    final_model_path = log_dir / "ppo_battle_royale_final"
    model.save(str(final_model_path))

    env.close()
    eval_env.close()

    return final_model_path


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train BattleRoyale2D with Stable-Baselines3 PPO")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--enemy-count", type=int, default=1)
    parser.add_argument("--loot-count", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=2500)
    parser.add_argument("--timesteps", type=int, default=5_000_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log-dir", type=str, default="runs/sb3_battle_royale_enemy_v2")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--clip-range", type=float, default=0.2)
    args = parser.parse_args()

    return TrainConfig(
        num_envs=args.num_envs,
        enemy_count=args.enemy_count,
        loot_count=args.loot_count,
        max_steps=args.max_steps,
        total_timesteps=args.timesteps,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
        device=args.device,
        log_dir=args.log_dir,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        clip_range=args.clip_range,
    )


if __name__ == "__main__":
    cfg = parse_args()
    path = train(cfg)
    print(f"Training complete. Saved final model to: {path}")
