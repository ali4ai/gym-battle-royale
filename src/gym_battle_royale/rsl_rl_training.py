from __future__ import annotations

import argparse
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch

import gym_battle_royale  # noqa: F401 - registers env


@dataclass
class TrainConfig:
    num_envs: int = 64
    enemy_count: int = 18
    loot_count: int = 50
    max_steps: int = 1800
    total_iterations: int = 1000
    steps_per_env: int = 24
    save_interval: int = 100
    seed: int = 1
    device: str = "cpu"
    log_dir: str = "runs/rsl_rl_battle_royale"


def flatten_obs(obs_dict: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate(
        [
            obs_dict["player"].reshape(-1),
            obs_dict["enemies"].reshape(-1),
            obs_dict["loot"].reshape(-1),
            obs_dict["zone"].reshape(-1),
        ],
        axis=0,
    ).astype(np.float32)


class RslVecBattleRoyaleEnv:
    """Minimal vectorized env wrapper compatible with rsl_rl OnPolicyRunner expectations."""

    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.num_envs = cfg.num_envs
        self.device = torch.device(cfg.device)

        self.envs = [
            gym.make(
                "BattleRoyale2D-v0",
                enemy_count=cfg.enemy_count,
                loot_count=cfg.loot_count,
                max_steps=cfg.max_steps,
                render_mode=None,
                seed=cfg.seed + i,
            )
            for i in range(cfg.num_envs)
        ]

        example_obs, _ = self.envs[0].reset(seed=cfg.seed)
        self.num_obs = int(flatten_obs(example_obs).shape[0])
        self.num_privileged_obs = None
        self.num_actions = int(np.sum(self.envs[0].action_space.nvec))

        self.max_episode_length = cfg.max_steps
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        self._action_dims = self.envs[0].action_space.nvec.astype(np.int64)
        self._action_slices: list[tuple[int, int]] = []
        start = 0
        for dim in self._action_dims:
            end = start + int(dim)
            self._action_slices.append((start, end))
            start = end

    def reset(self):
        obs_batch = []
        self.episode_length_buf.zero_()
        for i, env in enumerate(self.envs):
            obs, _ = env.reset(seed=self.cfg.seed + i)
            obs_batch.append(flatten_obs(obs))

        obs_tensor = torch.tensor(np.stack(obs_batch), device=self.device)
        return obs_tensor, None

    def step(self, policy_actions: torch.Tensor):
        # policy_actions expected as logits for each discrete dimension concatenated.
        actions_np = self._decode_actions(policy_actions)

        next_obs = []
        rewards = np.zeros((self.num_envs,), dtype=np.float32)
        dones = np.zeros((self.num_envs,), dtype=bool)

        infos: list[dict] = []
        for i, env in enumerate(self.envs):
            obs, rew, terminated, truncated, info = env.step(actions_np[i])
            done = bool(terminated or truncated)
            rewards[i] = rew
            dones[i] = done
            self.episode_length_buf[i] += 1

            if done:
                reset_obs, reset_info = env.reset()
                obs = reset_obs
                info = {**info, **{f"reset_{k}": v for k, v in reset_info.items()}}
                self.episode_length_buf[i] = 0

            next_obs.append(flatten_obs(obs))
            infos.append(info)

        obs_tensor = torch.tensor(np.stack(next_obs), device=self.device)
        reward_tensor = torch.tensor(rewards, device=self.device).unsqueeze(-1)
        done_tensor = torch.tensor(dones, device=self.device).unsqueeze(-1)

        # RSL-RL expects a dict with an optional "episode" field for logging.
        info_dict = {"raw_infos": infos}
        return obs_tensor, None, reward_tensor, done_tensor, info_dict

    def _decode_actions(self, policy_actions: torch.Tensor) -> np.ndarray:
        if policy_actions.ndim != 2:
            raise ValueError(
                f"Expected policy action tensor of shape [num_envs, action_logits], got {tuple(policy_actions.shape)}"
            )

        actions_np = np.zeros((self.num_envs, len(self._action_dims)), dtype=np.int64)
        act = policy_actions.detach().cpu().numpy()

        for idx, (s, e) in enumerate(self._action_slices):
            dim_logits = act[:, s:e]
            actions_np[:, idx] = np.argmax(dim_logits, axis=1)

        return actions_np

    def get_observations(self):
        obs, _ = self.reset()
        return obs

    def close(self):
        for env in self.envs:
            env.close()


def make_rsl_runner_cfg(env: RslVecBattleRoyaleEnv, cfg: TrainConfig) -> dict:
    return {
        "runner_class_name": "OnPolicyRunner",
        "seed": cfg.seed,
        "policy": {
            "class_name": "ActorCritic",
            "activation": "elu",
            "init_noise_std": 1.0,
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
        },
        "algorithm": {
            "class_name": "PPO",
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.01,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 1e-4,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
        },
        "runner": {
            "policy_class_name": "ActorCritic",
            "algorithm_class_name": "PPO",
            "num_steps_per_env": cfg.steps_per_env,
            "max_iterations": cfg.total_iterations,
            "save_interval": cfg.save_interval,
            "experiment_name": "battle_royale",
            "run_name": "rsl_rl_2d",
            "resume": False,
            "load_run": -1,
            "checkpoint": -1,
            "resume_path": None,
        },
    }


def train(cfg: TrainConfig) -> None:
    try:
        from rsl_rl.runners import OnPolicyRunner
    except ImportError as exc:
        raise ImportError(
            "rsl_rl is required for training. Install it first, e.g. `pip install rsl-rl`."
        ) from exc

    vec_env = RslVecBattleRoyaleEnv(cfg)
    runner_cfg = make_rsl_runner_cfg(vec_env, cfg)

    runner = OnPolicyRunner(vec_env, runner_cfg, log_dir=cfg.log_dir, device=cfg.device)
    runner.learn(num_learning_iterations=cfg.total_iterations, init_at_random_ep_len=True)

    vec_env.close()


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train BattleRoyale2D with RSL-RL PPO")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--enemy-count", type=int, default=18)
    parser.add_argument("--loot-count", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=1800)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--steps-per-env", type=int, default=24)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--log-dir", type=str, default="runs/rsl_rl_battle_royale")
    args = parser.parse_args()

    return TrainConfig(
        num_envs=args.num_envs,
        enemy_count=args.enemy_count,
        loot_count=args.loot_count,
        max_steps=args.max_steps,
        total_iterations=args.iterations,
        steps_per_env=args.steps_per_env,
        save_interval=args.save_interval,
        seed=args.seed,
        device=args.device,
        log_dir=args.log_dir,
    )


if __name__ == "__main__":
    train(parse_args())
