from gymnasium.envs.registration import register

from gym_battle_royale.env import BattleRoyale2DEnv

register(
    id="BattleRoyale2D-v0",
    entry_point="gym_battle_royale.env:BattleRoyale2DEnv",
)

__all__ = ["BattleRoyale2DEnv"]
