from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from gym_battle_royale.env import BattleRoyale2DEnv

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO


@dataclass
class ActionState:
    move: int = 0
    aim_x: int = 5
    aim_y: int = 5
    attack: int = 0
    interact: int = 0
    reload: int = 0
    heal: int = 0
    swap: int = 0

    def to_array(self) -> np.ndarray:
        return np.array(
            [
                self.move,
                self.aim_x,
                self.aim_y,
                self.attack,
                self.interact,
                self.reload,
                self.heal,
                self.swap,
            ],
            dtype=np.int64,
        )

    def update_from_payload(self, payload: dict[str, Any]) -> None:
        self.move = int(payload.get("move", self.move))
        self.aim_x = int(payload.get("aim_x", self.aim_x))
        self.aim_y = int(payload.get("aim_y", self.aim_y))
        self.attack = int(payload.get("attack", self.attack))
        self.interact = int(payload.get("interact", self.interact))
        self.reload = int(payload.get("reload", self.reload))
        self.heal = int(payload.get("heal", self.heal))
        self.swap = int(payload.get("swap", self.swap))


app = FastAPI(title="Gym Battle Royale WebSocket Bridge")
model = PPO.load("runs/sb3_battle_royale_enemy_v2/best_model/best_model.zip")

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _serialize_state(
    env: BattleRoyale2DEnv,
    reward: float,
    terminated: bool,
    truncated: bool,
    info: dict[str, Any],
    ai_action: list[int],
) -> dict[str, Any]:
    return {
        "step": env.step_count,
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
        "info": info,
        "map_size": env.map_size,
        "zone": {
            "center": [float(env.zone_center[0]), float(env.zone_center[1])],
            "radius": float(env.zone_radius),
        },
        "player": {
            "pos": [float(env.player["pos"][0]), float(env.player["pos"][1])],
            "aim": [float(env.player["aim"][0]), float(env.player["aim"][1])],
            "health": float(env.player["health"]),
            "armor": float(env.player["armor"]),
            "kills": int(env.player["kills"]),
            "medkits": int(env.player["medkits"]),
            "current_slot": int(env.player["current_slot"]),
            "weapon_slots": env.player["weapon_slots"],
        },
        "enemies": [
            {
                "pos": [float(enemy["pos"][0]), float(enemy["pos"][1])],
                "health": float(enemy["health"]),
                "alive": bool(enemy["alive"]),
            }
            for enemy in env.enemies
        ],
        "loot": [
            {
                "pos": [float(item["pos"][0]), float(item["pos"][1])],
                "kind": str(item["kind"]),
                "name": item.get("name"),
                "amount": int(item.get("amount", 0)),
            }
            for item in env.loot
        ],
        "ai_action": ai_action,
    }


def _compute_ai_action(env: BattleRoyale2DEnv) -> list[int]:
    alive = [enemy for enemy in env.enemies if enemy["alive"]]
    if not alive:
        return [0, 5, 5, 0, 0, 0, 0, 0]

    obs = env._build_obs()
    action, _states = model.predict(obs, deterministic=True)
    return action.tolist()


@app.websocket("/ws/game")
async def game_socket(websocket: WebSocket) -> None:
    await websocket.accept()

    env = BattleRoyale2DEnv(render_mode="ansi")
    _, info = env.reset()
    current_action = ActionState()
    reward = 0.0
    terminated = False
    truncated = False
    reset_requested = asyncio.Event()

    async def receiver() -> None:
        while True:
            message = await websocket.receive_json()
            msg_type = message.get("type")
            payload = message.get("payload", {})
            if msg_type == "player_action":
                current_action.update_from_payload(payload)
            elif msg_type == "reset":
                reset_requested.set()

    listener = asyncio.create_task(receiver())

    try:
        while True:
            if reset_requested.is_set() or terminated or truncated:
                _, info = env.reset()
                reward = 0.0
                terminated = False
                truncated = False
                reset_requested.clear()

            ai_action = _compute_ai_action(env)
            _, reward, terminated, truncated, info = env.step(np.array(ai_action))

            await websocket.send_json(
                {
                    "type": "game_state",
                    "payload": _serialize_state(env, reward, terminated, truncated, info, ai_action),
                }
            )
            await asyncio.sleep(1 / 15)
    except WebSocketDisconnect:
        pass
    finally:
        listener.cancel()
        env.close()
