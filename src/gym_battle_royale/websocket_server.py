from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from gym_battle_royale.env import BattleRoyale2DEnv


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
        "observation_radius": env.observation_radius,
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
            "current_weapon": env.player["weapon_slots"][env.player["current_slot"]],
            "current_mag": int(env.player["magazines"].get(env.player["weapon_slots"][env.player["current_slot"]], 0))
            if env.player["weapon_slots"][env.player["current_slot"]] is not None
            else 0,
            "reserve_ammo": int(env.player["reserve_ammo"].get(env.player["weapon_slots"][env.player["current_slot"]], 0))
            if env.player["weapon_slots"][env.player["current_slot"]] is not None
            else 0,
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

    player_pos = env.player["pos"]
    nearest = min(alive, key=lambda e: float(np.linalg.norm(e["pos"] - player_pos)))
    delta = nearest["pos"] - player_pos
    dist = float(np.linalg.norm(delta))

    if dist > 1e-5:
        aim = delta / dist
    else:
        aim = np.array([1.0, 0.0], dtype=np.float32)

    aim_x = int(np.clip(np.round(aim[0] * 5 + 5), 0, 10))
    aim_y = int(np.clip(np.round(aim[1] * 5 + 5), 0, 10))

    move_x = 0 if abs(delta[0]) < 1.5 else (1 if delta[0] > 0 else -1)
    move_y = 0 if abs(delta[1]) < 1.5 else (1 if delta[1] > 0 else -1)
    move_lookup = {
        (0, 0): 0,
        (0, -1): 1,
        (0, 1): 2,
        (-1, 0): 3,
        (1, 0): 4,
        (-1, -1): 5,
        (1, -1): 6,
        (-1, 1): 7,
        (1, 1): 8,
    }
    move = move_lookup[(move_x, move_y)]

    should_attack = int(dist <= 35.0)
    should_heal = int(env.player["health"] < 40 and env.player["medkits"] > 0)
    return [move, aim_x, aim_y, should_attack, 0, 0, should_heal, 0]


@app.websocket("/ws/game")
async def game_socket(websocket: WebSocket) -> None:
    await websocket.accept()

    env = BattleRoyale2DEnv(render_mode="ansi", enemy_count=1)
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
            _, reward, terminated, truncated, info = env.step(current_action.to_array())

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
