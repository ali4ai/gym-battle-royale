from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass
class Weapon:
    name: str
    damage: float
    range: float
    magazine_size: int
    reload_steps: int


WEAPONS: dict[str, Weapon] = {
    "pistol": Weapon("pistol", damage=11.0, range=24.0, magazine_size=12, reload_steps=8),
    "smg": Weapon("smg", damage=7.0, range=20.0, magazine_size=30, reload_steps=10),
    "ar": Weapon("ar", damage=12.0, range=35.0, magazine_size=30, reload_steps=12),
    "sniper": Weapon("sniper", damage=42.0, range=55.0, magazine_size=5, reload_steps=16),
}


class BattleRoyale2DEnv(gym.Env[dict[str, np.ndarray], np.ndarray]):
    """
    2D single-player battle royale environment inspired by PUBG/Suroi.

    Action encoding (MultiDiscrete):
    - 0: move direction (0-8): stay, N, S, W, E, NW, NE, SW, SE
    - 1: aim x delta bucket (0-10 -> -1.0..1.0)
    - 2: aim y delta bucket (0-10 -> -1.0..1.0)
    - 3: attack trigger (0/1)
    - 4: interact/loot trigger (0/1)
    - 5: reload trigger (0/1)
    - 6: use heal trigger (0/1)
    - 7: swap weapon slot (0=none, 1..4)
    """

    metadata = {"render_modes": ["ansi", "rgb_array"], "render_fps": 15}

    def __init__(
        self,
        map_size: float = 180.0,
        enemy_count: int = 23,
        loot_count: int = 60,
        max_steps: int = 2400,
        render_mode: str | None = None,
        render_size: int = 640,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if render_mode not in [None, "ansi", "rgb_array"]:
            raise ValueError(
                f"Unsupported render_mode={render_mode!r}. "
                f"Expected one of {self.metadata['render_modes']} or None."
            )

        self.map_size = map_size
        self.enemy_count = enemy_count
        self.loot_count = loot_count
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.render_size = max(200, int(render_size))
        self.rng = np.random.default_rng(seed)

        # move, aim_x, aim_y, attack, interact, reload, heal, switch weapon
        self.action_space = spaces.MultiDiscrete(np.array([9, 11, 11, 2, 2, 2, 2, 5]))

        self.max_enemies_tracked = max(enemy_count, 32)
        self.max_loot_tracked = max(loot_count, 80)

        self.observation_space = spaces.Dict(
            {
                "player": spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32),
                "enemies": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(self.max_enemies_tracked, 6),
                    dtype=np.float32,
                ),
                "loot": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(self.max_loot_tracked, 5),
                    dtype=np.float32,
                ),
                "zone": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
            }
        )

        self.player: dict[str, Any] = {}
        self.enemies: list[dict[str, Any]] = []
        self.loot: list[dict[str, Any]] = []
        self.zone_center = np.zeros(2, dtype=np.float32)
        self.zone_radius = 0.0
        self.step_count = 0
        self.reloading_until = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.step_count = 0
        self.zone_center = np.array([self.map_size / 2, self.map_size / 2], dtype=np.float32)
        self.zone_radius = self.map_size * 0.48
        self.reloading_until = 0

        spawn = self._random_point_in_circle(self.zone_center, self.zone_radius * 0.5)
        self.player = {
            "pos": spawn,
            "aim": np.array([1.0, 0.0], dtype=np.float32),
            "health": 100.0,
            "armor": 0.0,
            "weapon_slots": ["pistol", "smg", None, None],
            "current_slot": 0,
            "magazines": {"pistol": 12, "smg": 30, "ar": 0, "sniper": 0},
            "reserve_ammo": {"pistol": 36, "smg": 90, "ar": 0, "sniper": 0},
            "medkits": 2,
            "kills": 0,
        }

        self.enemies = []
        for _ in range(self.enemy_count):
            enemy_weapon = self.rng.choice(["pistol", "smg", "ar"])
            self.enemies.append(
                {
                    "pos": self._random_point_in_circle(self.zone_center, self.zone_radius * 0.95),
                    "health": 100.0,
                    "armor": float(self.rng.choice([0.0, 15.0, 30.0])),
                    "weapon": enemy_weapon,
                    "mag": WEAPONS[enemy_weapon].magazine_size,
                    "alive": True,
                }
            )

        self.loot = []
        for _ in range(self.loot_count):
            self.loot.append(self._spawn_loot())

        return self._build_obs(), self._info()

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        self.step_count += 1
        reward = 0.08
        terminated = False
        truncated = False

        move, aim_x, aim_y, attack, interact, reload_trigger, heal_trigger, swap = map(
            int, action
        )

        self._apply_movement(move)
        self._apply_aim(aim_x, aim_y)

        if swap in [1, 2, 3, 4]:
            self.player["current_slot"] = swap - 1

        if reload_trigger and self.step_count >= self.reloading_until:
            self._reload_current_weapon()

        if heal_trigger and self.player["medkits"] > 0 and self.player["health"] < 95:
            self.player["medkits"] -= 1
            self.player["health"] = min(100.0, self.player["health"] + 45.0)
            reward += 0.6

        if attack and self.step_count >= self.reloading_until:
            reward += self._player_attack()

        if interact:
            reward += self._loot_interact()

        reward += self._enemy_ai_and_attacks()
        reward += self._apply_zone_logic()

        alive_enemies = sum(1 for e in self.enemies if e["alive"])
        if self.player["health"] <= 0:
            terminated = True
            reward -= 20.0
        elif alive_enemies == 0:
            terminated = True
            reward += 50.0

        if self.step_count >= self.max_steps:
            truncated = True

        return self._build_obs(), reward, terminated, truncated, self._info()

    def render(self) -> str | np.ndarray:
        if self.render_mode == "rgb_array":
            return self._render_rgb_array()

        alive_enemies = sum(1 for e in self.enemies if e["alive"])
        pos = self.player["pos"]
        return (
            f"Step={self.step_count} pos=({pos[0]:.1f},{pos[1]:.1f}) "
            f"hp={self.player['health']:.1f} armor={self.player['armor']:.1f} "
            f"kills={self.player['kills']} enemies_left={alive_enemies} "
            f"zone_radius={self.zone_radius:.1f}"
        )

    def _apply_movement(self, move: int) -> None:
        directions = {
            0: np.array([0.0, 0.0]),
            1: np.array([0.0, -1.0]),
            2: np.array([0.0, 1.0]),
            3: np.array([-1.0, 0.0]),
            4: np.array([1.0, 0.0]),
            5: np.array([-1.0, -1.0]),
            6: np.array([1.0, -1.0]),
            7: np.array([-1.0, 1.0]),
            8: np.array([1.0, 1.0]),
        }
        step_vec = directions[move]
        if np.linalg.norm(step_vec) > 0:
            step_vec = step_vec / np.linalg.norm(step_vec)
        self.player["pos"] = np.clip(self.player["pos"] + step_vec * 1.7, 0.0, self.map_size)

    def _apply_aim(self, aim_x: int, aim_y: int) -> None:
        delta = np.array([(aim_x - 5) / 5.0, (aim_y - 5) / 5.0], dtype=np.float32)
        if np.linalg.norm(delta) > 0.1:
            self.player["aim"] = delta / np.linalg.norm(delta)

    def _player_attack(self) -> float:
        slot = self.player["current_slot"]
        weapon_name = self.player["weapon_slots"][slot]
        if weapon_name is None:
            return -0.05
        if self.player["magazines"][weapon_name] <= 0:
            return -0.05

        weapon = WEAPONS[weapon_name]
        self.player["magazines"][weapon_name] -= 1

        reward = 0.0
        for enemy in self.enemies:
            if not enemy["alive"]:
                continue
            to_enemy = enemy["pos"] - self.player["pos"]
            dist = float(np.linalg.norm(to_enemy))
            if dist > weapon.range or dist < 1e-6:
                continue
            ray = to_enemy / dist
            alignment = float(np.dot(ray, self.player["aim"]))
            if alignment < 0.93:
                continue

            spread = self.rng.uniform(0.7, 1.0)
            damage = weapon.damage * spread
            mitigated = max(0.0, damage - enemy["armor"] * 0.08)
            enemy["health"] -= mitigated
            reward += mitigated * 0.04
            if enemy["health"] <= 0:
                enemy["alive"] = False
                self.player["kills"] += 1
                reward += 10.0
                self.loot.append(
                    {
                        "pos": enemy["pos"].copy(),
                        "kind": self.rng.choice(["ammo", "medkit", "armor", "weapon"]),
                        "value": self.rng.choice([1.0, 2.0, 3.0]),
                    }
                )
            break
        return reward

    def _loot_interact(self) -> float:
        if not self.loot:
            return 0.0

        best_idx = -1
        best_dist = 9e9
        for i, item in enumerate(self.loot):
            d = float(np.linalg.norm(item["pos"] - self.player["pos"]))
            if d < best_dist:
                best_dist = d
                best_idx = i

        if best_idx < 0 or best_dist > 4.0:
            return -0.02

        item = self.loot.pop(best_idx)
        kind = item["kind"]
        value = item["value"]

        if kind == "ammo":
            for w in self.player["reserve_ammo"]:
                self.player["reserve_ammo"][w] += int(10 * value)
        elif kind == "medkit":
            self.player["medkits"] += int(value)
        elif kind == "armor":
            self.player["armor"] = min(100.0, self.player["armor"] + 15 * value)
        elif kind == "weapon":
            candidate = str(self.rng.choice(list(WEAPONS.keys())))
            insert_idx = next(
                (i for i, w in enumerate(self.player["weapon_slots"]) if w is None),
                self.rng.integers(0, 4),
            )
            self.player["weapon_slots"][insert_idx] = candidate
            if self.player["magazines"][candidate] == 0:
                self.player["magazines"][candidate] = WEAPONS[candidate].magazine_size
                self.player["reserve_ammo"][candidate] += WEAPONS[candidate].magazine_size

        return 1.0

    def _enemy_ai_and_attacks(self) -> float:
        reward = 0.0
        for enemy in self.enemies:
            if not enemy["alive"]:
                continue
            to_player = self.player["pos"] - enemy["pos"]
            dist = float(np.linalg.norm(to_player))
            if dist > 1e-6:
                heading = to_player / dist
            else:
                heading = np.zeros(2, dtype=np.float32)

            # Simple behavior: close in when far, strafe when close.
            if dist > 16.0:
                enemy["pos"] += heading * 1.1
            else:
                perp = np.array([-heading[1], heading[0]])
                enemy["pos"] += (0.6 * perp + 0.2 * heading) * self.rng.choice([-1.0, 1.0])

            enemy["pos"] = np.clip(enemy["pos"], 0.0, self.map_size)

            weapon = WEAPONS[enemy["weapon"]]
            if dist < weapon.range and enemy["mag"] > 0:
                hit_prob = 0.20 + 0.5 * max(0.0, 1.0 - dist / weapon.range)
                if self.rng.random() < hit_prob:
                    raw = weapon.damage * self.rng.uniform(0.55, 0.95)
                    absorbed = min(self.player["armor"], raw * 0.45)
                    self.player["armor"] -= absorbed
                    self.player["health"] -= raw - absorbed
                    reward -= (raw - absorbed) * 0.08
                enemy["mag"] -= 1
            elif enemy["mag"] <= 0 and self.rng.random() < 0.3:
                enemy["mag"] = weapon.magazine_size

        return reward

    def _apply_zone_logic(self) -> float:
        # Shrink zone in phases, with increasing gas damage.
        phase = self.step_count // 280
        target_radius = max(12.0, self.map_size * (0.48 - 0.06 * phase))
        self.zone_radius = max(target_radius, self.zone_radius - 0.12)

        damage = 0.0
        d = float(np.linalg.norm(self.player["pos"] - self.zone_center))
        if d > self.zone_radius:
            gas_phase_scale = 1.0 + 0.5 * phase
            gas_damage = 0.35 * gas_phase_scale
            self.player["health"] -= gas_damage
            damage += gas_damage

        return -damage * 0.6

    def _reload_current_weapon(self) -> None:
        slot = self.player["current_slot"]
        weapon_name = self.player["weapon_slots"][slot]
        if weapon_name is None:
            return

        weapon = WEAPONS[weapon_name]
        current = self.player["magazines"][weapon_name]
        missing = weapon.magazine_size - current
        if missing <= 0:
            return

        available = self.player["reserve_ammo"][weapon_name]
        if available <= 0:
            return

        to_load = min(missing, available)
        self.player["magazines"][weapon_name] += to_load
        self.player["reserve_ammo"][weapon_name] -= to_load
        self.reloading_until = self.step_count + weapon.reload_steps

    def _spawn_loot(self) -> dict[str, Any]:
        return {
            "pos": self._random_point_in_circle(self.zone_center, self.zone_radius * 0.98),
            "kind": self.rng.choice(["ammo", "medkit", "armor", "weapon"], p=[0.45, 0.2, 0.2, 0.15]),
            "value": float(self.rng.choice([1.0, 1.0, 2.0, 3.0])),
        }

    def _random_point_in_circle(self, center: np.ndarray, radius: float) -> np.ndarray:
        angle = self.rng.uniform(0, 2 * np.pi)
        rad = radius * np.sqrt(self.rng.uniform(0, 1))
        point = center + np.array([np.cos(angle), np.sin(angle)]) * rad
        return np.clip(point.astype(np.float32), 0.0, self.map_size)

    def _build_obs(self) -> dict[str, np.ndarray]:
        player = np.zeros(12, dtype=np.float32)
        player[0:2] = self.player["pos"] / self.map_size * 2 - 1
        player[2:4] = self.player["aim"]
        player[4] = self.player["health"] / 100 * 2 - 1
        player[5] = self.player["armor"] / 100 * 2 - 1
        player[6] = self.player["medkits"] / 10 * 2 - 1
        player[7] = self.player["kills"] / max(1, self.enemy_count) * 2 - 1
        player[8] = self.player["current_slot"] / 3 * 2 - 1
        total_ammo = sum(self.player["reserve_ammo"].values())
        player[9] = min(1.0, total_ammo / 400) * 2 - 1
        current_weapon = self.player["weapon_slots"][self.player["current_slot"]]
        if current_weapon:
            mag = self.player["magazines"][current_weapon]
            player[10] = mag / WEAPONS[current_weapon].magazine_size * 2 - 1
            player[11] = self.player["reserve_ammo"][current_weapon] / 120 * 2 - 1

        enemies = np.zeros((self.max_enemies_tracked, 6), dtype=np.float32)
        for i, enemy in enumerate(self.enemies[: self.max_enemies_tracked]):
            enemies[i, 0:2] = enemy["pos"] / self.map_size * 2 - 1
            enemies[i, 2] = enemy["health"] / 100 * 2 - 1
            enemies[i, 3] = enemy["armor"] / 100 * 2 - 1
            enemies[i, 4] = 1.0 if enemy["alive"] else -1.0
            enemies[i, 5] = min(1.0, np.linalg.norm(enemy["pos"] - self.player["pos"]) / 40) * 2 - 1

        loot = np.zeros((self.max_loot_tracked, 5), dtype=np.float32)
        kind_map = {"ammo": -1.0, "medkit": -0.3, "armor": 0.3, "weapon": 1.0}
        for i, item in enumerate(self.loot[: self.max_loot_tracked]):
            loot[i, 0:2] = item["pos"] / self.map_size * 2 - 1
            loot[i, 2] = kind_map[str(item["kind"])]
            loot[i, 3] = item["value"] / 3.0 * 2 - 1
            loot[i, 4] = min(1.0, np.linalg.norm(item["pos"] - self.player["pos"]) / 35) * 2 - 1

        zone = np.zeros(4, dtype=np.float32)
        zone[0:2] = self.zone_center / self.map_size * 2 - 1
        zone[2] = self.zone_radius / (self.map_size * 0.5) * 2 - 1
        zone_dist = np.linalg.norm(self.player["pos"] - self.zone_center)
        zone[3] = min(1.0, zone_dist / max(1.0, self.zone_radius)) * 2 - 1

        return {"player": player, "enemies": enemies, "loot": loot, "zone": zone}

    def _info(self) -> dict[str, Any]:
        return {
            "step": self.step_count,
            "kills": self.player["kills"],
            "health": self.player["health"],
            "alive_enemies": sum(1 for e in self.enemies if e["alive"]),
            "zone_radius": self.zone_radius,
        }

    def _to_canvas_point(self, world_point: np.ndarray) -> tuple[int, int]:
        px = int(np.clip((world_point[0] / self.map_size) * (self.render_size - 1), 0, self.render_size - 1))
        py = int(np.clip((world_point[1] / self.map_size) * (self.render_size - 1), 0, self.render_size - 1))
        return px, py

    def _draw_disk(self, canvas: np.ndarray, center: tuple[int, int], radius: int, color: tuple[int, int, int]) -> None:
        cx, cy = center
        x0, x1 = max(0, cx - radius), min(canvas.shape[1] - 1, cx + radius)
        y0, y1 = max(0, cy - radius), min(canvas.shape[0] - 1, cy + radius)
        for y in range(y0, y1 + 1):
            dy = y - cy
            for x in range(x0, x1 + 1):
                dx = x - cx
                if dx * dx + dy * dy <= radius * radius:
                    canvas[y, x] = color

    def _draw_line(self, canvas: np.ndarray, start: tuple[int, int], end: tuple[int, int], color: tuple[int, int, int]) -> None:
        x0, y0 = start
        x1, y1 = end
        steps = max(abs(x1 - x0), abs(y1 - y0), 1)
        for i in range(steps + 1):
            t = i / steps
            x = int(round(x0 + (x1 - x0) * t))
            y = int(round(y0 + (y1 - y0) * t))
            if 0 <= x < canvas.shape[1] and 0 <= y < canvas.shape[0]:
                canvas[y, x] = color

    def _render_rgb_array(self) -> np.ndarray:
        canvas = np.zeros((self.render_size, self.render_size, 3), dtype=np.uint8)
        canvas[:] = np.array([46, 112, 64], dtype=np.uint8)  # grass

        zone_center_px = self._to_canvas_point(self.zone_center)
        zone_radius_px = max(1, int((self.zone_radius / self.map_size) * self.render_size))

        # Gas overlay outside the safe zone.
        yy, xx = np.ogrid[: self.render_size, : self.render_size]
        dist2 = (xx - zone_center_px[0]) ** 2 + (yy - zone_center_px[1]) ** 2
        gas_mask = dist2 > zone_radius_px * zone_radius_px
        canvas[gas_mask] = (canvas[gas_mask] * 0.55 + np.array([120, 90, 40], dtype=np.float32) * 0.45).astype(np.uint8)

        # Zone ring.
        ring_mask = (dist2 > (zone_radius_px - 2) ** 2) & (dist2 < (zone_radius_px + 2) ** 2)
        canvas[ring_mask] = np.array([230, 214, 82], dtype=np.uint8)

        loot_colors = {
            "ammo": (63, 158, 255),
            "medkit": (250, 77, 87),
            "armor": (133, 97, 255),
            "weapon": (235, 235, 235),
        }
        for item in self.loot:
            p = self._to_canvas_point(item["pos"])
            self._draw_disk(canvas, p, radius=2, color=loot_colors[str(item["kind"])])

        for enemy in self.enemies:
            if not enemy["alive"]:
                continue
            p = self._to_canvas_point(enemy["pos"])
            self._draw_disk(canvas, p, radius=4, color=(220, 62, 62))

        player_pos = self._to_canvas_point(self.player["pos"])
        self._draw_disk(canvas, player_pos, radius=5, color=(72, 167, 255))

        aim_end_world = self.player["pos"] + self.player["aim"] * 10.0
        aim_end = self._to_canvas_point(aim_end_world)
        self._draw_line(canvas, player_pos, aim_end, color=(255, 255, 255))

        # Health and armor bars (HUD).
        hud_y = 10
        bar_w = self.render_size // 4
        hp_w = int(bar_w * (max(0.0, self.player["health"]) / 100.0))
        armor_w = int(bar_w * (max(0.0, self.player["armor"]) / 100.0))
        canvas[hud_y : hud_y + 8, 10 : 10 + bar_w] = (45, 45, 45)
        canvas[hud_y : hud_y + 8, 10 : 10 + hp_w] = (58, 216, 88)
        canvas[hud_y + 12 : hud_y + 20, 10 : 10 + bar_w] = (45, 45, 45)
        canvas[hud_y + 12 : hud_y + 20, 10 : 10 + armor_w] = (85, 169, 255)

        return canvas
