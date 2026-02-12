import './styles.css';

type Vec2 = [number, number];

type GameState = {
  step: number;
  reward: number;
  terminated: boolean;
  truncated: boolean;
  map_size: number;
  observation_radius: number;
  zone: { center: Vec2; radius: number };
  player: {
    pos: Vec2;
    aim: Vec2;
    health: number;
    armor: number;
    kills: number;
    medkits: number;
    current_slot: number;
    weapon_slots: Array<string | null>;
    current_weapon: string | null;
    current_mag: number;
    reserve_ammo: number;
  };
  enemies: Array<{ pos: Vec2; health: number; alive: boolean }>;
  loot: Array<{ pos: Vec2; kind: string; name?: string; amount?: number }>;
  ai_action: number[];
};

type LootAnimation = {
  x: number;
  y: number;
  ttl: number;
  kind: string;
  by: 'agent' | 'enemy';
};

const canvas = document.querySelector<HTMLCanvasElement>('#game');
const statusEl = document.querySelector<HTMLParagraphElement>('#status');
const playerStatsEl = document.querySelector<HTMLParagraphElement>('#player-stats');
const weaponStatsEl = document.querySelector<HTMLParagraphElement>('#weapon-stats');
const enemyStatsEl = document.querySelector<HTMLParagraphElement>('#enemy-stats');
const aiActionEl = document.querySelector<HTMLParagraphElement>('#ai-action');
const resetBtn = document.querySelector<HTMLButtonElement>('#reset');
const agentHealthEl = document.querySelector<HTMLDivElement>('#agent-health');
const agentArmorEl = document.querySelector<HTMLDivElement>('#agent-armor');
const enemyHealthEl = document.querySelector<HTMLDivElement>('#enemy-health');

if (
  !canvas ||
  !statusEl ||
  !playerStatsEl ||
  !weaponStatsEl ||
  !enemyStatsEl ||
  !aiActionEl ||
  !resetBtn ||
  !agentHealthEl ||
  !agentArmorEl ||
  !enemyHealthEl
) {
  throw new Error('Missing DOM elements');
}

const ctx = canvas.getContext('2d');
if (!ctx) {
  throw new Error('Cannot get canvas context');
}

const ws = new WebSocket('ws://localhost:8000/ws/game');
let latestState: GameState | null = null;
let lootAnimations: LootAnimation[] = [];
let prevLootMap = new Map<string, { pos: Vec2; kind: string }>();

const exploredMask = document.createElement('canvas');
exploredMask.width = canvas.width;
exploredMask.height = canvas.height;
const exploredCtx = exploredMask.getContext('2d');
if (!exploredCtx) {
  throw new Error('Cannot get explored mask context');
}
exploredCtx.fillStyle = 'black';
exploredCtx.fillRect(0, 0, canvas.width, canvas.height);

const keys = new Set<string>();
let mouseAim: Vec2 = [5, 5];

const toCanvas = (pos: Vec2, mapSize: number): Vec2 => [
  (pos[0] / mapSize) * canvas.width,
  (pos[1] / mapSize) * canvas.height,
];

const lootKey = (item: { pos: Vec2; kind: string; name?: string; amount?: number }): string =>
  `${item.kind}:${item.name ?? 'na'}:${item.amount ?? 0}:${Math.round(item.pos[0] * 10)}:${Math.round(item.pos[1] * 10)}`;

const lootIcon = (kind: string): string => {
  if (kind === 'weapon') return '🔫';
  if (kind === 'medkit') return '🩹';
  if (kind === 'armor') return '🛡️';
  if (kind === 'ammo') return '📦';
  return '✦';
};

const drawBadgeIcon = (x: number, y: number, bg: string, icon: string, size = 18) => {
  ctx.fillStyle = bg;
  ctx.beginPath();
  ctx.arc(x, y, size / 2, 0, Math.PI * 2);
  ctx.fill();

  ctx.font = '14px "Segoe UI Emoji", "Apple Color Emoji", sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillStyle = '#ffffff';
  ctx.fillText(icon, x, y + 0.5);
};

const updateLootAnimations = (state: GameState) => {
  const nextMap = new Map<string, { pos: Vec2; kind: string }>();
  for (const item of state.loot) {
    nextMap.set(lootKey(item), { pos: item.pos, kind: item.kind });
  }

  for (const [key, removed] of prevLootMap.entries()) {
    if (nextMap.has(key)) continue;

    const dPlayer = Math.hypot(removed.pos[0] - state.player.pos[0], removed.pos[1] - state.player.pos[1]);
    const aliveEnemy = state.enemies.find((enemy) => enemy.alive);
    const dEnemy = aliveEnemy
      ? Math.hypot(removed.pos[0] - aliveEnemy.pos[0], removed.pos[1] - aliveEnemy.pos[1])
      : Number.POSITIVE_INFINITY;

    if (Math.min(dPlayer, dEnemy) > 8.0) continue;

    const [cx, cy] = toCanvas(removed.pos, state.map_size);
    lootAnimations.push({
      x: cx,
      y: cy,
      ttl: 1,
      kind: removed.kind,
      by: dPlayer <= dEnemy ? 'agent' : 'enemy',
    });
  }

  prevLootMap = nextMap;
};

const drawWorld = (state: GameState) => {
  ctx.fillStyle = '#0f172a';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  const zoneCenter = toCanvas(state.zone.center, state.map_size);
  const zoneRadius = (state.zone.radius / state.map_size) * canvas.width;
  ctx.strokeStyle = '#facc15';
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.arc(zoneCenter[0], zoneCenter[1], zoneRadius, 0, Math.PI * 2);
  ctx.stroke();

  for (const item of state.loot) {
    const [x, y] = toCanvas(item.pos, state.map_size);
    const bg = item.kind === 'weapon' ? '#8b5cf6' : item.kind === 'medkit' ? '#ef4444' : item.kind === 'armor' ? '#0ea5e9' : '#f59e0b';
    drawBadgeIcon(x, y, bg, lootIcon(item.kind), 16);
  }

  for (const enemy of state.enemies) {
    if (!enemy.alive) continue;
    const [x, y] = toCanvas(enemy.pos, state.map_size);
    drawBadgeIcon(x, y, '#f97316', '☠', 22);

    const hp = Math.max(0, Math.min(1, enemy.health / 100));
    ctx.fillStyle = '#0b1220';
    ctx.fillRect(x - 20, y - 20, 40, 6);
    ctx.fillStyle = '#ef4444';
    ctx.fillRect(x - 20, y - 20, 40 * hp, 6);
  }

  const [px, py] = toCanvas(state.player.pos, state.map_size);
  drawBadgeIcon(px, py, '#0ea5e9', '🧍', 24);

  const aimEnd: Vec2 = [px + state.player.aim[0] * 28, py + state.player.aim[1] * 28];
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(px, py);
  ctx.lineTo(aimEnd[0], aimEnd[1]);
  ctx.stroke();
};

const drawLootAnimations = () => {
  const nextAnimations: LootAnimation[] = [];

  for (const anim of lootAnimations) {
    const y = anim.y - (1 - anim.ttl) * 24;
    const alpha = anim.ttl;
    const label = `${anim.by === 'agent' ? 'AGENT' : 'ENEMY'} +${lootIcon(anim.kind)}`;

    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.font = 'bold 13px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillStyle = anim.by === 'agent' ? '#34d399' : '#fb7185';
    ctx.fillText(label, anim.x, y);
    ctx.restore();

    const ttl = anim.ttl - 0.03;
    if (ttl > 0) {
      nextAnimations.push({ ...anim, ttl });
    }
  }

  lootAnimations = nextAnimations;
};

const applyObservationFog = (state: GameState) => {
  const [px, py] = toCanvas(state.player.pos, state.map_size);
  const revealRadius = (state.observation_radius / state.map_size) * canvas.width;

  exploredCtx.globalCompositeOperation = 'destination-out';
  const gradient = exploredCtx.createRadialGradient(px, py, revealRadius * 0.2, px, py, revealRadius);
  gradient.addColorStop(0, 'rgba(0,0,0,1)');
  gradient.addColorStop(1, 'rgba(0,0,0,0)');
  exploredCtx.fillStyle = gradient;
  exploredCtx.beginPath();
  exploredCtx.arc(px, py, revealRadius, 0, Math.PI * 2);
  exploredCtx.fill();
  exploredCtx.globalCompositeOperation = 'source-over';

  ctx.fillStyle = 'rgba(0,0,0,0.55)';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.save();
  ctx.globalAlpha = 0.9;
  ctx.drawImage(exploredMask, 0, 0);
  ctx.restore();
};

const keyMove = (): number => {
  const up = keys.has('w') || keys.has('arrowup');
  const down = keys.has('s') || keys.has('arrowdown');
  const left = keys.has('a') || keys.has('arrowleft');
  const right = keys.has('d') || keys.has('arrowright');

  if (up && left) return 5;
  if (up && right) return 6;
  if (down && left) return 7;
  if (down && right) return 8;
  if (up) return 1;
  if (down) return 2;
  if (left) return 3;
  if (right) return 4;
  return 0;
};

const sendAction = () => {
  if (ws.readyState !== WebSocket.OPEN) return;
  ws.send(
    JSON.stringify({
      type: 'player_action',
      payload: {
        move: keyMove(),
        aim_x: mouseAim[0],
        aim_y: mouseAim[1],
        attack: keys.has(' ') ? 1 : 0,
        interact: keys.has('e') ? 1 : 0,
        reload: keys.has('r') ? 1 : 0,
        heal: keys.has('h') ? 1 : 0,
        swap: 0,
      },
    }),
  );
};

ws.onopen = () => {
  statusEl.textContent = 'Connected to backend socket';
};

ws.onclose = () => {
  statusEl.textContent = 'Socket closed';
};

ws.onmessage = (event) => {
  const message = JSON.parse(event.data) as { type: string; payload: GameState };
  if (message.type !== 'game_state') return;

  const nextState = message.payload;
  updateLootAnimations(nextState);
  latestState = nextState;

  const aliveEnemy = latestState.enemies.find((enemy) => enemy.alive);
  const aliveEnemies = latestState.enemies.filter((enemy) => enemy.alive).length;
  statusEl.textContent = `Step ${latestState.step} · Reward ${latestState.reward.toFixed(2)} · Enemy Alive ${aliveEnemies}`;
  playerStatsEl.textContent = `Kills ${latestState.player.kills} · Armor ${latestState.player.armor.toFixed(0)} · Medkits ${latestState.player.medkits}`;
  const weaponName = latestState.player.current_weapon ?? 'None';
  weaponStatsEl.textContent = `Weapon ${weaponName.toUpperCase()} · Ammo ${latestState.player.current_mag}/${latestState.player.reserve_ammo}`;
  enemyStatsEl.textContent = aliveEnemy ? `Enemy HP ${aliveEnemy.health.toFixed(0)} / 100` : 'Enemy eliminated';
  aiActionEl.textContent = `[${latestState.ai_action.join(', ')}]`;

  agentHealthEl.style.width = `${Math.max(0, Math.min(100, latestState.player.health))}%`;
  agentArmorEl.style.width = `${Math.max(0, Math.min(100, latestState.player.armor))}%`;
  enemyHealthEl.style.width = `${Math.max(0, Math.min(100, aliveEnemy?.health ?? 0))}%`;
};

resetBtn.addEventListener('click', () => {
  if (ws.readyState !== WebSocket.OPEN) return;
  exploredCtx.fillStyle = 'black';
  exploredCtx.fillRect(0, 0, canvas.width, canvas.height);
  lootAnimations = [];
  prevLootMap = new Map();
  ws.send(JSON.stringify({ type: 'reset' }));
});

window.addEventListener('keydown', (e) => keys.add(e.key.toLowerCase()));
window.addEventListener('keyup', (e) => keys.delete(e.key.toLowerCase()));

canvas.addEventListener('mousemove', (e) => {
  if (!latestState) return;
  const rect = canvas.getBoundingClientRect();
  const x = ((e.clientX - rect.left) / rect.width) * latestState.map_size;
  const y = ((e.clientY - rect.top) / rect.height) * latestState.map_size;

  const dx = x - latestState.player.pos[0];
  const dy = y - latestState.player.pos[1];
  const len = Math.hypot(dx, dy);
  if (len < 1e-5) return;

  mouseAim = [
    Math.max(0, Math.min(10, Math.round((dx / len) * 5 + 5))),
    Math.max(0, Math.min(10, Math.round((dy / len) * 5 + 5))),
  ];
});

const draw = () => {
  requestAnimationFrame(draw);
  sendAction();
  if (!latestState) return;

  drawWorld(latestState);
  drawLootAnimations();
  applyObservationFog(latestState);
};

draw();
