import './styles.css';

type Vec2 = [number, number];

type GameState = {
  step: number;
  reward: number;
  terminated: boolean;
  truncated: boolean;
  map_size: number;
  zone: { center: Vec2; radius: number };
  player: {
    pos: Vec2;
    aim: Vec2;
    health: number;
    armor: number;
    kills: number;
    medkits: number;
  };
  enemies: Array<{ pos: Vec2; health: number; alive: boolean }>;
  loot: Array<{ pos: Vec2; kind: string }>;
  ai_action: number[];
};

const canvas = document.querySelector<HTMLCanvasElement>('#game');
const statusEl = document.querySelector<HTMLParagraphElement>('#status');
const playerStatsEl = document.querySelector<HTMLParagraphElement>('#player-stats');
const aiActionEl = document.querySelector<HTMLParagraphElement>('#ai-action');
const resetBtn = document.querySelector<HTMLButtonElement>('#reset');

if (!canvas || !statusEl || !playerStatsEl || !aiActionEl || !resetBtn) {
  throw new Error('Missing DOM elements');
}

const ctx = canvas.getContext('2d');
if (!ctx) {
  throw new Error('Cannot get canvas context');
}

const ws = new WebSocket('ws://localhost:8000/ws/game');
let latestState: GameState | null = null;

const keys = new Set<string>();
let mouseAim: Vec2 = [5, 5];

const toCanvas = (pos: Vec2, mapSize: number): Vec2 => [
  (pos[0] / mapSize) * canvas.width,
  (pos[1] / mapSize) * canvas.height,
];

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

  latestState = message.payload;
  const aliveEnemies = latestState.enemies.filter((enemy) => enemy.alive).length;
  statusEl.textContent = `Step ${latestState.step} · Reward ${latestState.reward.toFixed(2)} · Enemies ${aliveEnemies}`;
  playerStatsEl.textContent = `HP ${latestState.player.health.toFixed(0)} · Armor ${latestState.player.armor.toFixed(0)} · Kills ${latestState.player.kills} · Medkits ${latestState.player.medkits}`;
  aiActionEl.textContent = `AI Suggested Action [move, aimX, aimY, atk, int, rel, heal, swap]: [${latestState.ai_action.join(', ')}]`;
};

resetBtn.addEventListener('click', () => {
  if (ws.readyState !== WebSocket.OPEN) return;
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

  ctx.fillStyle = '#10202c';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  const zoneCenter = toCanvas(latestState.zone.center, latestState.map_size);
  const zoneRadius = (latestState.zone.radius / latestState.map_size) * canvas.width;
  ctx.strokeStyle = '#facc15';
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.arc(zoneCenter[0], zoneCenter[1], zoneRadius, 0, Math.PI * 2);
  ctx.stroke();

  for (const item of latestState.loot) {
    const [x, y] = toCanvas(item.pos, latestState.map_size);
    ctx.fillStyle = item.kind === 'weapon' ? '#f5f5f5' : item.kind === 'medkit' ? '#ef4444' : '#60a5fa';
    ctx.fillRect(x - 3, y - 3, 6, 6);
  }

  for (const enemy of latestState.enemies) {
    if (!enemy.alive) continue;
    const [x, y] = toCanvas(enemy.pos, latestState.map_size);
    ctx.fillStyle = '#f97316';
    ctx.beginPath();
    ctx.arc(x, y, 6, 0, Math.PI * 2);
    ctx.fill();
  }

  const [px, py] = toCanvas(latestState.player.pos, latestState.map_size);
  ctx.fillStyle = '#38bdf8';
  ctx.beginPath();
  ctx.arc(px, py, 7, 0, Math.PI * 2);
  ctx.fill();

  const aimEnd: Vec2 = [
    px + latestState.player.aim[0] * 24,
    py + latestState.player.aim[1] * 24,
  ];
  ctx.strokeStyle = '#ffffff';
  ctx.beginPath();
  ctx.moveTo(px, py);
  ctx.lineTo(aimEnd[0], aimEnd[1]);
  ctx.stroke();
};

draw();
