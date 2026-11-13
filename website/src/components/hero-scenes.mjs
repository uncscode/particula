// Decorative process illustrations, not numerical simulation results.
export const SCENES = [
  { id: 'cloud', label: 'Cloud drift' },
  { id: 'plume', label: 'Rising plume' },
  { id: 'fire', label: 'Fire & smoke' },
  { id: 'dust', label: 'Emissions' },
  { id: 'charge', label: 'Charged particles' },
  { id: 'condensation', label: 'Condensation' },
  { id: 'scavenging', label: 'Scavenging' },
];

const STORAGE_KEY = 'particula:hero-scenes:v2';

/** Visit every scene in a shuffled order before repeating, even across reloads. */
export function nextScene(storage, random = Math.random) {
  const ids = SCENES.map(({ id }) => id);
  let remaining = [];
  let last;
  try {
    const saved = JSON.parse(storage.getItem(STORAGE_KEY));
    if (saved && ids.includes(saved.last) && Array.isArray(saved.remaining)
      && saved.remaining.every((id) => ids.includes(id) && id !== saved.last)
      && new Set(saved.remaining).size === saved.remaining.length) {
      remaining = saved.remaining;
      last = saved.last;
    }
  } catch { /* Storage can be unavailable; a random scene still works. */ }
  if (!remaining.length) {
    remaining = [...ids];
    for (let i = remaining.length - 1; i > 0; i--) {
      const j = Math.floor(random() * (i + 1));
      [remaining[i], remaining[j]] = [remaining[j], remaining[i]];
    }
    if (remaining.at(-1) === last) {
      [remaining[0], remaining[remaining.length - 1]] = [remaining.at(-1), remaining[0]];
    }
  }
  const id = remaining.pop();
  try { storage.setItem(STORAGE_KEY, JSON.stringify({ last: id, remaining })); } catch {}
  return id;
}

export function createParticles(scene, random = Math.random) {
  const between = (a, b) => a + random() * (b - a);
  const lobes = [[220, 270, 115, 70], [320, 220, 120, 100], [420, 285, 95, 70], [310, 320, 140, 62]];
  const droplets = Array.from({ length: 28 }, () => ({
    x: between(185, 475), y: between(155, 365), r: between(5, 11),
    phase: between(-30, 0),
  }));
  const pairs = Array.from({ length: 56 }, () => ({
    x: between(-9, 9), y: between(-8, 8), phase: between(-15, 0),
  }));
  const collectors = Array.from({ length: 7 }, (_, i) => {
    const r = between(11, 18);
    const duration = 280 / r;
    return { x: 180 + i * 45, r, duration, phase: -random() * duration };
  });
  return Array.from({ length: 112 }, (_, i) => {
    const [cx, cy, rx, ry] = lobes[i % lobes.length];
    const angle = between(0, Math.PI * 2);
    const radius = Math.sqrt(random());
    const p = {
      x: cx + Math.cos(angle) * radius * rx,
      y: cy + Math.sin(angle) * radius * ry,
      r: i % 17 === 0 ? between(10, 20) : between(1.1, 4.5),
      opacity: between(.28, .85), dx: between(6, 22), dy: between(-20, -8),
      duration: between(18, 36), phase: 0,
      kind: 'particle', tone: i % 13 === 0 ? 'gold' : 'teal', sign: '',
      fall: 0, settlePhase: 0,
    };
    if (scene === 'plume' || scene === 'fire') {
      p.x = between(290, 365);
      p.y = between(370, 410);
      p.dx = between(-125, 135);
      p.dy = between(-290, -175);
      p.duration = scene === 'fire' ? between(7, 15) : between(13, 25);
      p.r = i % 9 === 0 ? between(10, 19) : between(1.5, 6);
      p.tone = scene === 'fire' ? (i % 9 === 0 ? 'smoke' : i % 3 ? 'ember' : 'gold') : 'teal';
    } else if (scene === 'dust') {
      p.x = between(145, 400);
      p.y = between(365, 395);
      p.dx = between(95, 185);
      p.dy = between(-175, -55);
      p.r = between(1.8, 6.5);
      p.duration = between(11, 23);
      p.tone = i % 3 ? 'dust' : 'gold';
    } else if (scene === 'charge') {
      // Neighbouring pairs alternately approach and separate; signs move with them.
      const pair = Math.floor(i / 2);
      const side = i % 2 ? 1 : -1;
      const row = Math.floor(pair / 8);
      const repels = pair % 3 === 0;
      const positive = repels ? pair % 2 === 0 : i % 2 === 1;
      p.x = 175 + (pair % 8) * 42 + side * 13 + pairs[pair].x;
      p.y = 155 + row * 35 + pairs[pair].y;
      p.r = pair % 5 === 0 ? 8 : between(2, 4);
      p.dx = side * (repels ? 12 : -4);
      p.dy = Math.sin(pair) * 12;
      p.duration = 16 + pair % 7;
      p.phase = pairs[pair].phase;
      p.tone = positive ? 'positive' : 'negative';
      if (p.r === 8) p.sign = positive ? '+' : '−';
    } else if (scene === 'condensation') {
      const drop = droplets[Math.floor(i / 4)];
      p.x = drop.x;
      p.y = drop.y;
      p.tone = 'teal';
      // Shared settling keeps each vapor envelope aligned with its growing drop.
      // The size-based speed is illustrative, not a terminal-velocity calculation.
      p.fall = drop.r * drop.r;
      p.settlePhase = drop.phase;
      if (i % 4 === 0) {
        p.kind = 'droplet';
        p.r = drop.r;
        p.dx = between(-8, 8);
        p.dy = between(-12, -5);
        p.duration = between(18, 30);
      } else {
        p.kind = 'vapor';
        p.x += Math.cos(angle) * between(25, 55);
        p.y += Math.sin(angle) * between(25, 55);
        p.dx = drop.x - p.x;
        p.dy = drop.y - p.y;
        p.r = between(1, 2.2);
        p.duration = between(6, 12);
      }
    } else if (scene === 'scavenging') {
      const collector = collectors[Math.floor(i / 16)];
      p.duration = collector.duration;
      p.phase = collector.phase;
      p.dx = 0;
      p.dy = 300;
      if (i % 16 === 0) {
        p.kind = 'collector';
        p.x = collector.x;
        p.y = 100;
        p.r = collector.r;
        p.tone = 'teal';
        p.opacity = .75;
      } else {
        p.kind = 'captured';
        p.x = collector.x + between(-.7, .7) * collector.r;
        p.y = between(140, 390);
        p.r = between(1.2, 3);
        p.tone = i % 3 ? 'dust' : 'gold';
        const captureTime = (p.y - 100) / 300 * collector.duration;
        // Fade at the time this column's collector crosses the particle's height.
        p.phase = -((-collector.phase - captureTime + p.duration) % p.duration);
      }
    }
    if (scene !== 'charge' && scene !== 'scavenging') p.phase = -random() * p.duration;
    return p;
  });
}

export function particleStyle(p) {
  return `--dx:${p.dx.toFixed(1)}px;--dy:${p.dy.toFixed(1)}px;--duration:${p.duration.toFixed(3)}s;--phase:${p.phase.toFixed(3)}s;--fall:${p.fall.toFixed(1)}px;--settle-phase:${p.settlePhase.toFixed(1)}s`;
}

export function particleFill(p) {
  if (p.tone === 'gold') return 'url(#cloud-gold)';
  if (p.tone === 'teal') return p.r > 7 || p.kind === 'droplet' ? 'url(#cloud-teal)' : 'var(--cloud-teal)';
  return `var(--cloud-${p.tone})`;
}
