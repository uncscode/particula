// Decorative process illustrations, not numerical simulation results.
export const SCENES = [
  { id: 'scattering', label: 'Light scattering', diagram: true },
  { id: 'pathways', label: 'Reaction pathways', diagram: true },
  { id: 'liquids', label: 'Liquid phase separation', diagram: true },
  { id: 'sizes', label: 'Size distributions', diagram: true },
  { id: 'chamber', label: 'Aerosol chamber', diagram: true },
  { id: 'cloud', label: 'Cloud parcels', viewBox: '78 38 484 366' },
  { id: 'charge', label: 'Charged collisions', viewBox: '78 47 484 366' },
  { id: 'nucleation', label: 'Ion-induced nucleation', diagram: true },
  { id: 'condensation', label: 'Condensation', viewBox: '78 69 484 366' },
  { id: 'scavenging', label: 'Scavenging', viewBox: '78 65 484 366' },
];

const STORAGE_KEY = 'particula:hero-scenes:v3';

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
  const droplets = Array.from({ length: 28 }, (_, i) => {
    const r = i === 0 ? 16 : between(4, 9);
    const fall = .25 * r * r;
    // Center the whole settling track in the drawing, including the vapor envelope.
    const midY = i === 0 ? 252 : between(198, 306);
    return {
      x: i === 0 ? 330 : between(184, 472), y: midY - fall / 2,
      r, fall, phase: between(-30, 0),
    };
  });
  const pairs = Array.from({ length: 56 }, () => ({
    x: between(-9, 9), y: between(-8, 8), phase: between(-15, 0),
  }));
  const collectors = Array.from({ length: 7 }, (_, i) => {
    const r = between(11, 18);
    const duration = 280 / r;
    return { x: 180 + i * 45, r, duration, phase: -random() * duration };
  });
  return Array.from({ length: 112 }, (_, i) => {
    const parcel = i % 3;
    const angle = between(0, Math.PI * 2);
    const p = {
      x: 134 + parcel * 138 + between(0, 78),
      y: 238 - parcel * 40 + between(0, 65),
      r: i % 17 === 0 ? between(5, 8) : between(1.2, 3.6),
      opacity: between(.28, .85), dx: between(2, 9), dy: between(-9, -2),
      duration: between(18, 36), phase: 0,
      kind: 'particle', tone: i % 13 === 0 ? 'gold' : 'teal', sign: '',
      fall: 0, settlePhase: 0,
    };
    if (scene === 'charge') {
      // Compare repulsion with capture by two oppositely charged droplets.
      const pair = Math.floor(i / 2);
      const side = i % 2 ? 1 : -1;
      const repels = pair < 28;
      const local = pair % 28;
      const focus = local === 0;
      const positive = repels ? pair % 2 === 0 : i % 2 === 1;
      const centerX = repels ? 208 : 409;
      p.x = focus ? centerX + side * 31 : centerX + (local % 5 - 2) * 29 + side * 6 + pairs[pair].x;
      p.y = focus ? 238 : 135 + Math.floor(local / 5) * 38 + pairs[pair].y;
      p.opacity = focus ? 1 : .28;
      p.r = focus ? 16 : between(1.3, 3.2);
      p.dx = side * (repels ? 14 : -10);
      p.dy = focus ? 0 : Math.sin(pair) * 6;
      p.duration = 16 + pair % 7;
      p.phase = pairs[pair].phase;
      p.tone = positive ? 'positive' : 'negative';
      if (focus) p.sign = positive ? '+' : '−';
      if (focus && !repels) {
        p.kind = 'collision-partner';
        p.x = centerX + side * 57;
        p.dx = -side * (57 - p.r);
        p.phase = 0;
        p.duration = 8;
      }
    } else if (scene === 'condensation') {
      const drop = droplets[Math.floor(i / 4)];
      p.x = drop.x;
      p.y = drop.y;
      p.tone = 'teal';
      // Shared settling keeps each vapor envelope aligned with its growing drop.
      // The size-based speed is illustrative, not a terminal-velocity calculation.
      p.fall = drop.fall;
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
