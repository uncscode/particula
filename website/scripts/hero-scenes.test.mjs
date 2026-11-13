import assert from 'node:assert/strict';
import test from 'node:test';
import { SCENES, nextScene, createParticles } from '../src/components/hero-scenes.mjs';

function seededRandom(seed) {
  return () => ((seed = (1664525 * seed + 1013904223) >>> 0) / 4294967296);
}

test('refresh cycles cover every scene and avoid repeats at cycle boundaries', () => {
  for (let seed = 1; seed <= 100; seed++) {
    const random = seededRandom(seed);
    let saved;
    const storage = { getItem: () => saved, setItem: (_, value) => { saved = value; } };
    const visits = Array.from({ length: SCENES.length * 10 }, () => nextScene(storage, random));
    for (let i = 0; i < visits.length; i += SCENES.length) {
      assert.equal(new Set(visits.slice(i, i + SCENES.length)).size, SCENES.length);
    }
    visits.slice(1).forEach((scene, i) => assert.notEqual(scene, visits[i]));
  }
});

test('blocked or malformed session storage still produces a valid scene', () => {
  const ids = SCENES.map(({ id }) => id);
  for (const value of ['broken JSON', 'null', '{}', '{"last":"dust","remaining":["unknown"]}', '{"last":"dust","remaining":["dust"]}']) {
    assert.ok(ids.includes(nextScene({ getItem: () => value, setItem() {} })));
  }
  assert.ok(ids.includes(nextScene(undefined)));
  assert.ok(ids.includes(nextScene({ getItem() { throw Error(); }, setItem() { throw Error(); } })));
});

test('randomized scenes stay finite and inside the intended illustration area', () => {
  for (const { id } of SCENES) {
    for (let seed = 1; seed <= 100; seed++) {
      const particles = createParticles(id, seededRandom(seed));
      assert.equal(particles.length, 112);
      for (const p of particles) {
        for (const key of ['x', 'y', 'r', 'opacity', 'dx', 'dy', 'duration', 'phase']) {
          assert.ok(Number.isFinite(p[key]), `${id}: ${key}`);
        }
        assert.ok(p.x > 75 && p.x < 550 && p.y > 75 && p.y < 425);
        assert.ok(p.r >= 1 && p.r <= 20 && p.opacity > 0 && p.opacity <= 1);
        assert.ok(p.duration >= 6 && p.phase <= 0 && p.phase >= -p.duration);
      }
    }
    assert.notDeepEqual(createParticles(id, seededRandom(1)), createParticles(id, seededRandom(2)));
  }
});

test('condensation vapor travels toward its associated droplet', () => {
  const particles = createParticles('condensation', seededRandom(42));
  particles.forEach((p, i) => {
    if (p.kind !== 'vapor') return;
    const droplet = particles[Math.floor(i / 4) * 4];
    assert.ok(Math.abs(p.x + p.dx - droplet.x) < 1e-10);
    assert.ok(Math.abs(p.y + p.dy - droplet.y) < 1e-10);
  });
});

test('like-charge pairs separate and opposite-charge pairs approach', () => {
  const particles = createParticles('charge', seededRandom(42));
  for (let i = 0; i < particles.length; i += 2) {
    const [left, right] = particles.slice(i, i + 2);
    const distanceChange = right.dx - left.dx;
    assert.equal(distanceChange > 0, left.tone === right.tone);
    assert.equal(left.phase, right.phase);
    assert.equal(left.duration, right.duration);
  }
});

test('larger condensation droplets settle faster with their vapor envelopes', () => {
  const particles = createParticles('condensation', seededRandom(21));
  const drops = particles.filter((p) => p.kind === 'droplet').sort((a, b) => a.r - b.r);
  drops.slice(1).forEach((p, i) => assert.ok(p.fall > drops[i].fall));
  particles.forEach((p, i) => {
    const drop = particles[Math.floor(i / 4) * 4];
    assert.equal(p.fall, drop.fall);
    assert.equal(p.settlePhase, drop.settlePhase);
  });
});

test('scavenging captures small particles when their collector crosses them', () => {
  const particles = createParticles('scavenging', seededRandom(21));
  assert.equal(particles.filter((p) => p.kind === 'collector').length, 7);
  particles.forEach((p, i) => {
    if (p.kind !== 'captured') return;
    const collector = particles[Math.floor(i / 16) * 16];
    assert.ok(Math.abs(p.x - collector.x) < collector.r);
    assert.equal(p.duration, collector.duration);
    const captureTime = (p.y - collector.y) / collector.dy * collector.duration;
    const localTime = ((captureTime + collector.phase - p.phase) % p.duration + p.duration) % p.duration;
    assert.ok(Math.min(localTime, p.duration - localTime) < 1e-10);
  });
});
