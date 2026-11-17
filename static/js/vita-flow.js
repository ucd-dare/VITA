const lavender = [118, 108, 158];
const copper = [183, 97, 65];
const plum = [129, 79, 158];
const rose = [157, 84, 120];
const amber = [177, 105, 66];
const fullTurn = Math.PI * 2;

const mix = (start, end, fraction) => start + (end - start) * fraction;
const clamp = (value) => Math.max(0, Math.min(value, 1));
const ease = (value) => {
  const bounded = clamp(value);
  return bounded * bounded * (3 - 2 * bounded);
};

function ink(color, alpha) {
  return `rgba(${color.join(',')},${clamp(alpha)})`;
}

function seededRandom(seed) {
  return () => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return (seed + 1) / 4294967297;
  };
}

function sampleGaussian(random) {
  const radius = Math.sqrt(-2 * Math.log(random())) * 0.35;
  const angle = random() * fullTurn;
  return [radius * Math.cos(angle), radius * Math.sin(angle)];
}

function makeGaussian() {
  const random = seededRandom(423);
  return Array.from({ length: 230 }, () => {
    const [x, y] = sampleGaussian(random);
    const [z] = sampleGaussian(random);
    const cap = Math.min(1, 1.07 / Math.hypot(x, y, z));
    return {
      point: [x * cap, y * cap, z * cap],
      size: 0.49 + random() ** 2 * 1.22,
      phase: random() * fullTurn,
    };
  });
}

function actionPoint(u, v) {
  const width = (0.82 + 0.18 * Math.cos(u * 2.45)) * (0.86 - 0.13 * Math.sin(u * 2.72));
  const ridge = 0.66 * Math.sin(u * 3.65 + v * 0.68);
  const fold = 0.29 * Math.cos(u * 6.4 - v * 1.7) * (0.78 + 0.22 * v);
  return [
    u * 1.27 + 0.11 * v * Math.cos(2.1 * u),
    (ridge + fold) * (0.85 + 0.15 * v) + 0.23 * v - 0.15 * v * v,
    width * v - 0.36 * Math.sin(u * 2.55) + 0.13 * Math.cos(v * 2.3 + u * 2.5),
  ];
}

function observationPoint(u, v) {
  const twist = -0.43 + u * 1.55;
  const width = v * (0.73 + 0.22 * Math.cos(u * 2.65));
  const curl = 0.26 * Math.sin(v * 4.2 - u * 2.2);
  return [
    1.17 * u + 0.24 * Math.sin(v * 2.1 + u * 2.6),
    width * Math.sin(twist) + 0.50 * Math.cos(u * 4.65 + v * 1.27) + curl,
    width * Math.cos(twist) + 0.26 * Math.sin(u * 4.4 + v * 1.8) - 0.13 * u,
  ];
}

function sourcePoint(u, v, alignment) {
  const source = observationPoint(u, v);
  const target = actionPoint(u, v);
  return [
    mix(source[0], target[0], alignment),
    mix(source[1], target[1] + 0.11, alignment),
    mix(source[2], target[2] + 0.058, alignment),
  ];
}

function projectFrom(x, y, scale, yaw, pitch) {
  const cosYaw = Math.cos(yaw);
  const sinYaw = Math.sin(yaw);
  const cosPitch = Math.cos(pitch);
  const sinPitch = Math.sin(pitch);

  return ([px, py, pz]) => {
    const across = px * cosYaw + pz * sinYaw;
    const depth = pz * cosYaw - px * sinYaw;
    const perspective = 5.5 / (5.5 - depth);
    return [
      x + across * scale * perspective,
      y - (py * cosPitch + depth * sinPitch) * scale * perspective,
      depth,
    ];
  };
}

function trace(context, project, pointAt, samples) {
  context.beginPath();
  for (let i = 0; i <= samples; i += 1) {
    const [x, y] = project(pointAt(i / samples));
    if (i === 0) context.moveTo(x, y);
    else context.lineTo(x, y);
  }
}

function ellipse(context, x, y, rx, ry, fill) {
  context.beginPath();
  context.ellipse(x, y, rx, ry, 0, 0, fullTurn);
  if (fill) context.fill();
  else context.stroke();
}

function halo(context, x, y, radius, color, opacity, flatten = 1) {
  context.save();
  context.translate(x, y);
  context.scale(1, flatten);
  const light = context.createRadialGradient(0, 0, 0, 0, 0, radius);
  light.addColorStop(0, ink(color, opacity));
  light.addColorStop(1, ink(color, 0));
  context.fillStyle = light;
  context.fillRect(-radius, -radius, radius * 2, radius * 2);
  context.restore();
}

function drawFloor(context, x, y, scale, color, strength) {
  if (strength < 0.01) return;

  halo(context, x, y, scale * 1.46, color, 0.105 * strength, 0.26);
  context.strokeStyle = ink(color, 0.22 * strength);
  context.lineWidth = 0.72;
  ellipse(context, x, y, scale * 1.21, scale * 0.19);
  context.setLineDash([1, 3]);
  context.strokeStyle = ink(color, 0.23 * strength);
  ellipse(context, x, y, scale * 1.38, scale * 0.26);
  context.setLineDash([]);
}

function drawSurface(context, project, pointAt, color, wash, scale, opacity) {
  const outline = (t) => {
    if (t < 0.25) return pointAt(mix(-1, 1, t * 4), -1);
    if (t < 0.5) return pointAt(1, mix(-1, 1, t * 4 - 1));
    if (t < 0.75) return pointAt(mix(1, -1, t * 4 - 2), 1);
    return pointAt(-1, mix(1, -1, t * 4 - 3));
  };

  const [x, y] = project([0, 0, 0]);
  const colorField = context.createLinearGradient(x - scale * 0.95, y - scale, x + scale * 0.78, y + scale);
  colorField.addColorStop(0, ink(wash, opacity * 0.26));
  colorField.addColorStop(0.46, ink(color, opacity * 0.12));
  colorField.addColorStop(1, ink(color, opacity * 0.25));
  trace(context, project, outline, 92);
  context.closePath();
  context.fillStyle = colorField;
  context.fill();

  for (let i = 0; i <= 17; i += 1) {
    const u = mix(-1, 1, i / 17);
    const edge = i === 0 || i === 17;
    context.strokeStyle = ink(color, (edge ? 0.64 : 0.30) * opacity);
    context.lineWidth = edge ? 1.06 : 0.64;
    trace(context, project, (t) => pointAt(u, mix(-1, 1, t)), 20);
    context.stroke();
  }

  for (let i = 0; i <= 11; i += 1) {
    const v = mix(-1, 1, i / 11);
    const edge = i === 0 || i === 11;
    const depth = mix(0.48, 0.87, (1 - v) / 2);
    context.strokeStyle = ink(color, (edge ? depth : depth * 0.64) * opacity);
    context.lineWidth = edge ? 1.18 : 0.70;
    trace(context, project, (t) => pointAt(mix(-1, 1, t), v), 46);
    context.stroke();
  }
}

function drawGaussian(context, project, particles, seconds, scale) {
  const [x, y] = project([0, 0, 0]);
  halo(context, x, y, scale * 1.13, lavender, 0.21);
  for (let i = 0; i < 3; i += 1) {
    const tilt = (i - 1) * Math.PI / 3 + seconds * 0.06;
    const radius = mix(0.82, 0.98, i / 2);
    trace(context, project, (t) => {
      const angle = t * fullTurn;
      return [Math.cos(angle) * radius, Math.sin(angle) * Math.sin(tilt) * radius, Math.sin(angle) * Math.cos(tilt) * radius];
    }, 52);
    context.strokeStyle = ink(lavender, i === 1 ? 0.36 : 0.25);
    context.lineWidth = 0.75;
    context.stroke();
  }

  for (const particle of particles) {
    const breathe = 1 + 0.035 * Math.sin(seconds * 1.1 + particle.phase);
    const [dx, dy, depth] = project(particle.point.map((coordinate) => coordinate * breathe));
    const alpha = mix(0.24, 0.75, clamp((depth + 1) / 2));
    const size = particle.size * mix(0.72, 1.01, alpha);
    context.fillStyle = ink(lavender, alpha);
    ellipse(context, dx, dy, size, size, true);
  }
}

function drawPassage(context, sourceProject, targetProject, source, approach, seconds, scale, conditioned = false) {
  const identity = (point) => point;
  const sourceCenter = sourceProject([0, 0, 0]);
  const targetCenter = targetProject([0, 0, 0]);
  const opacity = mix(0.93, 0.49, approach);
  const sweep = Math.min(scale * 0.88, (targetCenter[0] - sourceCenter[0]) * 0.33) * (1 - 0.67 * approach);
  const pigment = context.createLinearGradient(sourceCenter[0], sourceCenter[1], targetCenter[0], targetCenter[1]);
  const startInk = conditioned ? lavender : plum;
  pigment.addColorStop(0, ink(startInk, 0.23 * opacity));
  pigment.addColorStop(0.48, ink(rose, 0.46 * opacity));
  pigment.addColorStop(1, ink(copper, 0.26 * opacity));

  let junction;
  for (let i = 0; i < 8; i += 1) {
    const fraction = i / 7;
    const u = mix(-0.23, 0.65, (i * 3 % 8) / 7);
    const v = mix(-0.92, 0.92, fraction);
    const [startX, startY] = sourceProject(source(u, v, i));
    const [endX, endY] = targetProject(actionPoint(u, v));
    const phase = mix(-1.4, 1.4, fraction);
    const at = (t) => {
      const bend = Math.sin(t * Math.PI) * sweep;
      return [
        mix(startX, endX, t) + bend * 0.19 * Math.sin(t * fullTurn + phase),
        mix(startY, endY, t) + bend * (0.86 * Math.sin(t * 4.17 + phase + seconds * 0.3) + phase * 0.29),
      ];
    };

    if (conditioned && i === 3) junction = at(0.53);
    trace(context, identity, at, 24);
    context.strokeStyle = pigment;
    context.lineWidth = i % 3 === 0 ? 1 : 0.68;
    context.stroke();

    for (let j = 0; j < 2; j += 1) {
      const t = (seconds * (0.26 + 0.019 * (i % 3)) + fraction * 0.73 + j / 2) % 1;
      const brightness = Math.sin(t * Math.PI) ** 0.7 * opacity;
      const trail = Math.min(t, 0.092);
      const color = t < 0.46 ? startInk : rose;
      trace(context, identity, (step) => at(t - trail * step), 5);
      context.strokeStyle = ink(color, 0.60 * brightness);
      context.lineWidth = 1.35;
      context.stroke();
      const [x, y] = at(t);
      context.fillStyle = ink(color, 0.87 * brightness);
      ellipse(context, x, y, j === 0 ? 1.35 : 0.84, j === 0 ? 1.35 : 0.84, true);
    }
  }
  return junction;
}

function drawConditioning(context, width, height, seconds, junction) {
  const center = width * 0.505;
  const spread = Math.min(width * 0.096, 61);
  const middle = height * 0.216;
  const spacing = Math.min(height * 0.054, 17);
  const radius = mix(2.15, 3.2, clamp((spread - 28) / 33));
  const layers = [
    [-1, 0, 1].map((offset) => [center - spread, middle + offset * spacing]),
    [-0.55, 0.55].map((offset) => [center, middle + offset * spacing]),
    [[center + spread, middle]],
  ];
  const colors = [amber, rose, copper];

  context.save();
  halo(context, center, middle, spread * 1.26, amber, 0.13, 0.45);

  for (let depth = 0; depth < layers.length - 1; depth += 1) {
    const current = layers[depth];
    const next = layers[depth + 1];
    for (const [i, [x, y]] of current.entries()) {
      for (const [j, [nextX, nextY]] of next.entries()) {
        const inkField = context.createLinearGradient(x, y, nextX, nextY);
        inkField.addColorStop(0, ink(colors[depth], 0.42));
        inkField.addColorStop(1, ink(colors[depth + 1], 0.49));
        context.beginPath();
        context.moveTo(x, y);
        context.bezierCurveTo(x + spread * 0.46, y, nextX - spread * 0.46, nextY, nextX, nextY);
        context.strokeStyle = inkField;
        context.lineWidth = 0.94;
        context.stroke();

        const phase = (seconds * 0.55 + i * 0.31 + j * 0.24 - depth * 0.17 + 1) % 1;
        const transit = Math.sin(Math.PI * phase) ** 2;
        if (transit < 0.55) continue;
        const xOnPath = mix(x, nextX, phase + 0.38 * phase * (1 - phase) * (1 - 2 * phase));
        const yOnPath = mix(y, nextY, ease(phase));
        context.fillStyle = ink(colors[depth + 1], 0.72 * transit);
        ellipse(context, xOnPath, yOnPath, radius * 0.48, radius * 0.48, true);
      }
    }
  }

  for (const [depth, layer] of layers.entries()) {
    for (const [x, y] of layer) {
      const open = depth === 1;
      const size = radius * (open ? 1.33 : 1);
      context.fillStyle = ink(colors[depth], open ? 0.15 : 0.89);
      ellipse(context, x, y, size, size, true);
      if (!open) continue;
      context.strokeStyle = ink(rose, 0.66);
      context.lineWidth = 0.85;
      ellipse(context, x, y, size, size);
      context.fillStyle = ink(rose, 0.85);
      ellipse(context, x, y, radius * 0.43, radius * 0.43, true);
    }
  }

  const [startX, startY] = layers[2][0];
  const [endX, endY] = junction;
  const offset = Math.min(width * 0.033, 18);
  const drop = endY - startY;
  const channel = context.createLinearGradient(startX, startY, endX, endY);
  channel.addColorStop(0, ink(copper, 0.73));
  channel.addColorStop(1, ink(rose, 0.63));

  const branchPoint = (t) => {
    const inv = 1 - t;
    return [
      inv ** 3 * startX + 3 * inv ** 2 * t * (startX + offset) + 3 * inv * t ** 2 * (endX - offset) + t ** 3 * endX,
      inv ** 3 * startY + 3 * inv ** 2 * t * (startY + drop * 0.72) + 3 * inv * t ** 2 * (endY - drop * 0.19) + t ** 3 * endY,
    ];
  };

  trace(context, (point) => point, branchPoint, 25);
  context.strokeStyle = channel;
  context.lineWidth = 1.35;
  context.stroke();

  const phase = (seconds * 0.69 + 0.15) % 1;
  const [x, y] = branchPoint(phase);
  const glow = Math.sin(phase * Math.PI) ** 0.68;
  halo(context, x, y, radius * 2.4, copper, 0.28 * glow);
  context.fillStyle = ink(copper, 0.96 * glow);
  ellipse(context, x, y, radius * 0.62, radius * 0.62, true);

  halo(context, endX, endY, radius * 3.2, rose, 0.2);
  context.fillStyle = '#fff5ec';
  ellipse(context, endX, endY, radius * 1.01, radius * 1.01, true);
  context.fillStyle = ink(rose, 0.81);
  ellipse(context, endX, endY, radius * 0.59, radius * 0.59, true);
  context.restore();
}

export function createFlow(canvas, variant = 'vita') {
  const context = canvas.getContext('2d');
  const conventional = variant === 'conventional';
  const particles = conventional ? makeGaussian() : null;

  function render(progress, seconds) {
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    if (!context || !width || !height) return;

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const viewportWidth = Math.round(width * dpr);
    const viewportHeight = Math.round(height * dpr);
    if (canvas.width !== viewportWidth || canvas.height !== viewportHeight) {
      canvas.width = viewportWidth;
      canvas.height = viewportHeight;
    }

    context.setTransform(dpr, 0, 0, dpr, 0, 0);
    context.clearRect(0, 0, width, height);
    context.lineJoin = 'round';
    context.lineCap = 'round';

    const approach = conventional ? 0 : ease((progress - 0.07) / 0.91);
    const shape = ease((approach - 0.10) / 0.89);
    const startScale = Math.min(width * 0.156, height * 0.28);
    const finalScale = Math.min(width * 0.257, height * 0.303);
    const scale = mix(startScale, finalScale, approach);
    const yaw = -0.39 + 0.045 * Math.sin(seconds * 0.71);
    const pitch = 0.46 + 0.036 * Math.cos(seconds * 0.57);
    const xSource = width * mix(conventional ? 0.239 : 0.244, 0.481, approach);
    const xTarget = width * mix(0.743, 0.519, approach);
    const ySource = height * mix(conventional ? 0.581 : 0.472, 0.484, approach);
    const yTarget = height * mix(conventional ? 0.587 : 0.496, 0.535, approach);
    const sourceYaw = mix(yaw - (conventional ? 0 : 0.34), yaw, approach);
    const sourceProject = projectFrom(xSource, ySource, scale, sourceYaw, pitch);
    const targetProject = projectFrom(xTarget, yTarget, scale, yaw, pitch);
    const sourceColor = conventional ? lavender : plum;

    halo(context, xSource, ySource, scale * 1.44, sourceColor, 0.095);
    halo(context, xTarget, yTarget, scale * 1.51, copper, 0.11);
    drawFloor(context, xSource, ySource + scale * 0.91, scale, sourceColor, 0.89 * (1 - approach));
    drawFloor(context, xTarget, yTarget + scale * 0.79, scale, copper, 0.83);

    if (conventional) {
      const fromGaussianEdge = (_, v, i) => [0.27 + 0.07 * Math.cos(i * 3.1), v * 0.52, 0.29 * Math.sin(i * 2.13)];
      const junction = drawPassage(context, sourceProject, targetProject, fromGaussianEdge, 0, seconds, scale, true);
      drawSurface(context, targetProject, actionPoint, copper, [229, 145, 101], scale, 1);
      drawGaussian(context, sourceProject, particles, seconds, scale);
      drawConditioning(context, width, height, seconds, junction);
      return;
    }

    const currentSource = (u, v) => sourcePoint(u, v, shape);
    drawPassage(context, sourceProject, targetProject, currentSource, approach, seconds, scale);
    drawSurface(context, targetProject, actionPoint, copper, [229, 145, 101], scale, 0.98);
    drawSurface(context, sourceProject, currentSource, plum, [191, 145, 198], scale, 0.97);
  }

  return { render };
}
