const copper = [185, 91, 60];
const rose = [164, 73, 101];
const lavender = [115, 105, 151];
const gold = [232, 163, 101];
const fullTurn = Math.PI * 2;

const mix = (start, end, fraction) => start + (end - start) * fraction;
const clamp = (value) => Math.max(0, Math.min(1, value));
const smooth = (value) => {
    const fraction = clamp(value);
    return fraction * fraction * (3 - 2 * fraction);
};

function color(channels, opacity) {
    return `rgba(${channels.join(',')},${clamp(opacity)})`;
}

function dot(context, x, y, radius, fill) {
    context.beginPath();
    context.arc(x, y, radius, 0, fullTurn);
    context.fillStyle = fill;
    context.fill();
}

function glow(context, x, y, radius, channels, opacity) {
    const light = context.createRadialGradient(x, y, 0, x, y, radius);
    light.addColorStop(0, color(channels, opacity));
    light.addColorStop(.43, color(channels, opacity * .38));
    light.addColorStop(1, color(channels, 0));
    context.fillStyle = light;
    context.fillRect(x - radius, y - radius, radius * 2, radius * 2);
}

function line(context, pointAt, end = 1, samples = 58) {
    context.beginPath();
    for (let index = 0; index <= samples; index += 1) {
        const [x, y] = pointAt(end * index / samples);
        if (index === 0) context.moveTo(x, y);
        else context.lineTo(x, y);
    }
}

function stroke(context, pointAt, fill, width, end = 1) {
    line(context, pointAt, end);
    context.strokeStyle = fill;
    context.lineWidth = width;
    context.stroke();
}

function beam(context, from, to, left, right) {
    const gradient = context.createLinearGradient(from, 0, to, 0);
    gradient.addColorStop(0, left);
    gradient.addColorStop(1, right);
    return gradient;
}

function pulse(context, x, y, channels, scale, alpha = 1) {
    glow(context, x, y, 14 * scale, channels, .24 * alpha);
    dot(context, x, y, 4.2 * scale, color([255, 246, 232], .95 * alpha));
    dot(context, x, y, 2.1 * scale, color(channels, .93 * alpha));
}

function drawInference(context, frame) {
    const { width, height, progress, scale } = frame;
    const start = width * .12;
    const end = width * .90;
    const fullLength = end - start;
    const top = height * .23;
    const bottom = height * .77;
    const barHeight = Math.max(13, Math.min(26, height * .105));

    context.setLineDash([2, 5]);
    for (const stop of [0, .2, .4, .6, .8, 1]) {
        const x = start + fullLength * stop;
        stroke(context, t => [x, mix(top, bottom, t)], color(lavender, .19), scale);
    }
    context.setLineDash([]);
    stroke(context, t => [start, mix(top, bottom, t)], color(lavender, .38), scale);

    const fmLength = .96;
    const bars = [
        { y: .36, length: fmLength * (7.58 / 14.36), from: color(gold, .94), to: color(rose, .91) },
        { y: .64, length: fmLength, from: color(lavender, .66), to: color(lavender, .89) },
    ];

    for (const { y, length, from, to } of bars) {
        const top = y * height - barHeight / 2;
        const radius = 4 * scale;
        context.beginPath();
        context.roundRect(start, top, fullLength * length, barHeight, [0, radius, radius, 0]);
        context.fillStyle = color(lavender, .085);
        context.fill();

        const filled = fullLength * Math.min(progress, length);
        if (filled === 0) continue;
        context.beginPath();
        context.roundRect(start, top, filled, barHeight, [0, radius, radius, 0]);
        context.fillStyle = beam(context, start, start + fullLength * length, from, to);
        context.fill();
    }
}

function drawConvergence(context, frame) {
    const { width, height, progress, time, scale } = frame;
    const left = width * .16;
    const right = width * .91;
    const top = height * .17;
    const bottom = height * .82;
    const span = bottom - top;
    const xAt = (fraction) => mix(left, right, fraction);

    context.setLineDash([2, 5]);
    for (const level of [.15, .38, .61, .84]) {
        const y = top + span * level;
        stroke(context, (t) => [xAt(t), y], color(lavender, .15), .85 * scale);
    }
    context.setLineDash([]);
    stroke(context, (t) => [xAt(t), bottom], color(lavender, .21), .94 * scale);
    stroke(context, (t) => [left, mix(top, bottom, t)], color(lavender, .32), .94 * scale);

    const pointAt = (t, warm, trace = 0) => {
        const quiet = 1 - Math.exp(-8 * t);
        const jitter = warm
            ? Math.sin(t * 39) * Math.exp(-3.7 * t) * .027
            : (Math.sin(t * 30) * .044 + Math.sin(t * 61) * .011) * Math.exp(-1.28 * t);
        const recovery = warm ? .87 * (1 - Math.exp(-6.5 * t)) : .68 * (1 - Math.exp(-2.45 * t));
        const spread = trace * Math.sin(t * Math.PI) * .036;
        return [
            xAt(t),
            top + span * (.065 + recovery + quiet * jitter + spread),
        ];
    };

    if (progress > 0) {
        line(context, (t) => pointAt(t, true), progress);
        context.lineTo(xAt(progress), bottom);
        context.lineTo(left, bottom);
        context.closePath();
        const fill = context.createLinearGradient(0, top, 0, bottom);
        fill.addColorStop(0, color(gold, .14));
        fill.addColorStop(1, color(rose, .025));
        context.fillStyle = fill;
        context.fill();
    }

    for (const warm of [false, true]) {
        const tone = warm ? copper : lavender;
        const tint = warm ? rose : lavender;
        stroke(context, (t) => pointAt(t, warm), color(tone, .18), 1.18 * scale);
        if (progress > 0) {
            for (const offset of [-.88, .88]) {
                stroke(context, (t) => pointAt(t, warm, offset), color(tone, .2), .78 * scale, progress);
            }
            const active = beam(context, left, right, color(tone, warm ? .91 : .54), color(tint, warm ? .86 : .65));
            stroke(context, (t) => pointAt(t, warm), active, (warm ? 2.48 : 1.82) * scale, progress);
        }

        const [x, y] = pointAt(progress, warm);
        pulse(context, x, y, tint, (warm ? 1 : .81) * scale,
            warm ? .92 + Math.sin(time * 2) * .08 : .71);
    }
}

const referenceEnds = [
    [-.072, -.182],
    [.023, -.153],
    [.064, -.084],
    [-.024, -.058],
    [.044, .036],
    [.061, .091],
    [-.057, .119],
    [.018, .184],
];

function drawPrecision(context, frame) {
    const { width, height, progress, time, scale } = frame;
    const start = width * .105;
    const end = width * .806;
    const middle = height * .503;
    const radius = height * .117;
    const pathAt = (t, lane, distance, exit, arc) => {
        const entry = height * .223 * lane;
        const bend = Math.sin(t * fullTurn - .47 + lane * .66)
            * Math.sin(Math.PI * t) * height * arc;
        return [
            mix(start, end + width * distance, t),
            middle + mix(entry, height * exit, smooth(t)) + bend,
        ];
    };

    glow(context, end, middle, radius * 2.38, gold, .29);
    glow(context, end, middle, radius * 1.5, rose, .15);
    context.save();
    context.translate(end, middle);
    context.rotate(-.21);
    context.beginPath();
    context.ellipse(0, 0, width * .058, height * .192, 0, 0, fullTurn);
    context.strokeStyle = color(lavender, .22);
    context.lineWidth = scale;
    context.stroke();
    context.restore();

    for (let index = 0; index < referenceEnds.length; index += 1) {
        const lane = mix(-1, 1, index / (referenceEnds.length - 1));
        const [distance, exit] = referenceEnds[index];
        const refPoint = (t) => pathAt(t, lane, distance, exit, .087);
        stroke(context, refPoint, color(lavender, .20), .86 * scale);
    }

    const path = (t) => pathAt(t, .24 * (1 - smooth(t)), 0, 0, .09);
    stroke(context, path, color(copper, .17), 1.02 * scale);
    if (progress > 0) {
        const warmTint = beam(context, start, end, color(copper, .88), color(rose, .96));
        stroke(context, path, warmTint, 2.6 * scale, progress);
    }
    const [x, y] = path(progress);
    pulse(context, x, y, rose, scale, .87 + Math.sin(time * 2.6) * .11);
}

const drawings = {
    inference: drawInference,
    convergence: drawConvergence,
    precision: drawPrecision,
};

export function createPerformance(canvas, kind) {
    const context = canvas.getContext('2d');
    const draw = drawings[kind];

    return {
        render(progress, seconds) {
            const width = canvas.clientWidth;
            const height = canvas.clientHeight;
            if (!context || !draw || !width || !height) return;

            const ratio = Math.min(window.devicePixelRatio || 1, 2);
            const backingWidth = Math.round(width * ratio);
            const backingHeight = Math.round(height * ratio);
            if (canvas.width !== backingWidth || canvas.height !== backingHeight) {
                canvas.width = backingWidth;
                canvas.height = backingHeight;
            }

            context.setTransform(ratio, 0, 0, ratio, 0, 0);
            context.clearRect(0, 0, width, height);
            context.lineCap = 'round';
            context.lineJoin = 'round';
            draw(context, {
                width,
                height,
                progress: clamp(progress),
                time: Math.max(0, seconds),
                scale: Math.max(.78, Math.min(1.26, width / 395)),
            });
        },
    };
}
