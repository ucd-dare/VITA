import { createTransport } from './vita-water.js';
import { createFlow } from './vita-flow.js';
import { createPerformance } from './vita-performance.js';

const hero = document.querySelector('.vita-hero');
const waterCanvas = document.querySelector('#vita-water');
const waterButton = document.querySelector('#hero-playback');
const flowButton = document.querySelector('#flow-playback');
const performanceButton = document.querySelector('#performance-playback');
const restart = document.querySelector('#flow-restart');
const slider = document.querySelector('#flow-progress');
const reduced = matchMedia('(prefers-reduced-motion: reduce)');
const water = createTransport(waterCanvas);
const flows = new Map([
    ['#gaussian-canvas', 'conventional'],
    ['#vita-flow-canvas', 'vita'],
].map(([selector, variant]) => {
    const canvas = document.querySelector(selector);
    return [canvas, createFlow(canvas, variant)];
}));
const visibleFlows = new Set();
const performances = new Map(['inference', 'convergence', 'precision'].map(kind => {
    const canvas = document.querySelector('#' + kind + '-canvas');
    return [canvas, { renderer: createPerformance(canvas, kind), progress: reduced.matches ? 1 : 0, visible: false }];
}));
const duration = 7700;
const performanceDuration = 5100;

let waterTime = reduced.matches ? 2.6 : .06;
let progress = reduced.matches ? 1 : 0;
let waterVisible = false;
let waterPlaying = !reduced.matches;
let flowPlaying = false;
let flowVisited = false;
let performancePlaying = !reduced.matches;
let request = 0;
let lastTime = 0;

function labelButton(button, playing) {
    button.dataset.mode = playing ? 'pause' : 'play';
    button.setAttribute('aria-label', playing ? 'Pause animation' : 'Play animation');
}

function syncWater() {
    const active = waterVisible && waterPlaying && !document.hidden;
    hero.classList.toggle('is-animating', active);
    water.setPlaying(active);
}

function syncProgress() {
    const percentage = Math.round(progress * 100);
    slider.value = Math.round(progress * 1000);
    slider.style.setProperty('--progress', percentage + '%');
    slider.setAttribute('aria-valuetext', percentage + '%');
}

function setFlowPlaying(playing) {
    flowPlaying = playing;
    labelButton(flowButton, playing);
}

function drawFlow(renderer) {
    renderer.render(progress, progress * duration / 1000);
}

function drawComparison() {
    for (const renderer of flows.values()) drawFlow(renderer);
}

function drawPerformance(scene) {
    scene.renderer.render(scene.progress, scene.progress * performanceDuration / 1000);
}

function performanceFinished() {
    return Array.from(performances.values()).every(scene => scene.progress === 1);
}

function syncPerformanceButton() {
    const mode = performanceFinished() ? 'restart' : performancePlaying ? 'pause' : 'play';
    if (mode === 'restart') performancePlaying = false;
    performanceButton.dataset.mode = mode;
    const label = mode === 'restart' ? 'Restart animation' : mode === 'pause' ? 'Pause animation' : 'Play animation';
    performanceButton.setAttribute('aria-label', label);
    performanceButton.title = label;
}

function schedule() {
    const performanceMotion = performancePlaying && Array.from(performances.values()).some(scene => scene.visible && scene.progress < 1);
    const active = (waterPlaying && waterVisible) || (flowPlaying && visibleFlows.size) || performanceMotion;
    if (!document.hidden && active && !request) request = requestAnimationFrame(animate);
}

function animate(now) {
    request = 0;
    const delta = lastTime ? Math.min(now - lastTime, 45) : 0;
    lastTime = now;

    if (waterPlaying && waterVisible) {
        waterTime += delta / 1000;
        water.render(waterTime);
    }

    if (flowPlaying && visibleFlows.size) {
        progress = Math.min(1, progress + delta / duration);
        if (progress === 1) setFlowPlaying(false);
        drawComparison();
        syncProgress();
    }

    if (performancePlaying) {
        for (const scene of performances.values()) {
            if (!scene.visible || scene.progress === 1) continue;
            scene.progress = Math.min(1, scene.progress + delta / performanceDuration);
            drawPerformance(scene);
            if (scene.progress === 1) syncPerformanceButton();
        }
    }

    schedule();
    if (!request) lastTime = 0;
}

const visibility = new IntersectionObserver((entries) => {
    for (const entry of entries) {
        if (entry.target === waterCanvas) {
            waterVisible = entry.intersectionRatio >= .025;
            syncWater();
        } else if (performances.has(entry.target)) {
            performances.get(entry.target).visible = entry.intersectionRatio >= .35;
        } else if (entry.intersectionRatio >= .45) {
            visibleFlows.add(entry.target);
        } else {
            visibleFlows.delete(entry.target);
        }
    }

    if (visibleFlows.size && !flowVisited) {
        flowVisited = true;
        setFlowPlaying(!reduced.matches);
    }

    schedule();
}, { threshold: [0, .025, .35, .45] });

const resize = new ResizeObserver((entries) => {
    for (const entry of entries) {
        if (entry.target === waterCanvas) water.render(waterTime);
        else if (performances.has(entry.target)) drawPerformance(performances.get(entry.target));
        else drawFlow(flows.get(entry.target));
    }
});

for (const canvas of [waterCanvas, ...flows.keys(), ...performances.keys()]) {
    visibility.observe(canvas);
    resize.observe(canvas);
}

waterButton.addEventListener('click', () => {
    waterPlaying = !waterPlaying;
    labelButton(waterButton, waterPlaying);
    syncWater();
    schedule();
});

function startFlow(fromBeginning) {
    flowVisited = true;
    if (fromBeginning) {
        progress = 0;
        drawComparison();
        syncProgress();
    }
    setFlowPlaying(true);
    schedule();
}

flowButton.addEventListener('click', () => {
    if (flowPlaying) setFlowPlaying(false);
    else startFlow(progress === 1);
});

restart.addEventListener('click', () => startFlow(true));

slider.addEventListener('input', () => {
    flowVisited = true;
    setFlowPlaying(false);
    progress = Number(slider.value) / 1000;
    drawComparison();
    syncProgress();
});

performanceButton.addEventListener('click', () => {
    if (performanceFinished()) {
        for (const scene of performances.values()) {
            scene.progress = 0;
            drawPerformance(scene);
        }
    }
    performancePlaying = !performancePlaying;
    syncPerformanceButton();
    schedule();
});

document.addEventListener('visibilitychange', () => {
    syncWater();
    if (document.hidden) {
        cancelAnimationFrame(request);
        request = 0;
        lastTime = 0;
    } else {
        schedule();
    }
});

reduced.addEventListener('change', () => {
    if (!reduced.matches) return;
    waterPlaying = false;
    setFlowPlaying(false);
    waterTime = Math.max(waterTime, 2.6);
    progress = 1;
    water.render(waterTime);
    drawComparison();
    for (const scene of performances.values()) {
        scene.progress = 1;
        drawPerformance(scene);
    }
    syncPerformanceButton();
    labelButton(waterButton, false);
    syncWater();
    syncProgress();
});

function observeEntrances() {
    if (reduced.matches) return;
    const selector = [
        '.flow-intro', '.comparison-panel', '.flow-controls',
        '.performance-intro', '.performance-metric', '.performance-controls',
        '.research-content h2.title', '.research-content .content > p',
        '.research-content .hero-body .container > p', '.research-content .interpolation-image',
        '.research-content .discussion-callout', '.research-content .column:has(video)',
        '#BibTeX pre', '.footer .content',
    ].join(', ');
    const candidates = Array.from(document.querySelectorAll(selector)).filter(element =>
        !element.parentElement.closest(selector) && element.getBoundingClientRect().top >= innerHeight - 32
    );
    const groups = new Map();
    const entrance = new IntersectionObserver((entries) => {
        for (const entry of entries) {
            if (!entry.isIntersecting) continue;
            entry.target.dataset.entrance = 'shown';
            entrance.unobserve(entry.target);
        }
    }, { threshold: .08, rootMargin: '0px 0px -34px 0px' });

    for (const element of candidates) {
        const previous = groups.get(element.parentElement) ?? 0;
        groups.set(element.parentElement, previous + 1);
        element.style.setProperty('--entrance-delay', Math.min(previous, 3) * 65 + 'ms');
        element.dataset.entrance = 'pending';
        entrance.observe(element);
    }
}

labelButton(waterButton, waterPlaying);
syncProgress();
syncPerformanceButton();
observeEntrances();
