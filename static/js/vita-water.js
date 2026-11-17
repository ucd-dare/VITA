const vertexShader = `
  attribute vec2 aPosition;
  varying vec2 vUv;

  void main() {
    vUv = vec2(aPosition.x * .5 + .5, .5 - aPosition.y * .5);
    gl_Position = vec4(aPosition, 0., 1.);
  }
`;

const fragmentShader = `
  precision highp float;

  varying vec2 vUv;
  uniform float uTime;
  uniform float uPortrait;

  float hash(vec2 p) {
    vec3 q = fract(vec3(p.xyx) * .1031);
    q += dot(q, q.yzx + 33.33);
    return fract((q.x + q.y) * q.z);
  }

  float noise(vec2 p) {
    vec2 cell = floor(p);
    vec2 f = fract(p);
    vec2 w = f * f * (3. - 2. * f);
    return mix(
      mix(hash(cell), hash(cell + vec2(1., 0.)), w.x),
      mix(hash(cell + vec2(0., 1.)), hash(cell + 1.), w.x),
      w.y
    );
  }

  float current(float x, float t) {
    return .753 + .058 * sin(6.42 * x - .49) +
      .011 * sin(x * 14. - t * .79) +
      .006 * sin(x * 24. - t * 1.18);
  }

  void main() {
    vec2 uv = vUv;
    float x = uv.x;
    float t = uTime;
    float center = mix(current(x, t), .746 + .052 * sin(6.15 * x - .43), uPortrait * .65);
    float width = mix(.078 + .046 * sin(3.19 * x + .1), .072 + .032 * sin(3.2 * x + .18), uPortrait);
    float source = mix(.225, .255, uPortrait);
    float opening = smoothstep(source - .069, source + .17, x);
    float outlet = smoothstep(.80, 1.01, x);
    width *= mix(.27, 1., opening) * (1. - .54 * outlet);
    float q = (uv.y - center) / width;
    float reveal = smoothstep(-.55, 1.85, t);
    float reach = 1. - smoothstep(reveal * 1.68 - .34, reveal * 1.68 + .19, x);
    if (abs(q) > 2.65 || reach < .001) {
      gl_FragColor = vec4(0.);
      return;
    }

    float plume = noise(vec2(7. * x - t * .44, q * 1.87 + t * .026));
    float fringe = noise(vec2(12. * x - t * .7, 3.1 * q));
    float spread = 1. + .32 * (plume - .5);
    float lateral = (q + .34 * (plume - .5) + .16 * (fringe - .5)) / spread;
    float density = exp(-1.08 * lateral * lateral);
    float depth = density * density;
    float taper = 1. - smoothstep(1.9, 2.65, abs(q));
    float drift = x * (mix(9.2, 5.8, uPortrait)) - t * .79;
    float bend = q * 3.95 + .33 * sin(1.57 * drift + 2.4 * q);
    vec2 p = vec2(drift, bend);
    vec2 curl = vec2(noise(.68 * p + vec2(.7, 1.1)), noise(.73 * p + vec2(6.2, 4.4))) - .5;
    vec2 plane = p * vec2(1.34, .92) + curl * 2.08;
    vec2 surface = plane + 1.15 * vec2(
      noise(1.6 * plane + vec2(7.8, -4.1)) - .5,
      noise(1.32 * plane + vec2(-4.6, 7.3)) - .5
    );

    float broad = noise(plane);
    float fine = noise(surface * 2.9);
    float primaryPhase = 1.35 * surface.x + 2.25 * surface.y + 2.3 * broad;
    float crossPhase = 2.15 * surface.x - .53 * surface.y + 2. * fine;
    float longPhase = .68 * plane.x - .49 * plane.y + 1.6 * broad;
    float ridges = sin(primaryPhase);
    float crossRipple = sin(crossPhase);
    vec2 slope = .33 * cos(primaryPhase) * vec2(.78, .63) +
      .24 * cos(crossPhase) * vec2(.75, -.42) +
      .34 * cos(longPhase) * vec2(.31, -.92);

    vec3 normal = normalize(vec3(-slope.x + .13 * curl.x, -slope.y + .11 * q, .91));
    vec3 illumination = normalize(vec3(-.44, .46, .81));
    vec3 halfway = normalize(illumination + vec3(0., 0., 1.));
    float reflected = pow(max(0., dot(normal, halfway)), 19.);
    float reflectedSide = pow(max(0., dot(normal, normalize(vec3(.5, -.2, .9)))), 10.);
    float lightWeb = exp(-11. * ridges * ridges);
    float foam = lightWeb * (.19 + .81 * smoothstep(-.50, .89, crossRipple));
    float glint = reflected * (.38 + .62 * fine);
    float fold = sin(1.58 * surface.x - 1.14 * surface.y + 2.1 * broad);
    float refracted = smoothstep(.52, .96, fold) * smoothstep(.24, .74, broad);
    float bands = .5 + .5 * sin(drift * 1.46 - q * 5.9 + 3.4 * curl.y);

    float travel = x + .18 * (broad - .5) + .045 * q;
    float roseStage = smoothstep(.13, .58, travel);
    float lilacStage = smoothstep(.46, .99, travel);
    vec3 amber = vec3(.78, .375, .17);
    vec3 rose = vec3(.72, .285, .495);
    vec3 plum = vec3(.46, .305, .68);
    vec3 pigment = mix(mix(amber, rose, roseStage), plum, lilacStage);
    vec3 wash = mix(vec3(.99, .82, .62), vec3(.80, .71, .91), smoothstep(.18, .94, travel));
    vec3 base = mix(wash, pigment, .29 + .61 * depth * (1. - .49 * reflectedSide));
    base *= .91 + .17 * bands + .075 * dot(normal, illumination);
    float cycle = fract(t * .195);
    float pulse = (x - mix(-.08, 1.05, cycle)) * 10.;
    float carry = exp(-pulse * pulse);
    float light = (.35 * foam + .68 * glint + .31 * refracted + .24 * carry) * (.24 + .76 * density);
    vec3 pearl = mix(vec3(1., .965, .87), vec3(.987, .945, 1.), lilacStage);
    base = mix(base, pearl, min(light, .91));

    float bendMist = (q + .69 + .64 * (broad - .5)) * .91;
    float ghost = exp(-bendMist * bendMist) * (.23 + .77 * broad) * (.53 + .47 * bands);
    float haze = exp(-.63 * (q - .23) * (q - .23)) * (.35 + .65 * plume);
    float intake = smoothstep(.012, .075, x) * mix(.78, 1., opening);
    float eddy = .74 + .26 * smoothstep(-.19, .72, q);
    float mist = (.15 * haze + .14 * ghost) * (1. - depth);
    float body = density * (.36 + .61 * depth) * (.82 + .18 * broad);
    float opacity = (body * eddy + mist) * intake * (1. - .90 * outlet) * reach * taper;
    vec3 diffuse = mix(vec3(.99, .72, .52), vec3(.77, .65, .91), smoothstep(.13, .95, travel));
    gl_FragColor = vec4(mix(diffuse, base, density), opacity);
  }
`;

function compile(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);

  if (gl.getShaderParameter(shader, gl.COMPILE_STATUS)) return shader;
  gl.deleteShader(shader);
  return null;
}

function createWater(gl) {
  const vertex = compile(gl, gl.VERTEX_SHADER, vertexShader);
  const fragment = compile(gl, gl.FRAGMENT_SHADER, fragmentShader);
  if (!vertex || !fragment) {
    if (vertex) gl.deleteShader(vertex);
    if (fragment) gl.deleteShader(fragment);
    return null;
  }

  const program = gl.createProgram();
  gl.attachShader(program, vertex);
  gl.attachShader(program, fragment);
  gl.linkProgram(program);
  gl.deleteShader(vertex);
  gl.deleteShader(fragment);

  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    gl.deleteProgram(program);
    return null;
  }
  gl.useProgram(program);

  const quad = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, quad);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);

  const vertexPosition = gl.getAttribLocation(program, 'aPosition');
  gl.enableVertexAttribArray(vertexPosition);
  gl.vertexAttribPointer(vertexPosition, 2, gl.FLOAT, false, 0, 0);

  const portrait = gl.getUniformLocation(program, 'uPortrait');
  const time = gl.getUniformLocation(program, 'uTime');

  return (width, height, seconds, ratio) => {
    gl.viewport(0, 0, width, height);
    gl.uniform1f(portrait, Math.max(0, Math.min(1, 1.75 - ratio * 1.20)));
    gl.uniform1f(time, seconds);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  };
}

export function createTransport(canvas) {
  const gl = canvas.getContext('webgl', {
    alpha: true,
    antialias: false,
    depth: false,
    stencil: false,
    premultipliedAlpha: false,
    preserveDrawingBuffer: true,
  });
  let draw = gl && createWater(gl);
  let previousTime = 0;
  let playing = false;
  let video;
  let videoType;

  function setPlaying(active) {
    playing = active;
    if (!video?.isConnected) return;

    if (active) {
      if (video.paused) video.play().catch(() => {});
    } else {
      video.pause();
    }
  }

  function showVideo(ratio) {
    if (!video) {
      video = document.createElement('video');
      video.className = 'hero-water-video';
      video.setAttribute('aria-hidden', 'true');
      video.muted = true;
      video.defaultMuted = true;
      video.loop = true;
      video.playsInline = true;
      video.preload = 'auto';
    }

    if (!video.isConnected) {
      canvas.after(video);
      canvas.parentElement.classList.add('hero-water-fallback');
    }

    const nextType = ratio < .92 ? 'mobile' : 'wide';
    if (nextType === videoType && video.getAttribute('src')) return;
    videoType = nextType;
    video.poster = new URL('../media/vita-water-' + videoType + '.webp', import.meta.url).href;
    video.src = new URL('../media/vita-water-' + videoType + '.mp4', import.meta.url).href;
    setPlaying(playing);
  }

  function render(time) {
    previousTime = time;
    const cssWidth = canvas.clientWidth;
    const cssHeight = canvas.clientHeight;
    if (!cssWidth || !cssHeight) return;

    if (!draw || gl.isContextLost()) {
      showVideo(cssWidth / cssHeight);
      return;
    }

    const scale = Math.min(window.devicePixelRatio || 1, 1.35, 1500 / cssWidth, 1080 / cssHeight);
    const width = Math.max(1, Math.round(cssWidth * scale));
    const height = Math.max(1, Math.round(cssHeight * scale));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    draw(width, height, time, cssWidth / cssHeight);
  }

  if (gl) {
    canvas.addEventListener('webglcontextlost', (event) => {
      event.preventDefault();
      render(previousTime);
    });
    canvas.addEventListener('webglcontextrestored', () => {
      draw = createWater(gl);
      if (draw && video?.isConnected) {
        video.pause();
        video.remove();
        video.removeAttribute('src');
        video.load();
        canvas.parentElement.classList.remove('hero-water-fallback');
      }
      render(previousTime);
    });
  }

  return { render, setPlaying };
}
