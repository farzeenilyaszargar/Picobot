// Decorative, browser-local ASCII. No images, network requests, or saved state.
const canvas = document.getElementById('ascii-background');
const context = canvas?.getContext('2d');

if (context) {
  const glyphs = ['.', ':', '+', '-', '0', '1', '/', '~', '*'];
  const motion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const tau = Math.PI * 2;
  let width = 0;
  let height = 0;
  let particles = [];
  let frame = null;
  let lastTick = null;
  let lastPaint = -Infinity;
  let elapsed = 0;

  const wrap = (value, limit) => ((value % limit) + limit) % limit;
  const smoothstep = (low, high, value) => {
    const t = Math.max(0, Math.min(1, (value - low) / (high - low)));
    return t * t * (3 - 2 * t);
  };

  function resize() {
    width = document.documentElement.clientWidth;
    height = window.innerHeight;
    const ratio = Math.min(window.devicePixelRatio || 1, 1.5);
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.font = '500 13px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';
    context.textAlign = 'center';
    context.textBaseline = 'middle';
    const count = Math.min(900, Math.max(100, Math.round(width * height / 1300)));
    particles = Array.from({ length: count }, () => ({
      x: Math.random(),
      y: Math.random(),
      phase: Math.random() * tau,
      speed: 2 + Math.random() * 3,
      drift: (Math.random() - .5) * 4,
      opacity: .2 + Math.random() * .24,
      glyph: Math.floor(Math.random() * glyphs.length),
      changeEvery: 5 + Math.random() * 9,
    }));
    draw(elapsed);
  }

  function draw(time) {
    context.clearRect(0, 0, width, height);
    for (const particle of particles) {
      const x = wrap(particle.x * (width + 60) + time * particle.drift +
        Math.sin(time * .09 + particle.phase) * 22, width + 60) - 30;
      const y = wrap(particle.y * (height + 60) - time * particle.speed +
        Math.cos(time * .07 + particle.phase) * 16, height + 60) - 30;

      // Keep the middle quiet, with a gradual falloff towards the margins.
      const distance = Math.hypot((x - width / 2) / (width * .44),
        (y - height * .48) / (height * .43));
      const clearing = .1 + .9 * smoothstep(.4, 1.1, distance);
      const edge = Math.min(1, Math.max(0, Math.min(x, width - x, y, height - y) / 35));
      // Glyph changes happen at zero opacity, preventing visible flicker.
      const progress = time / particle.changeEvery + particle.phase / tau;
      const fade = Math.sin((progress % 1) * Math.PI) ** 2;
      const alpha = particle.opacity * clearing * edge * fade;
      if (alpha < .006) continue;
      const glyph = glyphs[(particle.glyph + Math.floor(progress)) % glyphs.length];
      context.fillStyle = `rgba(110, 117, 104, ${alpha})`;
      context.fillText(glyph, x, y);
    }
  }

  function animate(now) {
    if (lastTick !== null) elapsed += Math.min((now - lastTick) / 1000, .1);
    lastTick = now;
    if (now - lastPaint >= 1000 / 18) {
      draw(elapsed);
      lastPaint = now;
    }
    frame = window.requestAnimationFrame(animate);
  }

  function syncAnimation() {
    if (frame !== null) window.cancelAnimationFrame(frame);
    frame = null;
    lastTick = null;
    if (!document.hidden && !motion.matches) {
      frame = window.requestAnimationFrame(animate);
    } else if (!document.hidden) {
      draw(elapsed);
    }
  }

  window.addEventListener('resize', resize, { passive: true });
  document.addEventListener('visibilitychange', syncAnimation);
  motion.addEventListener('change', syncAnimation);
  resize();
  syncAnimation();
}
