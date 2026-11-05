// Small CSS/SVG birds. The chat has no dependency on this decorative layer.
const toggle = document.getElementById('sky-toggle');
const label = document.getElementById('sky-toggle-label');
const motion = window.matchMedia('(prefers-reduced-motion: reduce)');
let paused = false;

function syncSky() {
  document.body.classList.toggle('sky-paused', paused);
  document.body.classList.toggle('sky-idle', document.hidden);
  toggle.hidden = motion.matches;
  toggle.setAttribute('aria-pressed', String(paused));
  const text = paused ? 'Play birds' : 'Pause birds';
  toggle.setAttribute('aria-label', text);
  label.textContent = text;
}

toggle.addEventListener('click', () => { paused = !paused; syncSky(); });
motion.addEventListener('change', syncSky);
document.addEventListener('visibilitychange', syncSky);
syncSky();
