// The ASCII treatment is baked into small, same-origin video assets.
// Reduced motion and data saver get the poster without downloading the video.
const layer = document.getElementById('ambient-background');
const video = document.getElementById('ambient-video');
const toggle = document.getElementById('ambient-toggle');
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
const mobileLayout = window.matchMedia('(max-width: 600px)');
const connection = navigator.connection;
let userPaused = false;
let loaded = false;
let loadedVariant;
let failed = false;

function syncBackground() {
  const onChatPage = location.hash !== '#about';
  layer.hidden = !onChatPage;
  const allowed = !reducedMotion.matches && !connection?.saveData && !failed;
  const variant = mobileLayout.matches ? 'mobileSrc' : 'src';
  if (loadedVariant !== variant) {
    loaded = false;
    layer.classList.remove('is-playing');
  }
  video.poster = mobileLayout.matches ? './media/drifters-ascii-mobile-poster.png' : './media/drifters-ascii-desktop-poster.png';
  toggle.hidden = !onChatPage || !allowed;
  toggle.setAttribute('aria-pressed', String(userPaused));
  toggle.setAttribute('aria-label', userPaused ? 'Play background animation' : 'Pause background animation');
  if (!onChatPage || !allowed || document.hidden || userPaused) {
    video.pause();
    return;
  }
  if (!loaded) {
    for (const source of video.querySelectorAll('source')) source.src = source.dataset[variant] || source.dataset.src;
    video.muted = true;
    video.load();
    loaded = true;
    loadedVariant = variant;
  }
  video.play().catch(() => {
    if (document.hidden || location.hash === '#about' || reducedMotion.matches || connection?.saveData || userPaused || failed || loadedVariant !== variant) return;
    // Autoplay can be blocked by battery-saving/browser policies. Keep a still.
    userPaused = true;
    toggle.setAttribute('aria-pressed', 'true');
    toggle.setAttribute('aria-label', 'Play background animation');
  });
}

video.addEventListener('playing', () => layer.classList.add('is-playing'));
video.addEventListener('error', () => {
  failed = true;
  layer.classList.remove('is-playing');
  syncBackground();
});
toggle.addEventListener('click', () => {
  userPaused = !userPaused;
  syncBackground();
});
reducedMotion.addEventListener('change', () => {
  // Restore the deliberately chosen poster when motion is disabled mid-play.
  if (reducedMotion.matches) layer.classList.remove('is-playing');
  syncBackground();
});
connection?.addEventListener('change', syncBackground);
mobileLayout.addEventListener('change', syncBackground);
document.addEventListener('visibilitychange', syncBackground);
window.addEventListener('hashchange', syncBackground);
syncBackground();
