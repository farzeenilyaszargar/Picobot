// Suspend decorative animation while the page is hidden. Reduced motion is handled by CSS.
function syncSky() {
  document.body.classList.toggle('sky-idle', document.hidden);
}
document.addEventListener('visibilitychange', syncSky);
syncSky();
