import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

const script = fs.readFileSync(new URL('../../web/dist/ambient-background.js', import.meta.url), 'utf8');
function fixture({ reduced = false, saveData = false, about = false, blocked = false, mobile = false } = {}) {
  const element = () => ({ hidden: false, attributes: {}, events: {},
    setAttribute(key,value) { this.attributes[key] = value; },
    addEventListener(key, fn) { this.events[key] = fn; },
  });
  const classes = new Set();
  const layer = { ...element(), classList: { add: c => classes.add(c), remove: c => classes.delete(c) } };
  const sources = [{ dataset: { src: 'loop.webm', mobileSrc: 'mobile.webm' } }, { dataset: { src: 'loop.mp4', mobileSrc: 'mobile.mp4' } }];
  const video = { ...element(), paused: true, loads: 0, plays: 0,
    querySelectorAll: () => sources,
    load() { this.loads++; },
    pause() { this.paused = true; },
    play() { this.plays++; this.paused = blocked; if (!blocked) this.events.playing?.(); return blocked ? Promise.reject(new Error('Autoplay denied')) : Promise.resolve(); },
  };
  const toggle = element();
  const motion = { ...element(), matches: reduced };
  const layout = { ...element(), matches: mobile };
  const connection = { ...element(), saveData };
  const document = { ...element(), hidden: false,
    getElementById: id => ({ 'ambient-background': layer, 'ambient-video': video, 'ambient-toggle': toggle })[id] };
  const window = { ...element(), matchMedia: query => query.includes('reduced-motion') ? motion : layout };
  const location = { hash: about ? '#about' : '' };
  vm.runInNewContext(script, { window, document, navigator: { connection }, location });
  return { layer,video,toggle,sources,motion,layout,connection,document,window,location,classes };
}
const normal = fixture();
assert.equal(normal.video.loads, 1);
assert.equal(normal.video.paused, false);
assert.equal(normal.sources[0].src, 'loop.webm');
assert.ok(normal.classes.has('is-playing'));
normal.toggle.events.click();
assert.equal(normal.video.paused, true);
assert.equal(normal.toggle.attributes['aria-label'], 'Play background animation');
normal.toggle.events.click();
assert.equal(normal.video.paused, false);
assert.equal(normal.video.loads, 1);
normal.location.hash = '#about'; normal.window.events.hashchange();
assert.equal(normal.layer.hidden, true);
assert.equal(normal.video.paused, true);
normal.location.hash = ''; normal.window.events.hashchange();
assert.equal(normal.layer.hidden, false);
assert.equal(normal.video.paused, false);
normal.document.hidden = true; normal.document.events.visibilitychange();
assert.equal(normal.video.paused, true);
normal.document.hidden = false; normal.document.events.visibilitychange();
assert.equal(normal.video.paused, false);
normal.motion.matches = true; normal.motion.events.change();
assert.equal(normal.video.paused, true);
assert.equal(normal.classes.has('is-playing'), false);
for (const options of [{ reduced:true }, { saveData:true }, { about:true }]) {
  const still = fixture(options);
  assert.equal(still.video.loads, 0);
  assert.equal(still.sources[0].src, undefined);
  assert.equal(still.toggle.hidden, true);
}
const portrait = fixture({ mobile:true });
assert.equal(portrait.sources[0].src, 'mobile.webm');
assert.match(portrait.video.poster, /mobile-poster/);
portrait.layout.matches = false; portrait.layout.events.change();
assert.equal(portrait.sources[0].src, 'loop.webm');
assert.equal(portrait.video.loads, 2);
assert.match(portrait.video.poster, /desktop-poster/);
portrait.toggle.events.click();
portrait.layout.matches = true; portrait.layout.events.change();
assert.equal(portrait.video.paused, true);
assert.equal(portrait.classes.has('is-playing'), false);
assert.equal(portrait.video.loads, 2);
portrait.toggle.events.click();
assert.equal(portrait.sources[0].src, 'mobile.webm');
assert.equal(portrait.video.loads, 3);
const failed = fixture();
failed.video.events.error();
assert.equal(failed.classes.has('is-playing'), false);
assert.equal(failed.toggle.hidden, true);
assert.equal(failed.video.paused, true);
const blocked = fixture({ blocked:true });
await Promise.resolve();
assert.equal(blocked.toggle.attributes['aria-label'], 'Play background animation');
console.log('Passed: source loading, playback, pause/resume, route and visibility suspension, reduced motion, data saver, media failure, autoplay denial, mobile source selection, breakpoint switching, paused layout changes.');
