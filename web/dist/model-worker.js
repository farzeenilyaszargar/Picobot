import { PicoboiEngine } from './inference.js';
import { createNormalizer } from './normalize.js';

let engine, normalize, active = 0;
async function load() {
  const base = new URL('./models/picoboi-100k-v1/', import.meta.url);
  const get = async (path) => {
    const response = await fetch(new URL(path, base));
    if (!response.ok) throw new Error('Could not download the model. Check your connection and reload.');
    return response;
  };
  const [manifest, tokenizer] = await Promise.all([
    get('manifest.json').then(r => r.json()), get('tokenizer.json').then(r => r.json()),
  ]);
  const weights = await get(manifest.weights).then(r => r.arrayBuffer());
  const hash = [...new Uint8Array(await crypto.subtle.digest('SHA-256', weights))].map(n => n.toString(16).padStart(2, '0')).join('');
  if (hash !== manifest.weights_sha256) throw new Error('The model download was incomplete. Please reload.');
  engine = new PicoboiEngine(manifest, tokenizer, weights);
  normalize = createNormalizer(tokenizer);
  self.postMessage({ type: 'ready' });
}
const ready = load().catch(error => { self.postMessage({ type: 'load-error', message: error.message }); throw error; });
// The catch above reports the load failure; consume its rejection until a request arrives.
ready.catch(() => {});
self.onmessage = async ({ data }) => {
  if (data.type === 'cancel') { active = data.id; return; }
  if (data.type !== 'generate') return;
  const id = data.id;
  active = id;
  try {
    await ready;
    const messages = data.messages.map(message => ({ ...message }));
    const latest = messages.at(-1);
    const normalized = normalize(latest.content);
    if (normalized.changed) self.postMessage({ type: 'normalized', id, text: normalized.text });
    if (normalized.unsupported.length) {
      const hints = normalized.suggestions.filter(s => s.candidates.length).map(s => `${s.word}: ${s.candidates.join(' or ')}`);
      throw new Error(`Outside the 800-word vocabulary: ${normalized.unsupported.slice(0, 8).join(', ')}.${hints.length ? ` Did you mean ${hints.join('; ')}?` : ' Try simpler words.'}`);
    }
    latest.content = normalized.text;
    const result = await engine.generate(messages,
      text => { if (active === id) self.postMessage({ type: 'token', id, text }); },
      () => active !== id);
    if (active === id) self.postMessage({ type: 'done', id, ...result, normalizedInput: normalized.text });
  } catch (error) {
    if (active === id) self.postMessage({ type: 'error', id, message: error.message });
  }
};
