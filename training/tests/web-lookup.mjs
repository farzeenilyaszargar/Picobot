import assert from 'node:assert/strict';
import { lookupQuery, lookupWeb } from '../../web/dist/web-lookup.js';

assert.equal(lookupQuery('what is photosynthesis?'), 'photosynthesis');
assert.equal(lookupQuery('please tell me about quantum computing'), 'quantum computing');
assert.equal(lookupQuery('what is the capital of France?'), 'capital France');
assert.equal(lookupQuery('Tell me about São Paulo'), 'São Paulo');
assert.ok(lookupQuery('x'.repeat(1000)).length <= 240);
const calls = [];
const fetchImpl = async (url, options) => {
  calls.push({ url: new URL(url), options });
  return { ok: true, json: async () => ({ query: { pages: [
    { index: 2, title: 'Related topic', extract: 'Another result.' },
    { index: 1, title: 'Quantum computing', extract: 'A quantum computer uses quantum states.' },
    { index: 3, title: 'Empty result', extract: '' },
  ] } }) };
};
const answer = await lookupWeb('what is quantum computing?', ['quantum', 'computing'], { fetchImpl });
assert.equal(answer.sources[0].title, 'Quantum computing');
assert.equal(answer.text, 'A quantum computer uses quantum states.');
assert.equal(calls[0].url.searchParams.get('gsrsearch'), 'quantum computing');
assert.equal(calls[0].url.searchParams.get('explaintext'), '1');
assert.equal(calls[0].options.credentials, 'omit');
assert.equal(calls[0].options.referrerPolicy, 'no-referrer');
assert.match(answer.sources[0].url, /^https:\/\/en.wikipedia.org\/wiki\//);
let attempts = 0;
const empty = await lookupWeb('explain quantum physics', ['quantum'], { fetchImpl: async () => {
  attempts++;
  return { ok: true, json: async () => ({}) };
} });
assert.equal(attempts, 2);
assert.equal(empty.text, '');
assert.deepEqual(empty.sources, []);
await assert.rejects(lookupWeb('quantum', [], { fetchImpl: async () => ({ ok: false }) }), /unavailable/);
await assert.rejects(lookupWeb('quantum', [], { fetchImpl: async () => ({ ok: true, json: async () => ({ error: {} }) }) }), /shorter topic/);
const controller = new AbortController();
controller.abort();
await assert.rejects(lookupWeb('quantum', [], { signal: controller.signal, fetchImpl: async (_, {signal}) => {
  signal.throwIfAborted();
} }), { name: 'AbortError' });
console.log('Passed: query extraction, ranked sources, plain text, privacy options, empty results, bounded retry, service errors, cancellation.');
