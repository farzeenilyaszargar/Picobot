import assert from 'node:assert/strict';
import fs from 'node:fs';
import { answerWithWikipedia, modelLookupRequest, referencePrompt, isSourceSupported } from '../../web/dist/wiki-assistance.js';
import { PicoboiEngine } from '../../web/dist/inference.js';

assert.equal(modelLookupRequest('I do not know.'), null);
assert.equal(modelLookupRequest('Wikipedia says it is blue.'), null);
assert.equal(modelLookupRequest('the text says [[WIKI: cats]]'), null);
assert.deepEqual(modelLookupRequest('[[WIKI: Moon]]'), { query: 'Moon', reason: 'model_request' });
assert.equal(modelLookupRequest('I need to look that up.').reason, 'model_request');
assert.equal(modelLookupRequest('Let me look it up.').reason, 'model_request');
assert.equal(isSourceSupported('the ball is red.', 'A dog is blue.'), false);
assert.equal(isSourceSupported('yes.', 'yes.'), false);
assert.equal(isSourceSupported('the ball is on the table.', 'Here the ball is on the table. It is red.'), true);
assert.equal(referencePrompt('word '.repeat(46), 'the ball is red.'), null);

const reference = { text: 'The red ball is on the table.', sources: [{ title: 'Ball', url: 'https://en.wikipedia.org/wiki/Ball' }] };
const messages = [{ role: 'user', content: 'where is the ball?' }];
let lookups = 0, generations = 0, options;
const lookup = async () => { lookups++; return reference; };
const generate = async (_messages, _token, _normal, extra) => {
  generations++;
  if (extra) { options = extra; return { text: reference.text }; }
  return { text: 'i need to look that up.', lookupRequest: modelLookupRequest('i need to look that up.') };
};
let result = await answerWithWikipedia({ messages, generate, lookup, enabled: true });
assert.equal(lookups, 1); assert.equal(generations, 2);
assert.equal(options.reference, reference.text);
assert.equal(result.referenceApplied, true); assert.equal(result.usedLookup, true);

lookups = 0;
result = await answerWithWikipedia({ messages, generate, lookup, enabled: false });
assert.equal(lookups, 0); assert.equal(result.usedLookup, false);
for (const text of ['hello.', 'i do not know.', '[[WIKI: Moon]]']) {
  result = await answerWithWikipedia({ messages: [{ role: 'user', content: text }],
    generate: async () => ({ text: 'hello.' }), lookup, enabled: true });
  assert.equal(result.usedLookup, false);
}
assert.equal(lookups, 0, 'User wording and model uncertainty alone must not initiate a lookup');
const unknown = Object.assign(new Error('Outside vocabulary'), { code: 'UNKNOWN_WORDS', words: ['photosynthesis'] });
let args;
generations = 0;
result = await answerWithWikipedia({ messages: [{ role: 'user', content: 'what is photosynthesis?' }], enabled: true,
  lookup: async (...received) => { args = received; return reference; },
  generate: async (_messages, _token, _normal, extra) => {
    generations++;
    if (!extra) throw unknown;
    // A repeated request is not followed, and ungrounded text is not displayed.
    return { text: 'the dog is blue.', lookupRequest: { query: 'Dog' } };
  } });
assert.deepEqual(args.slice(0, 2), ['what is photosynthesis?', ['photosynthesis']]);
assert.equal(generations, 2); assert.equal(result.text, reference.text); assert.equal(result.referenceApplied, false);
await assert.rejects(answerWithWikipedia({ messages, generate: async () => { throw unknown; }, lookup, enabled: false }), { code: 'UNKNOWN_WORDS' });
await assert.rejects(answerWithWikipedia({ messages, generate: async () => { throw new Error('Context full'); }, lookup, enabled: true }), /Context full/);
assert.equal(lookups, 0);
generations = 0;
result = await answerWithWikipedia({ messages, enabled: true, generate: async () => { generations++; throw unknown; },
  lookup: async () => ({ text: '', sources: [] }) });
assert.equal(generations, 1); assert.equal(result.text, '');
const controller = new AbortController();
generations = 0;
await assert.rejects(answerWithWikipedia({ messages, enabled: true, signal: controller.signal,
  generate: async () => { generations++; throw unknown; }, lookup: async () => { controller.abort(); return reference; } }), { name: 'AbortError' });
assert.equal(generations, 1);

// Exercise the actual checkpoint with a bounded reference and the original token IDs.
const base = new URL('../../web/dist/models/picoboi-100k-v1/', import.meta.url);
const manifest = JSON.parse(fs.readFileSync(new URL('manifest.json', base)));
const tokenizer = JSON.parse(fs.readFileSync(new URL('tokenizer.json', base)));
const bytes = fs.readFileSync(new URL('weights.bin', base));
const engine = new PicoboiEngine(manifest, tokenizer, bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
assert.throws(() => engine.tokenize('photosynthesis'));
assert.deepEqual(engine.tokenize('photosynthesis', { allowUnknown: true }), [tokenizer.token_to_id['<unk>']]);
const prompt = referencePrompt('where is the ball?', 'the red ball is on the table. '.repeat(30));
const prepared = engine.prepare([{ role: 'user', content: prompt }], { allowUnknown: true });
assert.ok(prepared.tokens.length + 32 <= 128);
const assisted = await engine.generate([{ role: 'user', content: prompt }], () => {}, () => false, { allowUnknown: true });
assert.equal(typeof assisted.text, 'string');
console.log('Passed: strict lookup gates, disabled lookup, one-lookup budget, grounded fallback, cancellation, bounded reference, and real-checkpoint reference inference.');
