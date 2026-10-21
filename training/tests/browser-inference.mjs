import assert from 'node:assert/strict';
import fs from 'node:fs';
import { createHash } from 'node:crypto';
import { PicoboiEngine, InputError } from '../../web/dist/inference.js';

const base = new URL('../../web/dist/models/picoboi-100k-v1/', import.meta.url);
const manifest = JSON.parse(fs.readFileSync(new URL('manifest.json', base)));
const tokenizer = JSON.parse(fs.readFileSync(new URL('tokenizer.json', base)));
const weights = fs.readFileSync(new URL('weights.bin', base));
assert.equal(createHash('sha256').update(weights).digest('hex'), manifest.weights_sha256);
const engine = new PicoboiEngine(manifest, tokenizer, weights.buffer.slice(weights.byteOffset, weights.byteOffset + weights.byteLength));
const fixtures = JSON.parse(fs.readFileSync(new URL('./browser-parity.json', import.meta.url)));
let maxError = 0;
for (const fixture of fixtures) {
  engine.reset();
  let logits;
  for (const token of fixture.tokens) logits = engine.step(token);
  for (let i = 0; i < logits.length; i++) {
    const error = Math.abs(logits[i] - fixture.logits[i]);
    maxError = Math.max(maxError, error);
    assert.ok(error < 0.001, `PyTorch parity error ${error} at logit ${i}`);
  }
  const result = await engine.generate([{ role: 'user', content: fixture.prompt }]);
  assert.equal(result.text, fixture.answer);
}
assert.throws(() => engine.tokenize('spaceship'), InputError);
assert.throws(() => engine.tokenize('2 apples'), InputError);
assert.throws(() => engine.prepare([{ role: 'user', content: 'hello '.repeat(100) }]), InputError);
const history = [];
for (let i = 0; i < 20; i++) history.push({ role: i % 2 ? 'assistant' : 'user', content: i % 2 ? 'the ball is on the table.' : fixtures[0].prompt });
history.push({ role: 'user', content: 'what color is it?' });
const trimmed = engine.prepare(history);
assert.ok(trimmed.dropped > 0 && trimmed.dropped % 2 === 0);
assert.ok(trimmed.tokens.length + 32 <= 128);
assert.equal(history.length, 21);
await assert.rejects(engine.generate([{role:'user',content:fixtures[0].prompt}],()=>{},()=>true), /Cancelled/);
assert.equal((await engine.generate([{role:'user',content:fixtures[0].prompt}])).text, fixtures[0].answer);
console.log(JSON.stringify({status:'passed', pytorch_logit_max_error:maxError, greedy_parity:fixtures.length,
  checks:['export checksum','per-stage KV caches','vocabulary rejection','context trimming','cancellation and reuse']}));
