// CPU inference with a separate attention cache for each executed block.
// Every generated token comes from the exported checkpoint's logits.
export class InputError extends Error {}

export class PicoboiEngine {
  constructor(manifest, tokenizer, buffer) {
    const c = manifest.config;
    if (manifest.format !== 'picoboi-f32-le-v1' || buffer.byteLength !== manifest.weights_bytes ||
        manifest.parameters !== 99120 || c.d_model !== 48 || c.n_heads !== 2 || c.vocab_size !== tokenizer.tokens.length) {
      throw new Error('The model assets do not match. Please reload the page.');
    }
    this.config = c;
    this.tokenizer = tokenizer;
    this.ids = tokenizer.token_to_id;
    this.weights = {};
    const view = new DataView(buffer);
    for (const [name, tensor] of Object.entries(manifest.tensors)) {
      const values = new Float32Array(tensor.length);
      for (let i = 0; i < values.length; i++) values[i] = view.getFloat32((tensor.offset + i) * 4, true);
      this.weights[name] = values;
    }
    this.reset();
  }

  reset() {
    this.position = 0;
    this.caches = this.config.block_order.map(() => ({
      keys: new Float32Array(this.config.train_context * this.config.d_model),
      values: new Float32Array(this.config.train_context * this.config.d_model),
    }));
  }

  linear(x, name) {
    const w = this.weights[name];
    const rows = w.length / x.length;
    const out = new Float32Array(rows);
    for (let row = 0; row < rows; row++) {
      let sum = 0;
      const offset = row * x.length;
      for (let j = 0; j < x.length; j++) sum += w[offset + j] * x[j];
      out[row] = sum;
    }
    return out;
  }

  norm(x, name) {
    let mean = 0;
    for (const value of x) mean += value * value;
    const scale = 1 / Math.sqrt(mean / x.length + this.config.norm_epsilon);
    const w = this.weights[name];
    return Float32Array.from(x, (value, i) => value * scale * w[i]);
  }

  rotate(x) {
    const dim = this.config.d_model / this.config.n_heads;
    for (let head = 0; head < this.config.n_heads; head++) {
      for (let j = 0; j < dim; j += 2) {
        const angle = this.position * Math.pow(this.config.rope_theta, -j / dim);
        const cos = Math.cos(angle), sin = Math.sin(angle), i = head * dim + j;
        const even = x[i], odd = x[i + 1];
        x[i] = even * cos - odd * sin;
        x[i + 1] = even * sin + odd * cos;
      }
    }
    return x;
  }

  step(token) {
    const c = this.config, d = c.d_model, dim = d / c.n_heads;
    if (!Number.isInteger(token) || token < 0 || token >= c.vocab_size) throw new Error('Invalid token');
    if (this.position >= c.train_context) throw new InputError('This chat has reached the 128-token context limit. Start a new chat.');
    let x = this.weights['embedding.weight'].slice(token * d, (token + 1) * d);
    c.block_order.forEach((block, stage) => {
      const prefix = `blocks.${block}.`;
      const normalized = this.norm(x, prefix + 'attention_norm.weight');
      const q = this.rotate(this.linear(normalized, prefix + 'q.weight'));
      const k = this.rotate(this.linear(normalized, prefix + 'k.weight'));
      const v = this.linear(normalized, prefix + 'v.weight');
      const cache = this.caches[stage];
      cache.keys.set(k, this.position * d);
      cache.values.set(v, this.position * d);
      const attended = new Float32Array(d);
      for (let head = 0; head < c.n_heads; head++) {
        const scores = new Float32Array(this.position + 1);
        let max = -Infinity;
        for (let t = 0; t <= this.position; t++) {
          let score = 0;
          for (let j = 0; j < dim; j++) score += q[head * dim + j] * cache.keys[t * d + head * dim + j];
          scores[t] = score / Math.sqrt(dim);
          max = Math.max(max, scores[t]);
        }
        let total = 0;
        for (let t = 0; t < scores.length; t++) { scores[t] = Math.exp(scores[t] - max); total += scores[t]; }
        for (let j = 0; j < dim; j++) {
          let sum = 0;
          for (let t = 0; t < scores.length; t++) sum += scores[t] / total * cache.values[t * d + head * dim + j];
          attended[head * dim + j] = sum;
        }
      }
      const attention = this.linear(attended, prefix + 'o.weight');
      x = Float32Array.from(x, (value, i) => value + attention[i]);
      const z = this.norm(x, prefix + 'ffn_norm.weight');
      const gate = this.linear(z, prefix + 'gate.weight');
      const up = this.linear(z, prefix + 'up.weight');
      for (let i = 0; i < gate.length; i++) gate[i] = gate[i] / (1 + Math.exp(-gate[i])) * up[i];
      const down = this.linear(gate, prefix + 'down.weight');
      x = Float32Array.from(x, (value, i) => value + down[i]);
    });
    this.position++;
    return this.linear(this.norm(x, 'final_norm.weight'), 'embedding.weight');
  }

  tokenize(text) {
    const normalized = text.toLowerCase().trim();
    if (!normalized) throw new InputError('Write a message first.');
    const pieces = normalized.match(/[a-z]+|[.,?!:]/g) || [];
    if (pieces.join('') !== normalized.replace(/\s/g, '')) {
      throw new InputError('This model accepts English words and . , ? ! : punctuation. Spell numbers as words.');
    }
    const unknown = [...new Set(pieces.filter(word => !Object.hasOwn(this.ids, word)))];
    if (unknown.length) throw new InputError(`Outside this model’s 800-word vocabulary: ${unknown.slice(0, 8).join(', ')}. Try simpler words or one of the example prompts.`);
    return pieces.map(piece => this.ids[piece]);
  }

  prepare(messages) {
    if (!Array.isArray(messages) || !messages.length || messages.length % 2 !== 1) throw new InputError('Start a new chat and try again.');
    let turns = messages.map((message, i) => {
      if (message.role !== (i % 2 ? 'assistant' : 'user')) throw new InputError('Start a new chat and try again.');
      return [this.ids[`<${message.role}>`], ...this.tokenize(message.content), this.ids['<turn>']];
    });
    const reserve = 32;
    const size = () => 2 + turns.reduce((total, turn) => total + turn.length, 0);
    let dropped = 0;
    while (size() + reserve > this.config.train_context && turns.length > 1) {
      turns = turns.slice(2); dropped += 2;
    }
    if (size() + reserve > this.config.train_context) throw new InputError('That message is too long. Keep it under about 85 words and punctuation marks.');
    return { tokens: [this.ids['<bos>'], ...turns.flat(), this.ids['<assistant>']], dropped };
  }

  async generate(messages, onToken = () => {}, cancelled = () => false) {
    const { tokens, dropped } = this.prepare(messages);
    this.reset();
    let logits;
    for (let i = 0; i < tokens.length; i++) {
      if (cancelled()) throw new Error('Cancelled');
      logits = this.step(tokens[i]);
      if (i % 8 === 0) await new Promise(resolve => setTimeout(resolve, 0));
    }
    const forbidden = new Set(['<pad>', '<unk>', '<bos>', '<user>', '<assistant>'].map(t => this.ids[t]));
    const generated = [];
    let ended = false;
    const limit = Math.min(48, this.config.train_context - tokens.length);
    for (let n = 0; n < limit; n++) {
      if (cancelled()) throw new Error('Cancelled');
      let next = -1;
      for (let i = 0; i < logits.length; i++) if (!forbidden.has(i) && (next < 0 || logits[i] > logits[next])) next = i;
      if (next === this.ids['<turn>'] || next === this.ids['<eos>']) { ended = true; break; }
      generated.push(this.tokenizer.tokens[next]);
      const text = generated.join(' ').replace(/\s+([.,?!:])/g, '$1');
      onToken(text);
      logits = this.step(next);
      await new Promise(resolve => setTimeout(resolve, 0));
    }
    return { text: generated.join(' ').replace(/\s+([.,?!:])/g, '$1'), dropped, truncated: !ended };
  }
}
