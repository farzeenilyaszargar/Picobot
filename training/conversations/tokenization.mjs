// Browser-compatible codecs for the next model. Tokenizer choice belongs to its weights.
export function createTokenizer(spec) {
  const tokens = spec.tokens;
  const ids = new Map(tokens.map((token, i) => [token, i]));
  if (ids.size !== tokens.length) throw new Error('Duplicate token strings');
  const bpe = spec.type === 'bpe';
  const reserved = new Set(spec.reserved_token_names || []);
  const references = new Set(spec.reference_tokens);
  const roles = new Set(['<pad>', '<bos>', '<eos>', '<user>', '<assistant>', '<turn>']);
  const cache = new Map();
  function pieces(word) {
    if (!/^[a-z]+$/.test(word)) throw new Error('Expected a lowercase ASCII word');
    if (!bpe) {
      if (!ids.has(word)) throw new Error(`Word outside word-level vocabulary: ${word}`);
      return [word];
    }
    if (!cache.has(word)) {
      let parts = [...word, spec.end_of_word];
      for (const [left, right] of spec.merges) {
        const merged = [];
        for (let i = 0; i < parts.length; i++) {
          if (parts[i] === left && parts[i + 1] === right) {
            merged.push(left + right);
            i++;
          } else merged.push(parts[i]);
        }
        parts = merged;
      }
      if (parts.some(part => !ids.has(part))) throw new Error('Unsupported alphabet');
      cache.set(word, parts);
    }
    return [...cache.get(word)];
  }
  function encode(text) {
    const parts = text.match(/<term_\d+>|[a-z]+|[.,?!:]/g) || [];
    if (text !== text.toLowerCase() || parts.join('') !== text.replace(/\s/g, '')) {
      throw new Error('Expected lowercase canonical text, punctuation, and references only');
    }
    return parts.flatMap(part => {
      if (/^[a-z]+$/.test(part)) return pieces(part).map(piece => ids.get(piece));
      if (!references.has(part) && !/^[.,?!:]$/.test(part)) throw new Error('Unknown reference');
      return [ids.get(part)];
    });
  }
  function decode(sequence, skipRoleTokens = false) {
    const chunks = [];
    let pending = '';
    for (const index of sequence) {
      if (!Number.isInteger(index) || index < 0 || index >= tokens.length) throw new Error('Invalid token ID');
      const token = tokens[index];
      if (bpe && !reserved.has(token)) {
        pending += token;
        if (pending.endsWith(spec.end_of_word)) {
          chunks.push(pending.slice(0, -spec.end_of_word.length));
          pending = '';
        }
      } else {
        if (pending) { chunks.push(pending); pending = ''; }
        if (!skipRoleTokens || !roles.has(token)) chunks.push(token);
      }
    }
    if (pending) chunks.push(pending);
    return chunks.join(' ').replace(/\s+([.,?!:])/g, '$1');
  }
  return { encode, decode, pieces };
}
