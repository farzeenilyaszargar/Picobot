// GPT-2 ByteLevel alphabet and pre-tokenization, as used by the exported tokenizer.
const bytes = new Map();
let extra = 256;
for (let byte = 0; byte < 256; byte++) {
  const printable = (byte >= 33 && byte <= 126) || (byte >= 161 && byte <= 172) || byte >= 174;
  bytes.set(byte, String.fromCharCode(printable ? byte : extra++));
}
const inverse = new Map([...bytes].map(([byte, char]) => [char, byte]));
const pieces = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

export function createByteBpe(tokenizer) {
  const ids = tokenizer.model.vocab;
  const tokens = [];
  for (const [token, id] of Object.entries(ids)) tokens[id] = token;
  const ranks = new Map(tokenizer.model.merges.map((pair, rank) => [pair.join(' '), rank]));
  const encoder = new TextEncoder();
  const encode = (text) => (text.match(pieces) ?? []).flatMap(piece => {
    let symbols = Array.from(encoder.encode(piece), byte => bytes.get(byte));
    while (symbols.length > 1) {
      let best = Infinity;
      let selected;
      for (let i = 0; i < symbols.length - 1; i++) {
        const key = symbols[i] + ' ' + symbols[i + 1];
        const rank = ranks.get(key);
        if (rank !== undefined && rank < best) { best = rank; selected = key; }
      }
      if (selected === undefined) break;
      const merged = [];
      for (let i = 0; i < symbols.length; i++) {
        if (symbols[i] + ' ' + symbols[i + 1] === selected) merged.push(symbols[i] + symbols[++i]);
        else merged.push(symbols[i]);
      }
      symbols = merged;
    }
    return symbols.map(symbol => ids[symbol]);
  });
  const decode = (sequence, streaming) => new TextDecoder().decode(
    Uint8Array.from(sequence.flatMap(id => Array.from(tokens[id], char => inverse.get(char)))),
    { stream: streaming },
  );
  return { token_to_id: ids, tokens, encode, decode };
}
