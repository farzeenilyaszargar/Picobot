// The tokenizer should encode text segments as ordinary text and slot segments as reserved IDs.
export function createCitySlots(aliases) {
  const entries = Object.keys(aliases).sort((a, b) => b.length - a.length);
  const names = entries.map(name => name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|');
  const protectedText = '```[\\s\\S]*?(?:```|$)|`[^`\\n]*`|"[^"\\n]*"|“[^”\\n]*”';
  const pattern = entries.length ? new RegExp(
    `${protectedText}|(?<![\\p{L}\\p{N}_])(?:${names})(?![\\p{L}\\p{N}_])`, 'gu'
  ) : null;
  const byIdentity = new Map();
  const bindings = [];

  function encode(text, role) {
    if (!pattern) return [{ text }];
    const segments = [];
    let cursor = 0;
    for (const match of text.matchAll(pattern)) {
      const [name] = match;
      const start = match.index;
      if (!Object.hasOwn(aliases, name)) continue;
      if (name === 'Paris' && !/\b(?:in|from|to|via|visit|leave|reach)\s+$/i.test(text.slice(0, start))) continue;
      const identity = aliases[name];
      let slot = byIdentity.get(identity);
      if (!slot && (role === 'user' || role === 'tool') && bindings.length < 8) {
        slot = bindings.push({ identity, name, start, end: start + name.length });
        byIdentity.set(identity, slot);
      }
      if (!slot) continue;
      if (start > cursor) segments.push({ text: text.slice(cursor, start) });
      segments.push({ slot });
      cursor = start + name.length;
    }
    if (cursor < text.length) segments.push({ text: text.slice(cursor) });
    return segments;
  }

  return {
    encode,
    resolve: slot => Number.isInteger(slot) && bindings[slot - 1] ? { ...bindings[slot - 1] } : null,
  };
}
