// Only model-produced requests and the worker's vocabulary error authorize lookup.
// Never parse user text or Wikipedia extracts as tool instructions.
export function modelLookupRequest(text) {
  const marker = /^\s*\[\[WIKI:\s*([^\]\n]{1,240})\]\]\s*$/i.exec(text);
  if (marker) return { query: marker[1].trim(), reason: 'model_request' };
  if (/^\s*(?:i (?:need|want) to|let me) look (?:it|that|this) up[.!]?\s*$/i.test(text)) {
    return { reason: 'model_request' };
  }
  return null;
}

const pieces = text => String(text).toLowerCase().match(/[a-z]+|\d+|[.,?!:]/g) || [];

export function referencePrompt(question, excerpt) {
  const query = pieces(question);
  // Leave room for the role tokens and a 32-token answer in the 128-token model.
  if (!query.length || query.length > 45) return null;
  const context = pieces(excerpt).slice(0, 88 - query.length - 3);
  if (!context.length) return null;
  return `read this: ${context.join(' ')} ${query.join(' ')}`;
}

export function isSourceSupported(answer, excerpt) {
  const words = text => pieces(text).filter(piece => /^[a-z0-9]+$/.test(piece));
  const candidate = words(answer);
  if (candidate.length < 4) return false;
  return ` ${words(excerpt).join(' ')} `.includes(` ${candidate.join(' ')} `);
}

export async function answerWithWikipedia({ messages, generate, lookup, enabled,
  onToken = () => {}, onNormalized = () => {}, onLookup = () => {}, signal }) {
  const check = () => signal?.throwIfAborted();
  const question = messages.at(-1).content;
  let initial, request, vocabularyError;
  check();
  try {
    initial = await generate(messages, onToken, onNormalized);
    request = initial.lookupRequest;
  } catch (error) {
    check();
    if (error.code !== 'UNKNOWN_WORDS') throw error;
    vocabularyError = error;
    request = { reason: 'unknown_words', words: error.words || [] };
  }
  check();
  if (!request || !enabled) {
    if (vocabularyError) throw vocabularyError;
    return { ...initial, usedLookup: false };
  }
  onLookup(request);
  // One lookup per turn. The assisted generation cannot initiate another one.
  const reference = await lookup(request.query || question, request.words || [], { signal });
  check();
  if (!reference.text) return { text: '', sources: [], usedLookup: true, referenceApplied: false };
  let assisted;
  try {
    assisted = await generate(messages, () => {}, () => {}, { reference: reference.text });
  } catch {
    // An incompatible or failed reference pass falls back to the attributed text.
    check();
  }
  check();
  const supported = typeof assisted?.text === 'string' && isSourceSupported(assisted.text, reference.text);
  return {
    text: supported ? assisted.text : reference.text,
    sources: reference.sources,
    usedLookup: true,
    referenceApplied: supported,
  };
}
