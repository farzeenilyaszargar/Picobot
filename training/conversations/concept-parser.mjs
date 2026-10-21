// Prototype for the next model. Nothing here changes the deployed 812-token model.
import { createNormalizer } from '../../web/dist/normalize.js';

const surfaceConcepts = Object.freeze({
  cat: 'animal', cats: 'animals', dog: 'animal', dogs: 'animals',
  lion: 'animal', lions: 'animals', tiger: 'animal', tigers: 'animals',
  horse: 'animal', horses: 'animals', rabbit: 'animal', rabbits: 'animals',
  kitten: 'animal', kittens: 'animals', puppy: 'animal', puppies: 'animals',
  hammer: 'tool', hammers: 'tools', screwdriver: 'tool', screwdrivers: 'tools',
  truck: 'vehicle', trucks: 'vehicles', bicycle: 'vehicle', bicycles: 'vehicles',
  rose: 'flower', roses: 'flowers', oak: 'tree', oaks: 'trees',
});
const referencePattern = /<term_\d+>/g;
const wordPattern = /[a-z]+(?:'[a-z]+)?/g;
const copy = binding => ({ ...binding });

export function createConceptParser(tokenizer) {
  const mappings = { ...surfaceConcepts, ...tokenizer.generalized_words };
  const vocabulary = new Set(tokenizer.word_vocabulary ?? tokenizer.tokens.filter(t => /^[a-z]+$/.test(t)));
  const slots = tokenizer.reference_tokens;
  if (!Array.isArray(slots) || !slots.length || new Set(slots).size !== slots.length ||
      slots.some((t, i) => t !== `<term_${i}>` || !tokenizer.tokens.includes(t))) {
    throw new Error('Concept parser needs a tokenizer with ordered reference slots');
  }
  for (const concept of Object.values(mappings)) {
    if (!vocabulary.has(concept)) throw new Error(`Concept outside model vocabulary: ${concept}`);
  }
  // Spelling repair may produce an intermediate surface such as apple before generalization.
  const normalize = createNormalizer({ ...tokenizer, tokens: [...vocabulary, ...Object.keys(mappings)] });
  let bindings = [];

  return {
    // A slot identifies a surface term, not a particular animal or physical object.
    // One parser instance belongs to one chat; do not reassign slots mid-chat.
    parse(originalInput) {
      if (typeof originalInput !== 'string') throw new TypeError('Input must be text');
      if (/<\s*\/?\s*term_\d+\s*>/i.test(originalInput)) {
        throw new Error('Reference tokens are reserved for the parser');
      }
      const normalized = normalize(originalInput);
      const candidate = bindings.map(copy);
      const changes = [...normalized.changes];
      const overflow = new Set();
      let modelText = normalized.text.replace(wordPattern, surface => {
        // Keep known words, inflections, and negation. Only explicit mappings generalize.
        if (vocabulary.has(surface) || !Object.hasOwn(mappings, surface)) return surface;
        let binding = candidate.find(item => item.surface === surface);
        if (!binding) {
          if (candidate.length === slots.length) { overflow.add(surface); return surface; }
          binding = { token: slots[candidate.length], surface, concept: mappings[surface] };
          candidate.push(binding);
        }
        const to = `${binding.concept} ${binding.token}`;
        changes.push({ from: surface, to, reason: 'concept reference' });
        return to;
      });
      modelText = modelText.replace(/\b(a|an)\s+([a-z]+\s+<term_\d+>)/g, (from, article, reference) => {
        const to = `${/^[aeiou]/.test(reference) ? 'an' : 'a'} ${reference}`;
        if (from !== to) changes.push({ from, to, reason: 'reference article agreement' });
        return to;
      });
      const withoutReferences = modelText.replace(referencePattern, '');
      const unsupported = [...new Set((withoutReferences.match(wordPattern) || [])
        .filter(word => !vocabulary.has(word)))];
      const unsupportedCharacters = [...new Set(withoutReferences.replace(/[a-z\s.,?!:]/g, ''))];
      const ready = !unsupported.length && !unsupportedCharacters.length;
      // Failed parses do not consume slots or change the bindings for the next turn.
      if (ready) bindings = candidate;
      return {
        originalInput, normalizedInput: normalized.text, modelText, ready,
        bindings: candidate.map(copy), unsupported, unsupportedCharacters,
        referenceCapacityExceeded: [...overflow], changes,
      };
    },

    render(modelText) {
      if (typeof modelText !== 'string') throw new TypeError('Output must be text');
      let rendered = modelText;
      for (const binding of bindings) {
        // Replace only an explicit bound reference, never a generic category noun.
        const expression = new RegExp(`(?:\\b(a|an)\\s+)?\\b${binding.concept}\\s+${binding.token}(?![a-z0-9_])`, 'g');
        rendered = rendered.replace(expression, (from, article) => article
          ? `${/^[aeiou]/.test(binding.surface) ? 'an' : 'a'} ${binding.surface}` : binding.surface);
      }
      if (/<\s*\/?\s*term_\d+\s*>/i.test(rendered)) {
        throw new Error('Output has an unknown reference or the wrong concept for its slot');
      }
      return rendered;
    },

    get bindings() { return bindings.map(copy); },
    reset() { bindings = []; },
  };
}
