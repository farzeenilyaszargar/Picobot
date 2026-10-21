// A small, inspectable input adapter, not another language model.
// Keep the trained vocabulary/token IDs unchanged; never normalize model output.
const contractions = {
  "what's": 'what is', whats: 'what is', "where's": 'where is', wheres: 'where is',
  "who's": 'who is', whos: 'who is', "how's": 'how is', hows: 'how is',
  "that's": 'that is', thats: 'that is', "there's": 'there is', theres: 'there is',
  "here's": 'here is', heres: 'here is', "it's": 'it is',
  "i'm": 'i am', im: 'i am', "you're": 'you are', youre: 'you are',
  "we're": 'we are', "they're": 'they are', theyre: 'they are',
  "isn't": 'is not', isnt: 'is not', "aren't": 'are not', arent: 'are not',
  "wasn't": 'was not', wasnt: 'was not', "weren't": 'were not', werent: 'were not',
  "don't": 'do not', dont: 'do not', "doesn't": 'does not', doesnt: 'does not',
  "didn't": 'did not', didnt: 'did not', "can't": 'can not', cant: 'can not',
  "couldn't": 'could not', couldnt: 'could not', "won't": 'will not', wont: 'will not',
  "wouldn't": 'would not', wouldnt: 'would not', "shouldn't": 'should not', shouldnt: 'should not',
  "haven't": 'have not', havent: 'have not', "hasn't": 'has not', hasnt: 'has not',
  "i've": 'i have', ive: 'i have', "you've": 'you have', youve: 'you have',
  "we've": 'we have', "they've": 'they have',
  "i'll": 'i will', "you'll": 'you will', "we'll": 'we will', "they'll": 'they will',
  "let's": 'let us', lets: 'let us',
};
const aliases = {
  yo: 'hello', hiya: 'hello', howdy: 'hello', sup: 'hello',
  yep: 'yes', yeah: 'yes', yup: 'yes', nope: 'no', nah: 'no',
  pls: 'please', plz: 'please', thx: 'thank you', ty: 'thank you',
  u: 'you', idk: 'i do not know', wanna: 'want to', gonna: 'going to',
};
const typos = {
  helo: 'hello', helllo: 'hello', hlelo: 'hello',
  teh: 'the', thsi: 'this', taht: 'that', whta: 'what', waht: 'what',
  becuase: 'because', beacuse: 'because', pleae: 'please', plese: 'please',
  thnaks: 'thanks', tihs: 'this', sentense: 'sentence', freind: 'friend',
  aples: 'apples', aple: 'apple',
};

function oneEdit(a, b) {
  if (Math.abs(a.length - b.length) > 1 || a === b) return false;
  let i = 0;
  while (i < Math.min(a.length, b.length) && a[i] === b[i]) i++;
  if (a.length === b.length) {
    return a.slice(i + 1) === b.slice(i + 1) ||
      (a[i] === b[i + 1] && a[i + 1] === b[i] && a.slice(i + 2) === b.slice(i + 2));
  }
  return a.length < b.length ? a.slice(i) === b.slice(i + 1) : a.slice(i + 1) === b.slice(i);
}

export function createNormalizer(tokenizer) {
  const vocabulary = new Set(tokenizer.tokens.filter(word => /^[a-z]+$/.test(word)));
  const words = [...vocabulary];
  const allowed = text => text.split(' ').every(word => vocabulary.has(word));
  for (const replacement of [...Object.values(contractions), ...Object.values(aliases), ...Object.values(typos)]) {
    if (!allowed(replacement)) throw new Error(`Normalizer replacement is outside vocabulary: ${replacement}`);
  }
  return function normalizeInput(original) {
    const changes = [];
    const replace = (from, to, reason) => { if (from !== to) changes.push({ from, to, reason }); return to; };
    let text = original.toLowerCase().trim().replace(/[’‘]/g, "'");
    // Only a complete greeting is canonicalized, not words inside another sentence.
    if (/^(?:hi|hello|hey|yo|hiya|howdy|sup|what'?s\s+up|what\s+is\s+up)[.!?]*$/.test(text)) {
      text = replace(text, 'hello.', 'greeting');
    } else {
      text = text.replace(/[a-z]+(?:'[a-z]+)?/g, word => {
        if (Object.hasOwn(contractions, word)) return replace(word, contractions[word], 'contraction');
        // Preserve every recognized word, including its inflection and tense.
        if (vocabulary.has(word)) return word;
        if (Object.hasOwn(aliases, word)) return replace(word, aliases[word], 'alias');
        if (Object.hasOwn(typos, word)) return replace(word, typos[word], 'spelling');
        return word;
      });
    }
    text = text.replace(/\s+/g, ' ').replace(/\s+([.,?!:])/g, '$1');
    text = text.replace(/^(is|are|was|were|do|does|did|can|could|will|would|should|have|has) not (i|you|he|she|it|we|they)\b(.*\?)$/,
      (from, auxiliary, subject, rest) => replace(from, `${auxiliary} ${subject} not${rest}`, 'question word order'));
    const unsupported = [...new Set((text.match(/[a-z]+(?:'[a-z]+)?/g) || []).filter(word => !vocabulary.has(word)))];
    const suggestions = unsupported.map(word => ({
      word,
      candidates: word.length >= 4 && word.length <= 24
        ? words.filter(candidate => oneEdit(word, candidate)).slice(0, 5) : [],
    }));
    // Unknown real words and names can resemble vocabulary words: suggest rather than silently guess.
    return { text, changed: changes.length > 0, changes, unsupported, suggestions };
  };
}
