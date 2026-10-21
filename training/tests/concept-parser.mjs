import assert from 'node:assert/strict';
import fs from 'node:fs';
import { createConceptParser } from '../conversations/concept-parser.mjs';
const tokenizer = JSON.parse(fs.readFileSync(new URL('../conversations/tokenizer.json', import.meta.url)));
const parser = createConceptParser(tokenizer);
const input = parser.parse('My cat is under the table. My dog is by the door.');
assert.equal(input.ready, true);
assert.equal(input.modelText, 'my animal <term_0> is under the table. my animal <term_1> is by the door.');
assert.equal(parser.render('the animal <term_0> is under the table.'), 'the cat is under the table.');
assert.equal(parser.render('the animal <term_1> is by the door.'), 'the dog is by the door.');
assert.equal(parser.render('an animal needs food.'), 'an animal needs food.');
assert.equal(parser.parse('where is my dog?').modelText, 'where is my animal <term_1>?');
assert.equal(parser.parse('my cat is not by the door.').modelText, 'my animal <term_0> is not by the door.');
assert.equal(parser.render('my animal <term_0> and your animal <term_0> are here.'), 'my cat and your cat are here.');
// Surface slots preserve words; they do not resolve two cats into one physical animal.
assert.equal(parser.bindings.length, 2);
const copy = parser.bindings;
copy[0].surface = 'wrong';
assert.equal(parser.render('animal <term_0>'), 'cat');
assert.throws(() => parser.render('tool <term_0>'), /wrong concept/);
assert.throws(() => parser.render('animal <term_7>'), /unknown reference/);
assert.throws(() => parser.render('<term_0>'), /wrong concept/);
assert.throws(() => parser.parse('animal <term_0>'), /reserved/);
assert.throws(() => parser.parse('animal <TERM_0>'), /reserved/);
const unsupported = parser.parse('my hamster is here.');
assert.equal(unsupported.ready, false);
assert.deepEqual(unsupported.unsupported, ['hamster']);
assert.equal(parser.parse('catalog').ready, false); // No substring replacement of cat.
assert.equal(parser.parse('<script>cat</script>').ready, false);
assert.equal(parser.parse('my cat is not hungry.').modelText, 'my animal <term_0> is not hungry.');
assert.equal(parser.parse('i dont like my cat.').modelText, 'i do not like my animal <term_0>.');
const plural = parser.parse('two cats are by the tree.');
assert.equal(plural.modelText, 'two animals <term_2> are by the tree.');
assert.equal(parser.render('animals <term_2> are by the tree.'), 'cats are by the tree.');
assert.throws(() => parser.render('animal <term_2> is here.'), /wrong concept/);
parser.reset();
assert.deepEqual(parser.bindings, []);
assert.throws(() => parser.render('animal <term_0>'), /unknown reference/);
const failed = parser.parse('a cat and a hamster.');
assert.equal(failed.ready, false);
assert.deepEqual(parser.bindings, []);
assert.equal(parser.parse('a dog.').modelText, 'an animal <term_0>.');
assert.equal(parser.render('an animal <term_0> is here.'), 'a dog is here.');
// Unknown specificity is not silently inferred from an animal label.
assert.equal(parser.render('i do not know what sound the animal <term_0> makes.'),
  'i do not know what sound the dog makes.');
const crowded = createConceptParser(tokenizer);
assert.equal(crowded.parse('cat cats dog dogs lion lions tiger tigers').ready, true);
const full = crowded.parse('a rabbit.');
assert.equal(full.ready, false);
assert.deepEqual(full.referenceCapacityExceeded, ['rabbit']);
assert.equal(crowded.bindings.length, 8);
assert.equal(crowded.parse('the cat.').modelText, 'the animal <term_0>.');
const other = createConceptParser(tokenizer);
assert.equal(other.parse('my hammer and my truck.').modelText, 'my tool <term_0> and my vehicle <term_1>.');
assert.equal(other.render('the tool <term_0> is in the vehicle <term_1>.'), 'the hammer is in the truck.');
assert.equal(other.parse('the rose and the oak.').modelText, 'the flower <term_2> and the tree <term_3>.');
assert.equal(other.parse('an oak.').modelText, 'a tree <term_3>.');
assert.equal(other.render('a tree <term_3> is here.'), 'an oak is here.');
assert.throws(() => other.render('tool <term_0>more'), /unknown reference/);
assert.equal(other.parse('my bird is here.').modelText, 'my animal <term_4> is here.');
const fruit = createConceptParser(tokenizer);
assert.equal(fruit.parse('an apple and a banana.').modelText, 'a fruit <term_0> and a fruit <term_1>.');
assert.equal(fruit.render('a fruit <term_0> is here.'), 'an apple is here.');
assert.equal(fruit.render('the fruit <term_1> is yellow.'), 'the banana is yellow.');
assert.equal(fruit.parse('two apples and a lemon.').modelText, 'two fruits <term_2> and a fruit <term_3>.');
assert.equal(fruit.render('the fruits <term_2> are here.'), 'the apples are here.');
assert.equal(fruit.render('fruit is food.'), 'fruit is food.');
assert.equal(fruit.parse('an orange shirt.').modelText, 'an orange garment <term_4>.');
assert.equal(fruit.render('an orange garment <term_4>.'), 'an orange shirt.');
assert.equal(fruit.parse('an orange.').modelText, 'an orange.');
assert.equal(fruit.bindings.length, 5);
assert.equal(fruit.parse('pineapple').ready, false);
const spelling = createConceptParser(tokenizer);
assert.equal(spelling.parse('teh aples are here.').modelText, 'the fruits <term_0> are here.');
assert.equal(spelling.parse('the banana is not here.').modelText, 'the fruit <term_1> is not here.');
assert.equal(spelling.render('the fruit <term_1> is not here.'), 'the banana is not here.');
assert.equal(new Set(tokenizer.tokens.filter(t => /^[a-z]+$/.test(t))).size, 968);
for (const word of ['apple', 'apples', 'banana', 'lemon']) assert.ok(!tokenizer.tokens.includes(word));
for (const [raw, model] of [
  ['a carrot and a potato.', 'a vegetable <term_0> and a vegetable <term_1>.'],
  ['two carrots and three potatoes.', 'two vegetables <term_0> and three vegetables <term_1>.'],
  ['my bird and my dog.', 'my animal <term_0> and my animal <term_1>.'],
  ['my car and my bike.', 'my vehicle <term_0> and my vehicle <term_1>.'],
  ['two boats.', 'two vehicles <term_0>.'],
  ['the bottle is in the bag.', 'the container <term_0> is in the container <term_1>.'],
  ['two boxes.', 'two containers <term_0>.'],
  ['a hat and two socks.', 'a garment <term_0> and two garments <term_1>.'],
  ['a knife and two pencils.', 'a tool <term_0> and two tools <term_1>.'],
]) {
  const category = createConceptParser(tokenizer);
  assert.equal(category.parse(raw).modelText, model, raw);
  assert.equal(category.render(model), raw, raw);
}
for (const raw of ['the train is here.', 'i train the model.', 'an orange.',
  'i dress for work.', 'i fish.', 'my mother and my sister.', 'one two three.', 'one cup of water.']) {
  assert.equal(createConceptParser(tokenizer).parse(raw).modelText, raw, raw);
}
const groups = JSON.parse(fs.readFileSync(new URL('../conversations/concept-groups.json', import.meta.url)));
let mappedForms = 0;
for (const [concept, forms] of Object.entries(groups)) {
  for (const surface of forms) {
    const category = createConceptParser(tokenizer);
    const raw = `the ${surface}.`;
    const model = `the ${concept} <term_0>.`;
    assert.ok(!tokenizer.tokens.includes(surface), surface);
    assert.equal(category.parse(raw).modelText, model, surface);
    assert.equal(category.render(model), raw, surface);
    mappedForms++;
  }
}
console.log(JSON.stringify({ status: 'passed', category_forms_round_tripped: mappedForms }));
console.log(JSON.stringify({ status: 'passed', vocabulary_words: 968, token_ids: tokenizer.tokens.length,
  checks: ['bindings across turns', 'plural forms', 'negation', 'contractions', 'generic noun preservation',
    'unknown words', 'slot exhaustion', 'transactional state', 'reference validation', 'reset', 'known words'] }));

const authored = fs.readFileSync(new URL('../conversations/authored/batch-000047.jsonl', import.meta.url), 'utf8')
  .trim().split('\n').map(line => JSON.parse(line));
let fixtures = 0;
for (const row of authored.filter(row => row.parser_example)) {
  const fixture = createConceptParser(tokenizer);
  const parsed = fixture.parse(row.parser_example.raw_first_user);
  assert.equal(parsed.ready, true, row.id);
  assert.equal(parsed.modelText, row.messages[0].content, row.id);
  assert.deepEqual(parsed.bindings.map(binding => binding.surface), row.parser_example.surfaces, row.id);
  for (const message of row.messages.filter(message => message.role === 'assistant')) {
    assert.doesNotThrow(() => fixture.render(message.content), row.id);
  }
  fixtures++;
}
assert.equal(fixtures, 4);
console.log(JSON.stringify({ status: 'passed', authored_reference_conversations: fixtures }));
