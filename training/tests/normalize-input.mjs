import assert from 'node:assert/strict';
import fs from 'node:fs';
import { createNormalizer } from '../../web/dist/normalize.js';
const tokenizer=JSON.parse(fs.readFileSync(new URL('../../web/dist/models/picoboi-100k-v1/tokenizer.json',import.meta.url)));
const normalize=createNormalizer(tokenizer);
const cases=new Map([
  ['yo','hello.'],['hi','hello.'],['hello','hello.'],['hey!','hello.'],
  ['whats up','hello.'],["what’s up?",'hello.'],['what is up?','hello.'],
  ['whats the color?','what is the color?'],["I'm happy.",'i am happy.'],
  ['i dont want it.','i do not want it.'],["isn’t it red?",'is it not red?'],
  ['teh red ball is on the table.','the red ball is on the table.'],
  ['i have two aples.','i have two apples.'],['pls help me','please help me'],
  ['u wanna read?','you want to read?'],['i know its color.','i know its color.'],
  ['we were well.','we were well.'],['read the red book.','read the red book.'],
  ['the ball is not red.','the ball is not red.'],['their book is there.','their book is there.'],
]);
for(const [input,expected] of cases) {
  const actual=normalize(input);
  assert.equal(actual.text,expected,input);
  assert.equal(actual.unsupported.length,0,input);
  assert.equal(normalize(actual.text).text,actual.text,'Normalization must be idempotent');
}
const sameMeaning=normalize('i dont want it.');
assert.match(sameMeaning.text,/\bnot\b/);
for(const word of tokenizer.tokens.filter(t=>/^[a-z]+$/.test(t))) {
  if(['hi','hey','hello'].includes(word)) continue;
  if(word in {whats:1,wheres:1,whos:1,hows:1,thats:1,theres:1,heres:1,im:1,youre:1,theyre:1,isnt:1,arent:1,wasnt:1,werent:1,dont:1,doesnt:1,didnt:1,cant:1,couldnt:1,wont:1,wouldnt:1,shouldnt:1,havent:1,hasnt:1,ive:1,youve:1,lets:1}) continue;
  assert.equal(normalize(word).text,word,`Preserve known vocabulary: ${word}`);
}
assert.equal(normalize('planet').text,'planet');
assert.deepEqual(normalize('planet').unsupported,['planet']);
assert.ok(normalize('planet').suggestions[0].candidates.includes('plane'));
assert.equal(normalize('ur happy').text,'ur happy');
assert.ok(normalize('hellp').suggestions[0].candidates.length>1);
assert.equal(normalize('<script>alert(1)</script>').text,'<script>alert(1)</script>');
console.log(JSON.stringify({status:'passed',normalization_cases:cases.size,
  checks:['contractions','greeting aliases','known spelling fixes','negation','known-word preservation','ambiguity suggestions','idempotence']}));
