import array
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tokenization import load_variant


class TokenizerVariants(unittest.TestCase):
    def test_round_trips_and_browser_parity(self):
        fixtures = []
        for name, size in [('word-level', 988), ('bpe-512', 512)]:
            codec = load_variant(name)
            self.assertEqual(len(codec.tokens), size)
            texts = codec.spec['word_vocabulary'] + [
                'i know the facts.', 'the animal <term_0> is not hungry.',
                'what is this? yes, i know: it is a fruit <term_7>!']
            if name == 'bpe-512':
                texts.append('hamster')
            for text in texts:
                ids = codec.encode(text)
                self.assertEqual(codec.decode(ids), text)
                fixtures.append(dict(name=name, text=text, ids=ids))
            for text in ['Hello', '<user>', '<term_8>', 'café', 'two + two']:
                with self.assertRaises(ValueError):
                    codec.encode(text)
            ids, mask = codec.encode_conversation([
                dict(role='user', content='hi.'), dict(role='assistant', content='hello.')])
            turn = ids.index(codec.ids['<assistant>'])
            self.assertEqual(mask[:turn + 1], [0] * (turn + 1))
            self.assertEqual(mask[turn + 1:], [1] * (len(ids) - turn - 1))
        script = '''
import fs from 'node:fs';
import assert from 'node:assert/strict';
import {createTokenizer} from './training/conversations/tokenization.mjs';
import {createConceptParser} from './training/conversations/concept-parser.mjs';
const specs = Object.fromEntries(['word-level','bpe-512'].map(name => [name,
  JSON.parse(fs.readFileSync(`training/conversations/tokenizers/${name}/tokenizer.json`))]));
const codecs = Object.fromEntries(Object.entries(specs).map(([name,spec]) => [name,createTokenizer(spec)]));
for (const row of JSON.parse(fs.readFileSync(0,'utf8'))) {
  assert.deepEqual(codecs[row.name].encode(row.text),row.ids);
  assert.equal(codecs[row.name].decode(row.ids),row.text);
}
for (const [name,spec] of Object.entries(specs)) {
  const parser = createConceptParser(spec);
  const parsed = parser.parse('my dog is hungry.');
  assert.equal(parsed.ready,true);
  assert.equal(parser.render(codecs[name].decode(codecs[name].encode(parsed.modelText))),'my dog is hungry.');
  for (const text of ['Hello','<user>','<term_8>','café','two + two']) {
    assert.throws(() => codecs[name].encode(text));
  }
}
'''
        subprocess.run(['node', '--input-type=module', '-e', script],
                       input=json.dumps(fixtures), text=True, check=True, cwd=ROOT.parent)

    def test_packed_integrity(self):
        row_ids = []
        for name in ['word-level', 'bpe-512']:
            codec = load_variant(name)
            directory = ROOT/'conversations/tokenizers'/name
            manifest = json.loads((directory/'manifest.json').read_text())
            packed = ROOT.parent/manifest['local_data']
            tokens = (packed/'train.bin').read_bytes()
            mask = (packed/'train.assistant_mask.bin').read_bytes()
            self.assertEqual(hashlib.sha256(tokens).hexdigest(),manifest['tokens_sha256'])
            self.assertEqual(hashlib.sha256(mask).hexdigest(),manifest['assistant_mask_sha256'])
            self.assertEqual(len(tokens),2 * len(mask))
            self.assertEqual(sum(mask),manifest['assistant_targets'])
            sequence = array.array('H')
            sequence.frombytes(tokens)
            if sys.byteorder != 'little': sequence.byteswap()
            self.assertLess(max(sequence),len(codec.tokens))
            offsets = [json.loads(line) for line in (packed/'train.offsets.jsonl').read_text().splitlines()]
            self.assertEqual(len(offsets),1290)
            self.assertEqual(offsets[0]['start'],0)
            self.assertEqual(offsets[-1]['end'],len(mask))
            for i,row in enumerate(offsets):
                if i: self.assertEqual(row['start'],offsets[i-1]['end'])
                self.assertLessEqual(row['end']-row['start'],1024)
                self.assertEqual(sequence[row['start']],codec.ids['<bos>'])
                self.assertEqual(sequence[row['end']-1],codec.ids['<eos>'])
            row_ids.append([row['id'] for row in offsets])
        self.assertEqual(*row_ids)


if __name__ == '__main__': unittest.main()
