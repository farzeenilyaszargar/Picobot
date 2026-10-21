"""Exercise vocabulary, reference, and length failure paths without altering the corpus."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location('conversation_auditor', ROOT/'audit_conversations.py')
AUDITOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDITOR)


class ConversationContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.data = Path(self.temp.name)
        (self.data/'authored').mkdir()
        (self.data/'evaluation').mkdir()
        (self.data/'archive').mkdir()
        for name in ['contract.json', 'tokenizer.json', 'vocabulary.txt', 'archive/tokenizer-1020.json']:
            shutil.copyfile(ROOT/'conversations'/name, self.data/name)
        (self.data/'excluded-conversations.json').write_text('{}')
        self.row = dict(id='test', scenario='test', provenance='assistant_direct_authorship', topics=['test'],
                        messages=[dict(role='user', content='my animal <term_0> is here.'),
                                  dict(role='assistant', content='animal <term_0>.')])
        previous = AUDITOR.DATA
        AUDITOR.DATA = self.data
        self.addCleanup(setattr, AUDITOR, 'DATA', previous)

    def run_audit(self):
        (self.data/'authored/test.jsonl').write_text(json.dumps(self.row)+'\n')
        with contextlib.redirect_stdout(io.StringIO()):
            AUDITOR.audit()
        return json.loads((self.data/'audit.json').read_text())

    def test_reference_tokens_do_not_count_as_english_words(self):
        report = self.run_audit()
        self.assertEqual(report['english_words'], 5)
        self.assertEqual(report['model_tokens'], 15)
        self.assertEqual(report['tokenizer_size'], 1017)

    def test_unbound_output_reference_is_rejected(self):
        self.row['messages'][0]['content'] = 'hello.'
        with self.assertRaisesRegex(ValueError, 'unbound'):
            self.run_audit()

    def test_wrong_output_concept_is_rejected(self):
        self.row['messages'][1]['content'] = 'tool <term_0>.'
        with self.assertRaisesRegex(ValueError, 'mismatched reference'):
            self.run_audit()

    def test_reassigning_a_reference_concept_is_rejected(self):
        self.row['messages'] += [dict(role='user', content='tool <term_0>.'),
                                 dict(role='assistant', content='tool <term_0>.')]
        with self.assertRaisesRegex(ValueError, 'reference concept changed'):
            self.run_audit()

    def test_longer_context_is_accepted_but_not_unbounded(self):
        self.row['messages'] = [dict(role='user', content='hello '*600),
                                dict(role='assistant', content='hello.')]
        report = self.run_audit()
        self.assertEqual(report['conversations_over_512_tokens'], 1)
        self.assertEqual(report['conversations_over_1024_tokens'], 0)
        self.row['messages'][0]['content'] = 'hello '*1024
        with self.assertRaisesRegex(ValueError, 'exceeds 1024'):
            self.run_audit()

    def test_raw_unknown_words_need_the_parser(self):
        self.row['messages'][0]['content'] = 'my cat is here.'
        with self.assertRaisesRegex(ValueError, 'unknown words'):
            self.run_audit()

    def test_structural_token_ids_cannot_drift(self):
        path = self.data/'tokenizer.json'
        tokenizer = json.loads(path.read_text())
        tokenizer['tokens'][0], tokenizer['tokens'][1] = tokenizer['tokens'][1], tokenizer['tokens'][0]
        path.write_text(json.dumps(tokenizer))
        with self.assertRaisesRegex(ValueError, '12 structural token IDs'):
            self.run_audit()

    def test_vocabulary_cannot_silently_grow(self):
        with (self.data/'vocabulary.txt').open('a') as stream:
            stream.write('cat\n')
        with self.assertRaisesRegex(ValueError, 'exactly 997'):
            self.run_audit()

    def test_fruit_source_is_preserved_and_counted_once(self):
        from conversations.canonical import canonicalize
        self.row['messages'] = [dict(role='user', content='i have an apple.'),
                                dict(role='assistant', content='the apple is here.')]
        tokenizer = json.loads((self.data/'tokenizer.json').read_text())
        adapted = canonicalize(self.row, tokenizer)
        self.assertEqual(adapted['messages'][0]['content'], 'i have a fruit <term_0>.')
        self.assertEqual(adapted['messages'][1]['content'], 'the fruit <term_0> is here.')
        self.assertEqual(self.row['messages'][0]['content'], 'i have an apple.')
        self.assertEqual(self.run_audit()['english_words'], 8)

    def test_new_story_fruit_stays_generic_without_an_input_binding(self):
        from conversations.canonical import canonicalize
        self.row['messages'] = [dict(role='user', content='tell a story.'),
                                dict(role='assistant', content='she found an apple.')]
        tokenizer = json.loads((self.data/'tokenizer.json').read_text())
        adapted = canonicalize(self.row, tokenizer)
        self.assertEqual(adapted['messages'][1]['content'], 'she found a fruit.')
        self.run_audit()


if __name__ == '__main__':
    unittest.main()
