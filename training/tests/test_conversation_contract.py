"""Exercise vocabulary, reference, and length failure paths without altering the corpus."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
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
        for name in ['contract.json', 'tokenizer.json', 'vocabulary.txt', 'archive/tokenizer-818.json']:
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
        self.assertEqual(report['tokenizer_size'], 1020)

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

    def test_previous_token_ids_cannot_drift(self):
        path = self.data/'tokenizer.json'
        tokenizer = json.loads(path.read_text())
        tokenizer['tokens'][12], tokenizer['tokens'][13] = tokenizer['tokens'][13], tokenizer['tokens'][12]
        path.write_text(json.dumps(tokenizer))
        with self.assertRaisesRegex(ValueError, '818 token IDs'):
            self.run_audit()

    def test_vocabulary_cannot_silently_grow(self):
        with (self.data/'vocabulary.txt').open('a') as stream:
            stream.write('cat\n')
        with self.assertRaisesRegex(ValueError, 'exactly 1000'):
            self.run_audit()


if __name__ == '__main__':
    unittest.main()
