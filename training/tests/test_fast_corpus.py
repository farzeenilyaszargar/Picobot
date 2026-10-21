import argparse
import contextlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import re
import sys
import tempfile
import threading
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fast_corpus import generate as gen

TEXT = ('user: i have a red book. the book is by the table. where is my book?\n'
        'assistant: your red book is by the table. you put it there.')


class FastCorpus(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.codecs = {name:gen.load_variant(name) for name in ['word-level','bpe-512']}

    def test_strict_word_boundary_and_message_checks(self):
        words = self.codecs['word-level'].spec['word_vocabulary']
        pattern = re.compile(gen.grammar(words))
        self.assertIsNotNone(pattern.fullmatch(TEXT))
        messages,count,tokens = gen.validate(TEXT,self.codecs)
        self.assertEqual(count,26)
        self.assertEqual(len(messages),2)
        self.assertLess(tokens['bpe-512'],1024)
        for bad in [TEXT.replace('book','bookshelf'), TEXT.replace('red','Red'),
                    TEXT+'\nuser: hello.', TEXT.replace('book','<term_0>'),
                    TEXT.replace('red','réd')]:
            self.assertIsNone(pattern.fullmatch(bad))
            with self.assertRaises(ValueError):gen.validate(bad,self.codecs)
        with self.assertRaisesRegex(ValueError,'context'):
            gen.validate(TEXT,self.codecs,max_context=5)

    def test_exact_and_near_duplicate_screen(self):
        dedupe=gen.Dedupe()
        words=' '.join(self.codecs['word-level'].spec['word_vocabulary'][:100])
        row=[dict(role='user',content=words),dict(role='assistant',content='yes.')]
        self.assertIsNone(dedupe.add(row))
        self.assertEqual(dedupe.add(row),'exact_duplicate')
        similar=[dict(role='user',content=words+' tomorrow'),dict(role='assistant',content='yes.')]
        self.assertEqual(dedupe.add(similar),'near_duplicate')

    def test_plan_split_does_not_change_with_length(self):
        groups={}
        for i in range(2000):
            plan=gen.profile(i,42)
            if plan['group'] in groups:self.assertEqual(groups[plan['group']],plan['split'])
            groups[plan['group']]=plan['split']
        self.assertIn('validation',groups.values())
        self.assertIn('test',groups.values())

    def test_http_generation_resume_and_counts(self):
        captured=[]
        class Handler(BaseHTTPRequestHandler):
            fail=False
            def do_POST(self):
                captured.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
                if self.fail:
                    self.send_response(400)
                    self.end_headers()
                    return
                payload=json.dumps(dict(choices=[dict(finish_reason='stop',message=dict(content=TEXT))],
                                        usage=dict(completion_tokens=50))).encode()
                self.send_response(200)
                self.send_header('Content-Type','application/json')
                self.end_headers()
                self.wfile.write(payload)
            def log_message(self,*args):pass
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        with tempfile.TemporaryDirectory() as directory:
            args=argparse.Namespace(output=directory, model='offline-test-teacher',
                base_url=f'http://127.0.0.1:{server.server_port}/v1', seed=42,
                no_constraints=False,max_output_tokens=1400,preview=False,workers=2,
                target_words=100,max_jobs=2)
            with contextlib.redirect_stdout(io.StringIO()):gen.run(args)
            report=json.loads((Path(directory)/'report.json').read_text())
            self.assertEqual(report['english_words'],26)
            self.assertEqual(report['accepted_total'],1)
            self.assertEqual(report['rejected_this_run'],{'exact_duplicate':1})
            self.assertEqual(report['status'],'partial_pending_more_generation_and_quality_review')
            self.assertIn('structured_outputs',captured[0])
            args.max_jobs=1
            with contextlib.redirect_stdout(io.StringIO()):gen.run(args)
            rows=[json.loads(line) for line in (Path(directory)/'attempts.jsonl').read_text().splitlines()]
            self.assertEqual(sorted(row['job'] for row in rows),[0,1,2])
            report=json.loads((Path(directory)/'report.json').read_text())
            self.assertEqual(report['english_words'],26)
            args.model='different'
            with self.assertRaisesRegex(ValueError,'Configuration changed'):gen.run(args)
        Handler.fail=True
        with tempfile.TemporaryDirectory() as directory:
            args.output=directory
            args.max_jobs=1000
            before=len(captured)
            with contextlib.redirect_stdout(io.StringIO()):gen.run(args)
            report=json.loads((Path(directory)/'report.json').read_text())
            self.assertTrue(report['halted_for_endpoint_error'])
            self.assertLessEqual(len(captured)-before,args.workers)


if __name__=='__main__':unittest.main()
