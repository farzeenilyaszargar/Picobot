"""Correctness checks for attention, target masks, and checkpoint continuation."""
import contextlib
import io
import json
from pathlib import Path
import signal
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from model import Picoboi, masked_loss
import train


ROOT = Path(__file__).resolve().parent


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.config = json.loads((ROOT/'configs/picoboi-100k-looped.json').read_text())

    def test_parameter_count_and_causality(self):
        torch.manual_seed(9)
        model = Picoboi(self.config).eval()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 99120)
        x = torch.randint(12,812,(2,20)); changed = x.clone()
        changed[:,10:] = torch.randint(12,812,(2,10))
        with torch.no_grad():
            original=model(x)
            torch.testing.assert_close(original[:,:10],model(changed)[:,:10],rtol=0,atol=0)
            torch.testing.assert_close(original[:,:10],model(x[:,:10]),rtol=1e-4,atol=1e-5)

    def test_mask_excludes_user_and_padding_targets(self):
        logits=torch.randn(2,10,812,requires_grad=True)
        labels=torch.randint(0,812,(2,10)); mask=torch.zeros(2,10);mask[:,3:6]=1
        loss,_,_=masked_loss(logits,labels,mask);loss.backward()
        self.assertEqual(logits.grad[mask==0].abs().max().item(),0)
        self.assertGreater(logits.grad[mask==1].abs().sum().item(),0)
        data=train.Dialogues(ROOT/'data/picoboi-800-v1','train',128)
        indices=np.array([0,799,800,1200]);x,y,w,_=data.batch(indices,'cpu')
        for row,index in enumerate(indices):
            start,length=int(data.offsets[index]),int(data.lengths[index])
            np.testing.assert_array_equal(y[row,:length-1].numpy(),data.tokens[start+1:start+length])
            np.testing.assert_array_equal(w[row,:length-1].numpy(),data.masks[start+1:start+length])
            self.assertTrue(torch.all(w[row,length-1:]==0))

    def test_resume_matches_uninterrupted_training(self):
        source=train.Dialogues(ROOT/'data/picoboi-800-v1','train',128)
        class Tiny:
            def __len__(self): return 6
            def batch(self,indices,device): return source.batch(indices,device)
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            base=Path(temporary)
            def arguments(name,resume=False):
                run=base/name;run.mkdir(exist_ok=True)
                return SimpleNamespace(auto=False,device='cpu',threads=1,seed=99,lr=.003,
                                       batch_size=2,epochs=1,data=ROOT/'data/picoboi-800-v1',
                                       resume=resume,run_dir=run,eval_every=1,patience=10)
            old_handlers={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT)}
            try:
                train.train_model(self.config,Tiny(),Tiny(),arguments('full'))
                original_save=train.save
                def interrupt_after_save(path,state):
                    original_save(path,state)
                    if path.name=='last.pt' and state['step']==1:
                        signal.raise_signal(signal.SIGTERM)
                with patch.object(train,'save',interrupt_after_save):
                    train.train_model(self.config,Tiny(),Tiny(),arguments('resumed'))
                partial=torch.load(base/'resumed/last.pt',weights_only=False)
                self.assertEqual(partial['step'],1)
                train.train_model(self.config,Tiny(),Tiny(),arguments('resumed',True))
                full=torch.load(base/'full/last.pt',weights_only=False)
                resumed=torch.load(base/'resumed/last.pt',weights_only=False)
                for key in full['model']:
                    torch.testing.assert_close(full['model'][key],resumed['model'][key],rtol=0,atol=0)
                self.assertEqual(full['tokens_seen'],resumed['tokens_seen'])
                # Resuming an already finished checkpoint must also be a safe no-op.
                train.train_model(self.config,Tiny(),Tiny(),arguments('resumed',True))
            finally:
                for sig,handler in old_handlers.items(): signal.signal(sig,handler)


if __name__=='__main__':
    unittest.main()
