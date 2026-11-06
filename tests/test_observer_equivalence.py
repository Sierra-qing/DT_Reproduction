import argparse
import ast
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'gym'))
import experiment as original
import experiment_observed as observed

class NullWriter:
    def __init__(self,**kwargs): pass
    def __getattr__(self,name): return lambda *a,**k: None

class FakeEnv:
    observation_space=SimpleNamespace(shape=(11,))
    action_space=SimpleNamespace(shape=(3,),seed=lambda seed: None)
    def reset(self): return np.zeros(11)
    def step(self,action): return np.zeros(11), 0., True, {}
    def seed(self,seed): pass
    def close(self): pass


def fake_episode(*args,**kwargs):
    return float(np.random.uniform(100,200)),random.randint(10,30)

class CaptureTrainer(original.SequenceTrainer):
    latest=None
    def __init__(self,*a,**k):
        super().__init__(*a,**k)
        CaptureTrainer.latest=self


def seed(value):
    random.seed(value);np.random.seed(value);torch.manual_seed(value);torch.cuda.manual_seed_all(value)


if __name__=='__main__':
    tree=ast.parse((ROOT/'gym/experiment.py').read_text())
    main=next(n for n in tree.body if isinstance(n,ast.If))
    statements=main.body[:next(i for i,n in enumerate(main.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='args' for t in n.targets))]
    namespace={'argparse':argparse}
    exec(compile(ast.Module(body=statements,type_ignores=[]),'<official parser>','exec'),namespace)
    defaults=vars(namespace['parser'].parse_args([]))
    for key, value in defaults.items():
        if key == 'swanlab_mode':
            assert observed.build_parser().parse_args([]).swanlab_mode == value
            continue
        assert observed.CORE_DEFAULTS[key]==(100000 if key=='warmup_steps' else value),key
    print('PASS: official default values match, except explicitly documented paper warmup=100000',flush=True)
    original_evaluation=original.evaluate_episode_rtg
    original_make=original.gym.make
    original_trainer=original.SequenceTrainer
    with tempfile.TemporaryDirectory(prefix='dt-observer-test-') as temporary:
        args=observed.build_parser().parse_args(['--device','cuda','--seed','73','--embed_dim','16','--n_layer','1',
            '--batch_size','4','--max_iters','2','--num_steps_per_iter','3','--num_eval_episodes','2','--log_interval','2',
            '--eval_log_interval','1','--batch_snapshot_interval','2','--output_dir',temporary])
        config={key:getattr(args,key) for key in observed.CORE_DEFAULTS}
        if '--with-swanlab' in sys.argv:
            args.log_to_swanlab = True
            args.swanlab_mode = 'offline'
        with patch.object(original.gym,'make',lambda *a,**k: FakeEnv()), \
             patch.object(original,'evaluate_episode_rtg',fake_episode), \
             patch.object(original,'SequenceTrainer',CaptureTrainer):
            seed(73)
            with contextlib.redirect_stdout(io.StringIO()):
                original.experiment('test',dict(config))
            baseline={key:value.detach().cpu().clone() for key,value in CaptureTrainer.latest.model.state_dict().items()}
            optimizer=CaptureTrainer.latest.optimizer.state_dict()
            seed(73)
            with contextlib.redirect_stdout(io.StringIO()):
                run=observed.run_observed(args)
            result=torch.load(run/'checkpoint_latest.pt',map_location='cpu')
            for key,value in baseline.items():
                assert torch.equal(value,result['model_state_dict'][key].cpu()),key
            for parameter_id,state in optimizer['state'].items():
                for key,value in state.items():
                    candidate=result['optimizer_state_dict']['state'][parameter_id][key]
                    assert torch.equal(value.cpu(),candidate.cpu()) if isinstance(value,torch.Tensor) else value==candidate
            assert result['global_step']==6
            rows=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
            assert len(rows)==2
            import csv
            assert len(list(csv.DictReader((run/'train.csv').open())))==6
            assert len(list(csv.DictReader((run/'eval_episodes.csv').open())))==8
            assert len(list(csv.DictReader((run/'train_steps.csv').open())))==6
            provenance=[json.loads(line) for line in (run/'training_provenance.jsonl').read_text().splitlines()]
            assert [entry['global_step'] for entry in provenance]==list(range(1,7))
            from review_experiment import reconstruct_batch
            reconstructed=reconstruct_batch(run,2)
            with np.load(run/'training_batches/update_0000002.npz',allow_pickle=False) as snapshot:
                for name,array in reconstructed.items():
                    assert np.array_equal(array,snapshot[name]),name
                valid=snapshot['attention_mask']>0
                mse=np.mean((snapshot['action_predictions'][valid]-snapshot['actions_data'][valid])**2)
            step_rows=list(csv.DictReader((run/'train_steps.csv').open()))
            assert np.isclose(mse,float(step_rows[1]['action_mse']),rtol=1e-6)
            assert len(list((run/'checkpoints').glob('iteration_*.pt')))==2

            for name in ['dataset.json','metadata.json','config.json','console.log','events.jsonl']:
                assert (run/name).is_file(),name
            console=(run/'console.log').read_text()
            for title in ['实验与观测面板','训练进度','评估进度','本轮汇总']:
                assert title in console,title
            width=None
            for line in console.splitlines():
                if line.startswith('+'): width=observed.Tables.width(line)
                elif line.startswith('|'): assert observed.Tables.width(line)==width
        assert original.evaluate_episode_rtg is original_evaluation
        assert original.gym.make is original_make
        assert original.SequenceTrainer is original_trainer
    print('PASS: CUDA parameter updates and optimizer state exactly match unwrapped official training',flush=True)
    print('PASS: dense per-update metrics, all sampling provenance, exact batch reconstruction and prediction MSE',flush=True)
    print('PASS: progress, per-episode CSV, JSONL, per-iteration checkpoints, table alignment and patch restoration',flush=True)
