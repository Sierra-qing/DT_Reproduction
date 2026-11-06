'''新实验配置/数据转换的关键约束；不训练、不调用GPU、不联网。'''
import importlib.util,json,tempfile,unittest
from pathlib import Path
import numpy as np
from unittest.mock import patch
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]

def module(name,path):
 spec=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
catalog=module('catalog_builder','experiments/build_catalog.py')
manage=module('experiment_manager','experiments/manage.py')
prepare=module('prepare_data','experiments/prepare_data.py')
atari=module('atari_adapter','experiments/runners/atari.py')
naming=module('swanlab_naming','gym/swanlab_observer.py')

class Suite(unittest.TestCase):
 def test_short_names_preserve_task_identity(self):
  args=SimpleNamespace(env='hopper',dataset='medium-replay',model_type='dt',pct_traj=1.,mode='normal',seed=0)
  self.assertEqual(naming.display_name(args,'/long/local/path'),'Hopper-DT-Replay-s0')
  args.dataset='medium';args.model_type='bc';args.pct_traj=.1
  self.assertEqual(naming.display_name(args,'/long/local/path'),'Hopper-BC10%-Medium-s0')
  args.model_type='dt';args.mode='delayed'
  self.assertEqual(naming.display_name(args,'/long/local/path'),'Hopper-DT-Medium-Delayed-s0')
 def test_unique_and_dependencies(self):
  jobs=manage.catalog();raw=json.loads((ROOT/'experiments/configs/catalog.json').read_text())['jobs']
  self.assertEqual(len(raw),len(jobs))
  for j in raw:
   if j['backend']=='sweep':self.assertIn(j['dependency'],jobs)
 def test_paper_budgets(self):
  for job in manage.catalog().values():
   if job['backend']!='gym':continue
   p=job['params'];self.assertEqual(p['warmup_steps'],100000)
   self.assertEqual(p['max_iters']*p['num_steps_per_iter'],100000 if p['model_type']=='dt' else 25000)
   self.assertEqual(p['embed_dim'],128 if p['model_type']=='dt' else 256)
   self.assertEqual(p['eval_targets'][0],catalog.TARGETS[p['env']][0])
 def test_existing_completed_not_retrained(self):
  for seed in [0,1,2]:self.assertIsNotNone(manage.completed_run(manage.catalog()[f'dt-hopper-medium-s{seed}']))
 def test_terminal_timeout_and_partial_tail(self):
  data={k:np.arange(7).reshape(7,1) for k in ['observations','actions','next_observations']}
  data.update(rewards=np.ones(7),terminals=np.array([0,1,0,0,0,0,0]),timeouts=np.array([0,0,0,0,1,0,0]))
  paths=prepare.convert(data)
  self.assertEqual([len(p['rewards']) for p in paths],[2,3])
  self.assertEqual(paths[1]['observations'].reshape(-1).tolist(),[2,3,4])
 def test_percentile_keeps_whole_best_trajectory(self):
  data=(list(range(6)),np.arange(6),np.array([2,10,4,0]),np.array([2,4,6]),np.array([2,1,10,5,4,2]),np.array([0,1,0,1,0,1,0]))
  self.assertIs(atari.select_percentile(data,100),data)
  result=atari.select_percentile(data,25)
  self.assertEqual(result[0],[2,3]);self.assertEqual(result[3].tolist(),[2])
 def test_atari_preserves_paper_scope(self):
  jobs=manage.catalog()
  self.assertEqual(jobs['atari-pong-dt-s123']['params']['context_length'],50)
  self.assertEqual(jobs['atari-pong-dt-s123']['params']['batch_size'],512)
 def test_ready_plan_no_completed_dt(self):
  plan=json.loads((ROOT/'experiments/configs/hopper-ready.json').read_text())['job_ids']
  self.assertEqual(len(plan),15)
  self.assertFalse(any(x.startswith('dt-') for x in plan))
 def test_parallel_plans_are_disjoint(self):
  first=set(json.loads((ROOT/'experiments/configs/hopper-ready.json').read_text())['job_ids'])
  second=set(json.loads((ROOT/'experiments/configs/hopper-remainder.json').read_text())['job_ids'])
  self.assertFalse(first.intersection(second));self.assertEqual(len(first|second),42)
 def test_scheduler_rejects_overlapping_live_plan(self):
  import os
  with tempfile.TemporaryDirectory() as folder:
   root=Path(folder);(root/'state').mkdir()
   (root/'state/lane1.json').write_text(json.dumps({'driver_pid':os.getpid(),'job_ids':['first']}))
   with patch.object(manage,'HERE',root):
    with self.assertRaises(SystemExit):manage.register_lane(2,[{'id':'first'}],root/'batch')
    manage.register_lane(2,[{'id':'second'}],root/'batch')
   self.assertTrue((root/'state/lane2.json').exists())

if __name__=='__main__':unittest.main()
