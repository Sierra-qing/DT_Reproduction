'''只测图环境、随机数据与先验数学，不训练策略。'''
import importlib.util,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('graph_task',ROOT/'experiments/reimplementations/graph_shortest_path.py')
graph=importlib.util.module_from_spec(spec);spec.loader.exec_module(graph)

class GraphTests(unittest.TestCase):
 def setUp(self):
  edges=np.zeros((4,4),bool);edges[0,1]=True;edges[1,2]=True
  self.env=graph.Graph(edges,goal=2)
 def test_invalid_moves_stay(self):self.assertEqual(self.env.step(0,3),(0,-1.,False))
 def test_goal_and_distances(self):
  self.assertEqual(self.env.step(1,2),(2,-1.,True));self.assertEqual(self.env.step(2,0),(2,0.,True))
  np.testing.assert_equal(graph.distances(self.env),[2,1,0,np.inf])
 def test_random_dataset_rtg_is_actual_reward_sum(self):
  paths=graph.random_walks(self.env,1000,10,np.random.RandomState(0));self.assertEqual(len(paths),1000)
  for p in paths:
   self.assertLessEqual(len(p['states']),10)
   np.testing.assert_equal(p['costs'],-np.cumsum(np.asarray(p['rewards'])[::-1])[::-1])
 def test_generator_does_not_require_oracle(self):
  import inspect
  self.assertNotIn('distances(',inspect.getsource(graph.random_walks))
  self.assertNotIn('distances(',inspect.getsource(graph.rollout))
 def test_prior_favors_shorter_costs(self):
  import torch
  posterior=graph.return_posterior(torch.zeros(11),10)
  self.assertAlmostEqual(float(posterior.sum()),1.,places=6)
  self.assertTrue(bool((posterior[:-1]>posterior[1:]).all()))
 def test_future_actions_cannot_leak_into_return_or_action_predictions(self):
  import torch
  model=graph.make_model(4,10,8,1,1,0.);model.eval()
  first=torch.tensor([[0,6,16,1,5,17]])
  changed=first.clone();changed[0,2]=18;changed[0,5]=15
  with torch.no_grad():a=model.encode(first);b=model.encode(changed)
  torch.testing.assert_allclose(a[:,:2],b[:,:2])

if __name__=='__main__':unittest.main()
