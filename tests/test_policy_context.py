'''读取已记录轨迹和对应 checkpoint，验证还原的 K 步输入能重现实际动作。'''
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'gym'))
from decision_transformer.models.decision_transformer import DecisionTransformer
from review_experiment import load_episode, policy_context

if __name__=='__main__':
    run=sorted((ROOT/'reproduction/trajectory_validation_runs').iterdir())[-1]
    path=run/'trajectories/iteration_0002_target_3600_episode_0001.npz'
    arrays,metadata=load_episode(path)
    checkpoint=torch.load(run/metadata['checkpoint'],map_location='cpu')
    config=checkpoint['variant']
    model=DecisionTransformer(state_dim=11,act_dim=3,max_length=config['K'],max_ep_len=checkpoint['max_ep_len'],
        hidden_size=config['embed_dim'],n_layer=config['n_layer'],n_head=config['n_head'],
        n_inner=4*config['embed_dim'],activation_function=config['activation_function'],n_positions=1024,
        resid_pdrop=config['dropout'],attn_pdrop=config['dropout']).to('cuda')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    for timestep in [0,len(arrays['reward'])-1]:
        context=policy_context(path,timestep)
        assert np.array_equal(context['actions_history'][0,-1],np.zeros(3))
        tensors={key:torch.from_numpy(value).to('cuda') for key,value in context.items()}
        with torch.no_grad():
            _,predictions,_=model(tensors['states_normalized'],tensors['actions_history'],None,
                tensors['rtg_scaled'],tensors['timesteps'],attention_mask=tensors['attention_mask'])
        assert np.allclose(predictions[0,-1].cpu().numpy(),arrays['action'][timestep],atol=1e-6)
    print('PASS: reconstructed full policy context + matching iteration checkpoint reproduces actual actions')
