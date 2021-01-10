from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from hanabi_learning_environment import rl_env
import gin.tf
import os
from utils import *
import tensorflow as tf
import json
with open('score.json' ,'r') as f:
    jd = json.load(f)
    obss = jd['obss']
gin_dir = ['configs/hanabi_sac.gin']
gin.parse_config_files_and_bindings(gin_dir,bindings=[],skip_unknown=False)
episodes = 1000
num_player = 2
game_type = 'Hanabi-Full-CardKnowledge'
env = rl_env.make(environment_name=game_type,num_players=num_player,pyhanabi_path=None)
obs_stacker = create_obs_stacker(env)
base_dir = '/home/kevinjeon/hanabi_jeon/hanabi-learning-environment/hanabi_learning_environment/agents/adhoc/agents/'
agents_id = os.listdir(base_dir)
agents = []
features = dict()
for i,agent_id1 in enumerate(agents_id):
    agent_id_reward = []
    g1 = tf.Graph()
    features['agent{}'.format(i)] = dict(first=[],second=[],q=[])
    with g1.as_default():
        agent1 = create_agent(env, obs_stacker, agent_type='Rainbow')
        print('agent 1 created!!')
        log_dir1 = base_dir + '{}/logs/'.format(agent_id1)
        ckpt_dir1 = base_dir + '{}/checkpoints/'.format(agent_id1)
        _, ckpt1 = initialize_checkpointing(agent1, log_dir1, ckpt_dir1)
        for j,obs in enumerate(obss):
            obs = np.reshape(obs,(1,-1,1))
            prevs = agent1._sess.run(agent1._prevs,{agent1.state_ph:obs})
            features['agent{}'.format(i)]['first'].append(prevs[0][0].tolist())
            features['agent{}'.format(i)]['second'].append(prevs[1][0].tolist())
            qs = agent1._sess.run(agent1._q, {agent1.state_ph: obs})

            features['agent{}'.format(i)]['q'].append(qs[0].tolist())

with open('features.json','w') as f:
    json.dump(features,f)