from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from hanabi_learning_environment import rl_env
import gin.tf
import os
from utils import *
import tensorflow as tf

def get_feature(agents,env,obs_stacker):
    features = dict()
    obss = []
    for i in range(len(agents)):
        features['agent{}'.format(i)] = dict(first=[],second=[],q=[])
    for _ in range(1):
        epi_reward = 0
        obs_stacker.reset_stack()
        observations = env.reset()
        current_player, legal_moves, observation_vector = (
            parse_observations(observations, env.num_moves(), obs_stacker))
        state = np.reshape(observation_vector,(1,-1,1))
        for i,agent in enumerate(agents):
            prevs = agent._sess.run(agent._prevs, {agent.state_ph: state})
            first_layer,second_layer = prevs
            qs = agent._sess.run(agent._prevs, {agent.state_ph: state})
            features['agent{}'.format(i)]['first'].append(first_layer)
            features['agent{}'.format(i)]['second'].append(second_layer)
            features['agent{}'.format(i)]['q'].append(qs)
        action = agents[0]._select_action(observation_vector, legal_moves)
        step = 0
        obss.append(list(observation_vector))
        while True:
            step += 1
            observations, reward, is_done, _ = env.step(action.item())
            current_player, legal_moves, observation_vector = (
                parse_observations(observations, env.num_moves(), obs_stacker))
            epi_reward += reward
            if is_done:
                break
            action = agents[0]._select_action(observation_vector, legal_moves)
            obss.append(list(observation_vector))
    return obss

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
for i,agent_id1 in enumerate(agents_id):
    agent_id_reward = []
    g1 = tf.Graph()
    with g1.as_default():
        agent1 = create_agent(env, obs_stacker, agent_type='Rainbow')
        print('agent 1 created!!')
        log_dir1 = base_dir + '{}/logs/'.format(agent_id1)
        ckpt_dir1 = base_dir + '{}/checkpoints/'.format(agent_id1)
        _, ckpt1 = initialize_checkpointing(agent1, log_dir1, ckpt_dir1)
        #prevs = agent1._sess.run(agent1._prevs,{agent1.state_ph:observation_vector})
        #qs = agent1._sess.run(agent1._prevs, {agent1.state_ph: observation_vector})
    agents.append(agent1)
obss = get_feature(agents,env,obs_stacker)
obss = dict(obss=obss)
import json
with open('obss.json','w') as f:
    json.dump(obss,f)