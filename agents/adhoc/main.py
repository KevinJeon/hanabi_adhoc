from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from hanabi_learning_environment import rl_env
import gin.tf
import os
from utils import *
import tensorflow as tf
def adhoc(id1,id2,env,obs_stacker,base_dir):
    g1 = tf.Graph()
    with g1.as_default():
        agent1 = create_agent(env, obs_stacker, agent_type='Rainbow')
        print('agent 1 created!!')
        log_dir1 = base_dir + '{}/logs/'.format(id1)
        ckpt_dir1 = base_dir + '{}/checkpoints/'.format(id1)
        _, ckpt1 = initialize_checkpointing(agent1, log_dir1, ckpt_dir1)
        vars = tf.trainable_variables()
        #print(vars[0],agent1._sess.run(vars[0]))
    g2 = tf.Graph()
    with g2.as_default():
        agent2 = create_agent(env, obs_stacker, agent_type='Rainbow')
        print('agent 2 created!!')
        log_dir2 = base_dir + '{}/logs/'.format(id2)
        ckpt_dir2 = base_dir + '{}/checkpoints/'.format(id2)
        _, ckpt2 = initialize_checkpointing(agent2, log_dir2, ckpt_dir2)
        vars = tf.trainable_variables()

        #print(vars[0],agent2._sess.run(vars[0]))
    epi_rewards = []
    print('agent {} with ad-hoc team {} starts...'.format(id1, id2))
    agents = [agent1, agent2]
    for _ in range(episodes):
        epi_reward = 0
        obs_stacker.reset_stack()
        observations = env.reset()
        current_player, legal_moves, observation_vector = (
            parse_observations(observations, env.num_moves(), obs_stacker))
        action = agents[current_player]._select_action(observation_vector, legal_moves)
        step = 0
        while True:
            step += 1
            observations, reward, is_done, _ = env.step(action.item())
            current_player, legal_moves, observation_vector = (
                parse_observations(observations, env.num_moves(), obs_stacker))
            epi_reward += reward
            if is_done:
                break
            action = agents[current_player]._select_action(observation_vector, legal_moves)
        epi_rewards.append(epi_reward)
    print('agent {} & agent {} adhoc mean {}'.format(id1,id2,np.mean(epi_rewards)))
    return np.mean(epi_rewards)

gin_dir = ['configs/hanabi_rainbow.gin']
gin.parse_config_files_and_bindings(gin_dir,bindings=[],skip_unknown=False)
episodes = 1000
num_player = 2
game_type = 'Hanabi-Full-CardKnowledge'
env = rl_env.make(environment_name=game_type,num_players=num_player,pyhanabi_path=None)
obs_stacker = create_obs_stacker(env)
base_dir = '/home/kevinjeon/hanabi_jeon/hanabi-learning-environment/hanabi_learning_environment/agents/adhoc/agents/'
agents_id = os.listdir(base_dir)
total_rewards = [[] for _ in range(len(agents_id))]
for i,agent_id1 in enumerate(agents_id):
    agent_id_reward = []
    for j, agent_id2 in enumerate(agents_id):
        adhoc_reward = adhoc(agent_id1,agent_id2,env,obs_stacker,base_dir)
        agent_id_reward.append(adhoc_reward)
    total_rewards[i].append(float(agent_id_reward))

score = dict(mat=total_rewards)
print(total_rewards)
import json,io
with io.open('score.json','w',encoding='utf-8') as f:
    json.dump(score,f)