from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from hanabi_learning_environment import rl_env
import gin.tf
import os
from utils import *
import tensorflow as tf
def adhoc(chkpt1,chkpt2,env,obs_stacker,base_dir):
    log_dir = base_dir + '{}/logs/'.format(0)
    ckpt_dir = base_dir + '{}/checkpoints/'.format(0)
    g1 = tf.Graph()
    with g1.as_default():
        agent1 = create_agent(env, obs_stacker, agent_type='Rainbow')
        print('agent 1 created!!')
        _, ckpt1 = initialize_checkpointing(agent1, log_dir, ckpt_dir, chkpt1)
        vars = tf.trainable_variables()
        #print(vars[0],agent1._sess.run(vars[0]))
    g2 = tf.Graph()
    with g2.as_default():
        agent2 = create_agent(env, obs_stacker, agent_type='Rainbow')
        print('agent 2 created!!')
        _, ckpt2 = initialize_checkpointing(agent2, log_dir, ckpt_dir, chkpt2)
        vars = tf.trainable_variables()

        #print(vars[0],agent2._sess.run(vars[0]))
    epi_rewards = []
    print('agent {} with ad-hoc team {} starts...'.format(chkpt1, chkpt2))
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
    print('agent 0 checkpoint {} & {} adhoc mean {}'.format(chkpt1,chkpt2,np.mean(epi_rewards)))
    return np.mean(epi_rewards)

gin_dir = ['configs/hanabi_rainbow2.gin']
gin.parse_config_files_and_bindings(gin_dir,bindings=[],skip_unknown=False)
episodes = 1000
num_player = 2
game_type = 'Hanabi-Full-CardKnowledge'
env = rl_env.make(environment_name=game_type,num_players=num_player,pyhanabi_path=None)
obs_stacker = create_obs_stacker(env)
base_dir = '/home/kevinjeon/hanabi_jeon/hanabi-learning-environment/hanabi_learning_environment/agents/adhoc/agents/'
agents_id = os.listdir(base_dir)
total_rewards = [[] for _ in range(len(agents_id))]
ckpts = [1000*i+900 for i in range(10)]
for i,ckpt1 in enumerate(ckpts):
    agent_id_reward = []
    for j, ckpt2 in enumerate(ckpts):
        adhoc_reward = adhoc(ckpt1,ckpt2,env,obs_stacker,base_dir)
        agent_id_reward.append(adhoc_reward)
    total_rewards[i].append(float(agent_id_reward))

score = dict(mat=total_rewards)
print(total_rewards)
import json,io
with io.open('score2.json','w',encoding='utf-8') as f:
    json.dump(score,f)