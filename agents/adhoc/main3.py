from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from hanabi_learning_environment import rl_env
import gin.tf
import os
from utils import *
import tensorflow as tf

"""Simple Agent."""

from hanabi_learning_environment.rl_env import Agent


class SimpleAgent(Agent):
  """Agent that applies a simple heuristic."""

  def __init__(self, config, *args, **kwargs):
    """Initialize the agent."""
    self.config = config
    # Extract max info tokens or set default to 8.
    self.max_information_tokens = config.get('information_tokens', 8)

  @staticmethod
  def playable_card(card, fireworks):
    """A card is playable if it can be placed on the fireworks pile."""
    return card['rank'] == fireworks[card['color']]

  def act(self, observation):
    """Act based on an observation."""
    if observation['current_player_offset'] != 0:
      return None

    # Check if there are any pending hints and play the card corresponding to
    # the hint.
    for card_index, hint in enumerate(observation['card_knowledge'][0]):
      if hint['color'] is not None or hint['rank'] is not None:
        return {'action_type': 'PLAY', 'card_index': card_index}

    # Check if it's possible to hint a card to your colleagues.
    fireworks = observation['fireworks']
    if observation['information_tokens'] > 0:
      # Check if there are any playable cards in the hands of the opponents.
      for player_offset in range(1, observation['num_players']):
        player_hand = observation['observed_hands'][player_offset]
        player_hints = observation['card_knowledge'][player_offset]
        # Check if the card in the hand of the opponent is playable.
        for card, hint in zip(player_hand, player_hints):
          if SimpleAgent.playable_card(card,
                                       fireworks) and hint['color'] is None:
            return {
                'action_type': 'REVEAL_COLOR',
                'color': card['color'],
                'target_offset': player_offset
            }

    # If no card is hintable then discard or play.
    if observation['information_tokens'] < self.max_information_tokens:
      return {'action_type': 'DISCARD', 'card_index': 0}
    else:
      return {'action_type': 'PLAY', 'card_index': 0}

def adhoc(id1,id2,env,obs_stacker,base_dir):
    if id1 == 'simple':
        cfg = dict(players=num_player)
        agent1 = SimpleAgent(cfg)
    else:
        g1 = tf.Graph()
        with g1.as_default():
            agent1 = create_agent(env, obs_stacker, agent_type='Rainbow')
            print('agent 1 created!!')
            log_dir1 = base_dir + '{}/logs/'.format(id1)
            ckpt_dir1 = base_dir + '{}/checkpoints/'.format(id1)
            _, ckpt1 = initialize_checkpointing(agent1, log_dir1, ckpt_dir1)
            vars = tf.trainable_variables()
            #print(vars[0],agent1._sess.run(vars[0]))
    cfg = dict(players=num_player)
    agent2 = SimpleAgent(cfg)
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
            observations, reward, is_done, _ = env.step(action)
            current_player, legal_moves, observation_vector = (
                parse_observations(observations, env.num_moves(), obs_stacker))
            epi_reward += reward
            if is_done:
                break
            if (current_player) == 1 or (id1=='simple'):
                observation = observations['player_observations'][current_player]
                action = agents[current_player].act(observation)
            else:
                action = agents[current_player]._select_action(observation_vector, legal_moves)
                action = action.item()
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
agents_id = os.listdir(base_dir) + ['simple']
total_rewards = [[] for _ in range(len(agents_id))]
for i,agent_id1 in enumerate(agents_id[-1]):
    agent_id_reward = []
    adhoc_reward = adhoc(agent_id1,'simple',env,obs_stacker,base_dir)
    agent_id_reward.append(adhoc_reward)
    total_rewards[i].append(float(agent_id_reward))

score = dict(mat=total_rewards)
print(total_rewards)
import json,io
with io.open('score3.json','w') as f:
    json.dump(score,f)