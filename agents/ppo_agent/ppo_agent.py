
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch as tr
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.distributions import Categorical
from replay_memory import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import os

class PolicyNet(nn.Module):
  def __init__(self,num_state,num_action,num_hidden):
    super(PolicyNet,self).__init__()
    # actor
    self.action_layer = nn.Sequential(
      nn.Linear(num_state, num_hidden),
      nn.Tanh(),
      nn.Linear(num_hidden, num_hidden),
      nn.Tanh(),
      nn.Linear(num_hidden, num_action),
      nn.Softmax(dim=-1)
    )

    # critic
    self.value_layer = nn.Sequential(
      nn.Linear(num_state, num_hidden),
      nn.Tanh(),
      nn.Linear(num_hidden, num_hidden),
      nn.Tanh(),
      nn.Linear(num_hidden, 1)
    )

    self.num_state = num_state

  def act(self, state,legal_actions):
    action_probs = self.action_layer(state)
    action_probs = action_probs.cpu().detach().view(-1)
    action_probs[legal_actions == float('-inf')] = 0
    dist = Categorical(action_probs)
    action = dist.sample()
    return action,dist

  def evaluate(self, state, action):
    action_probs = self.action_layer(state)
    dist = Categorical(action_probs)
    action = tr.argmax(action, dim=1)
    action_logprobs = dist.log_prob(action)
    dist_entropy = dist.entropy()
    state_value = self.value_layer(state)
    return action_logprobs, tr.squeeze(state_value), dist_entropy

class DBPLAgent(object):
  def __init__(self,num_state,num_action,num_hidden,num_belief,num_player,log_dir):
    self.policy_modules = [None for _ in range(num_player)]
    self.old_policy_modules = [None for _ in range(num_player)]
    for i in range(num_player):
      self.policy_modules[i] = PolicyNet(num_state,num_action,num_hidden).cuda()
      self.old_policy_modules[i] = PolicyNet(num_state, num_action, num_hidden).cuda()
    self.transitions = [[] for _ in range(num_player)]
    self.num_player = num_player
    self.num_belief = num_belief
    self.num_action = num_action
    self.memory = ReplayBuffer(num_player)
    if not os.path.exists(log_dir):
      os.mkdir(log_dir)
    self.writer = SummaryWriter(log_dir)
    for i in range(num_player):
      self.old_policy_modules[i].load_state_dict(self.policy_modules[i].state_dict())
    # belief
    self.p_optim = [Adam(self.policy_modules[i].parameters(),lr=10e-4*2,betas=(0.9,0.999)) for i in range(num_player)]
    self.gamma = 0.99
    self.epsilon = 0.2
    self.k = 3
    self.batch_size = 64
    self.iter = [0]*2
    self.mse_loss = nn.MSELoss()
  def step(self,reward,current_player,legal_moves,observation_vector,is_done,begin=False):

    self._train()
    # train step
    self.action,logprobs  = self._select_action(current_player,observation_vector,legal_moves)
    self.memory.add_count += 1
    if begin:
      self.memory.add(current_player,np.array(observation_vector,dtype=np.uint8,copy=True),
                      np.eye(self.num_action)[self.action],0,is_done,legal_moves,logprobs)
    else:
      self.memory.add(current_player, np.array(observation_vector, dtype=np.uint8, copy=True),
                      np.eye(self.num_action)[self.action], reward, is_done, legal_moves, logprobs)
    return self.action,logprobs

  def end_episode(self,reward_since_last_action):
    for i,reward in enumerate(reward_since_last_action):
      self.memory.rews[i].append(reward)
      self.memory.rews[i] = self.memory.rews[i][1:]
    self._train(end=True)

  def _train(self,end=False):
    if self.eval_mode:
      return
    if (self.memory.add_count >= self.batch_size) or (end == True):
      #print('end',end,'memory_count',self.memory.add_count)
      for i in range(self.num_player):
        self.iter[i] += 1
        # train policy module
        observations,actions,rewards,logprobs,dones = self.memory.get_batch(i)
        if len(observations) == 0:
          self.iter[i] -= 1
          continue
        discounted_rewards = []
        discounted = 0
        #print('Rewards ',rewards)
        for reward,done in zip(reversed(rewards),reversed(dones)):
          if done:
            discounted = 0
          discounted = reward + (self.gamma*discounted)
          discounted_rewards.insert(0,discounted)
        rewards = tr.tensor(discounted_rewards,dtype=tr.float32).cuda()
        if len(rewards) == 1:
          pass
        else:
          rewards = (rewards-rewards.mean())/(rewards.std()+1e-5)
        old_obss = observations.cuda().detach()
        old_actions = actions.cuda().detach()
        old_logprobs = logprobs.cuda().detach()
        #print(len(old_obss),len(old_actions),len(old_logprobs),len(old_beliefs),len(confs.detach()))
        for _ in range(self.k):
          logprobs,values,entropy = self.policy_modules[i].evaluate(old_obss,old_actions)
          #logprobs,values,entropy = logprobs.cpu(),values.cpu(),entropy.cpu()
          ratio = tr.exp(logprobs-old_logprobs.detach())
          advantages = rewards - values.detach()
          surr_loss1 = ratio*advantages
          surr_loss2 = tr.clamp(ratio,1-self.epsilon,1+self.epsilon)*advantages
          loss = - tr.min(surr_loss1,surr_loss2)+0.5*self.mse_loss(values,rewards)-0.05*entropy
          self.p_optim[i].zero_grad()
          loss.mean().backward()
          self.p_optim[i].step()
        self.writer.add_scalar('policy_loss/agent{}'.format(i), loss.mean(),self.iter[i])
        #print('agent : {} policy loss : {} belief loss : {}'.format(i,loss.mean().item(),kl_loss.mean().item()))
        self.old_policy_modules[i].load_state_dict(self.policy_modules[i].state_dict())
      self.memory.clear()
  def _select_action(self,current_player,observation,legal_actions):
    observation = tr.from_numpy(observation).float().cuda()
    action,dist = self.old_policy_modules[current_player].act(observation,legal_actions)
    return action,dist.log_prob(action)

