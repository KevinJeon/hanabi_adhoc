
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

class BeliefNet(nn.Module):
  def __init__(self,num_state,num_action,num_hidden,num_belief):
    super(BeliefNet,self).__init__()
    self.fc1 = nn.Linear(num_state+num_action, num_hidden)
    self.gru = nn.GRUCell(num_hidden, num_hidden)
    self.fc2 = nn.Linear(num_hidden, num_belief)
    self.num_hidden = num_hidden
    self.num_state = num_state
  def init_hidden(self):
    return self.fc1.weight.new(1, self.num_hidden).zero_()

  def forward(self, observation,action, hidden_state):
    o = observation.view(-1, self.num_state)
    x = tr.cat([o,action],axis=-1)
    x = F.relu(self.fc1(x)).view(-1,self.num_hidden)
    h_in = hidden_state.reshape(-1, self.num_hidden)
    h = self.gru(x, h_in)
    o = self.fc2(h).view(-1,5,25)
    b = F.softmax(o,dim=-1)
    b = b.view(-1,125)
    return b, h

class PolicyNet(nn.Module):
  def __init__(self,num_state,num_action,num_hidden,num_belief):
    super(PolicyNet,self).__init__()
    self.fc1 = nn.Linear(num_state+num_belief, num_hidden)
    self.fc2 = nn.Linear(num_hidden, num_action)
    self.fc2_v = nn.Linear(num_hidden,1)
    self.belief_confidence = nn.GRUCell(num_belief,num_belief)
    self.num_belief = num_belief
    self.num_state = num_state
  def init_confidence(self):
    return tr.zeros(1,self.num_belief)
  def policy(self, observation,belief,h):
    h = F.sigmoid(self.belief_confidence(belief,h))
    o = observation.view(-1,self.num_state)
    x = tr.cat([o,h],axis=-1)
    x = F.relu(self.fc1(x))
    x = F.softmax(self.fc2(x),dim=-1)
    return x,h
  def evaluate(self,observation,belief,action,h):
    b = self.belief_confidence(belief,h)
    x = tr.cat([observation, b], axis=-1)
    x = F.relu(self.fc1(x))
    v = self.fc2_v(x)
    p,_ = self.policy(observation,belief,h)
    dist = Categorical(p)
    action = tr.argmax(action,dim=1)
    log_p = dist.log_prob(action)
    dist_entropy = dist.entropy()
    return log_p,v,dist_entropy

class DBPLAgent(object):
  def __init__(self,num_state,num_action,num_hidden,num_belief,num_player,log_dir):
    self.belief_module = BeliefNet(num_state,num_action,num_hidden,num_belief)
    self.belief_module = self.belief_module.cuda()
    self.policy_modules = [PolicyNet(num_state,num_action,num_hidden,num_belief) for _ in range(num_player)]
    self.policy_modules = [self.policy_modules[i].cuda() for i in range(num_player)]
    self.old_policy_modules = [PolicyNet(num_state, num_action, num_hidden, num_belief) for _ in range(num_player)]
    self.old_policy_modules = [self.old_policy_modules[i].cuda() for i in range(num_player)]
    self.h = [self.belief_module.init_hidden() for _ in range(num_player)]
    self.c = [self.policy_modules[i].init_confidence() for i in range(num_player)]
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
    self.belief = [None for _ in range(num_player)]
    self.b_optim = Adam(self.belief_module.parameters(),lr=10e-3)
    self.p_optim = [Adam(self.policy_modules[i].parameters(),lr=10e-3,betas=(0.9,0.999)) for i in range(num_player)]
    self.gamma = 0.99
    self.epsilon = 0.2
    self.k = 3
    self.batch_size = 64
    self.iter = [0]*2
  def begin_episode(self,current_player,legal_moves,observation_vector):
    # train step?
    self._train()
    self.init_belief()
    self.action,logprobs = self._select_action(current_player,observation_vector,
                                               self.belief[current_player],legal_moves)
    self.h = [self.belief_module.init_hidden() for _ in range(self.num_player)]
    self.c = [self.policy_modules[i].init_confidence() for i in range(self.num_player)]
    self.memory.add_count += 1
    return self.action,logprobs,self.belief[current_player],self.c[current_player].detach().numpy()

  def step(self,reward,current_player,legal_moves,observation_vector):
    self._train()
    # train step
    self.action,logprobs  = self._select_action(current_player,observation_vector,
                                                self.belief[current_player],legal_moves)
    self.memory.add_count += 1
    return self.action,logprobs,self.belief[current_player],self.c[current_player].cpu().detach().numpy()

  def end_episode(self,final_rewards):
    self._train(end=True)

  def _train(self,end=False):
    if self.eval_mode:
      return
    if (self.memory.add_count >= self.batch_size) or (end == True):
      #print('end',end,'memory_count',self.memory.add_count)
      for i in range(self.num_player):
        self.iter[i] += 1
        # train belief module
        ## if batch == 0 : no training
        for j in range(self.num_player):
          if i == j :
            continue
          mimic_obs,true_label,other_actions = self.memory.get_other_view(i,j,self.num_player,self.num_action,self.num_belief)
          if len(mimic_obs) == 0:
            self.iter[i] -= 1
            continue
          h = self.belief_module.init_hidden().expand(len(mimic_obs),-1)
          mimic_obs = tr.tensor(mimic_obs).float().cuda()
          other_actions = tr.tensor(other_actions).float().cuda()
          true_label = tr.tensor(true_label).float().cuda()
          #print('agent',i,'mimic',mimic_obs.size(),'other_actions',other_actions.size(),'label',true_label.size())
          beliefs,_ = self.belief_module(mimic_obs,other_actions,h)
          #print(beliefs[0],true_label[0])
          kl_loss = F.binary_cross_entropy(beliefs,true_label)
          self.b_optim.zero_grad()
          kl_loss.mean().backward()
          self.b_optim.step()
          self.writer.add_scalar('belief_loss/agent{}'.format(i),kl_loss.mean(),self.iter[i])
        # train policy module
        observations,actions,rewards,beliefs,logprobs,dones,confs = self.memory.get_batch(i)
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
        rewards = tr.tensor(discounted_rewards,dtype=tr.float32)
        if len(rewards) == 1:
          pass
        else:
          rewards = (rewards-rewards.mean())/(rewards.std()+1e-5)
        old_obss = observations.detach().cuda()
        old_actions = actions.detach().cuda()
        old_logprobs = logprobs.detach()
        old_beliefs = beliefs.detach().cuda()
        #print(len(old_obss),len(old_actions),len(old_logprobs),len(old_beliefs),len(confs.detach()))
        old_confs = confs.detach().squeeze(1).cuda()
        for _ in range(self.k):
          logprobs,values,entropy = self.policy_modules[i].evaluate(old_obss,old_beliefs,old_actions,old_confs)
          logprobs,values,entropy = logprobs.cpu(),values.cpu(),entropy.cpu()
          ratio = tr.exp(logprobs-old_logprobs.detach())
          advantages = rewards - values.detach()
          surr_loss1 = ratio*advantages
          surr_loss2 = tr.clamp(ratio,1-self.epsilon,1-self.epsilon)*advantages
          loss = - tr.min(surr_loss1,surr_loss2)+0.5*F.mse_loss(values,rewards)-0.01*entropy
          self.p_optim[i].zero_grad()
          loss.mean().backward()
          self.p_optim[i].step()
        self.writer.add_scalar('policy_loss/agent{}'.format(i), loss.mean(),self.iter[i])
        #print('agent : {} policy loss : {} belief loss : {}'.format(i,loss.mean().item(),kl_loss.mean().item()))
        self.old_policy_modules[i].load_state_dict(self.policy_modules[i].state_dict())
      self.memory.clear()
  def _select_action(self,current_player,observation,belief,legal_actions):
    observation = tr.from_numpy(observation).float().cuda()
    belief = tr.tensor(belief).float().view(1,-1).cuda()
    p,self.c[current_player] = self.old_policy_modules[current_player].policy(observation,belief,self.c[current_player].cuda())
    legal_actions[legal_actions==0] = 1
    legal_actions[legal_actions==float('-inf')] = 0
    p = p.cpu().detach() * legal_actions
    dist = Categorical(p)
    action = dist.sample()
    return action,dist.log_prob(action)
  def save_all(self,action,observations):
    for n,(obs,bel) in enumerate(zip(observations,self.belief)):
      self.memory.save_all(n,obs,action,bel)
  def init_belief(self):
    self.belief = [[1/25]*self.num_belief for _ in range(self.num_player)]
  def update_others_belief(self,current_player,all_observations,action):
    for n,obs in enumerate(all_observations):
      if n == current_player:
        continue
      obs = tr.tensor(obs).float().cuda()
      one_hot_action = tr.tensor(np.eye(self.num_action)[action.numpy()]).float().cuda()
      self.belief[n],self.h[n] = self.belief_module(obs,one_hot_action,self.h[n])
      self.belief[n] = self.belief[n].view(-1).cpu().detach().numpy()

