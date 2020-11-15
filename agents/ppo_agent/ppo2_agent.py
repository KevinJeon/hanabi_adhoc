
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


class Policy(nn.Module):
  def __init__(self, num_obs, num_action,num_hidden=512):
    super(Policy, self).__init__()
    init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                           constant_(x, 0), nn.init.calculate_gain('relu'))
    self.actor = nn.Sequential(
      init_(nn.Linear(num_obs,num_hidden),nn.Tanh()),
      init_(nn.Linear(num_hidden,num_hidden),nn.Tanh())
    )
    self.critic = nn.Sequential(
      init_(nn.Linear(num_obs, num_hidden), nn.Tanh()),
      init_(nn.Linear(num_hidden, num_hidden), nn.Tanh())
    )
    self.critic_linear = init_(nn.Linear(num_hidden,1))

    self.dist = Categorical(num_hidden, num_action)
  def forward(self, observations):
    raise NotImplementedError

  def act(self, observations, deterministic=False):
    v = self.critic_linear(self.critic(observations))
    h_a = self.actor(observations)
    dist = self.dist(h_a)
    if deterministic:
      action = dist.mode()
    else:
      action = dist.sample()

    action_log_probs = dist.log_probs(action)
    dist_entropy = dist.entropy().mean()

    return v, action, action_log_probs

  def get_value(self, observations):
    v = self.critic_linear(self.critic(observations))
    return v

  def evaluate_actions(self, observations,action):
    v = self.critic_linear(self.critic(observations))
    h_a = self.actor(observations)
    dist = self.dist(h_a)

    action_log_probs = dist.log_probs(action)
    dist_entropy = dist.entropy().mean()

    return v, action_log_probs, dist_entropy

class DBPLAgent(object):
  def __init__(self,num_state,num_action,num_hidden,num_belief,num_player,log_dir,eps=0.2):
    self.policy_modules = [None for _ in range(num_player)]
    self.old_policy_modules = [None for _ in range(num_player)]
    for i in range(num_player):
      self.policy_modules[i] = PolicyNet(num_state,num_action,num_hidden).cuda()
      self.old_policy_modules[i] = PolicyNet(num_state, num_action, num_hidden).cuda()
    self.transitions = [[] for _ in range(num_player)]
    self.num_player = num_player
    self.num_belief = num_belief
    self.num_action = num_action
    self.memory = RolloutStorage(batch_size=64,num_agent=num_player,obs_shape=num_state)
    if not os.path.exists(log_dir):
      os.mkdir(log_dir)
    self.writer = SummaryWriter(log_dir)
    for i in range(num_player):
      self.old_policy_modules[i].load_state_dict(self.policy_modules[i].state_dict())
    # belief
    self.clip_param = clip_param

    self.value_loss_coef = value_loss_coef
    self.entropy_coef = entropy_coef

    self.max_grad_norm = max_grad_norm
    self.use_clipped_value_loss = use_clipped_value_loss

    self.p_optim = [Adam(self.policy_modules[i].parameters(),lr=10e-4,eps=eps) for i in range(num_player)]
    self.gamma = 0.99
    self.k = 3
    self.batch_size = 64
    self.iter = [0]*2
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
        observations,actions,returns,old_logprobs,value_preds,dones,adv_targ\
          = self.memory.get_batch(i)
        if len(observations) == 0:
          self.iter[i] -= 1
          continue
        advantages = returns - value_preds
        if len(rewards) == 1:
          pass
        else:
          advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        #print(len(old_obss),len(old_actions),len(old_logprobs),len(old_beliefs),len(confs.detach()))
        for _ in range(self.k):
          values,action_log_probs,dist_entropy = self.policy_modules[i].evaluate_actions(observations,actions)
          ratio = tr.exp(action_log_probs-old_logprobs.detach())
          surr_loss1 = ratio*adv_targ
          surr_loss2 = tr.clamp(ratio,1-self.clip_param,1+self.clip_param)*adv_targ
          action_loss = -tr.min(surr_loss1, surr_loss2).mean()
          if self.use_clipped_value_loss:
            value_pred_clipped = value_preds + \
                                 (values - value_preds).clamp(-self.clip_param, self.clip_param)
            value_losses = (values - returns).pow(2)
            value_losses_clipped = (
                    value_pred_clipped - returns).pow(2)
            value_loss = 0.5 * tr.max(value_losses,
                                         value_losses_clipped).mean()
          else:
            value_loss = 0.5 * (returns - values).pow(2).mean()
          self.p_optim[i].zero_grad()
          (value_loss * self.value_loss_coef + action_loss -
           dist_entropy * self.entropy_coef).backward()
          nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                   self.max_grad_norm)
          self.p_optim[i].step()
        self.writer.add_scalar('policy_loss/agent{}'.format(i), loss.mean(),self.iter[i])
        #print('agent : {} policy loss : {} belief loss : {}'.format(i,loss.mean().item(),kl_loss.mean().item()))
        self.old_policy_modules[i].load_state_dict(self.policy_modules[i].state_dict())
      self.memory.clear()
  def _select_action(self,current_player,observation,legal_actions):
    observation = tr.from_numpy(observation).float().cuda()
    action,dist = self.old_policy_modules[current_player].act(observation,legal_actions)
    return action,dist.log_prob(action)

