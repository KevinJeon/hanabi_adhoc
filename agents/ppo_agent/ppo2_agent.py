
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import torch as tr
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from replay_memory import RolloutStorage
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import os

def init(layer,actor=False):
  if actor:
    nn.init.orthogonal_(layer.weight.data, gain=0.01)
    nn.init.constant_(layer.bias.data, 0)

  elif type(layer) == nn.Linear:
    nn.init.orthogonal_(layer.weight.data,gain=np.sqrt(2))
    nn.init.constant_(layer.bias.data,0)

class Policy(nn.Module):
  def __init__(self, num_obs, num_action,num_hidden=512):
    super(Policy, self).__init__()
    self.actor = nn.Sequential(
      nn.Linear(num_obs,num_hidden),nn.Tanh(),
      nn.Linear(num_hidden,num_hidden),nn.Tanh(),
    )
    self.actor.apply(init)
    self.critic = nn.Sequential(
      nn.Linear(num_obs, num_hidden), nn.Tanh(),
      nn.Linear(num_hidden, num_hidden), nn.Tanh()
    )
    self.critic.apply(init)
    self.critic_linear = nn.Linear(num_hidden,1)
    self.actor_linear = nn.Linear(num_hidden,num_action)
    init(self.critic_linear)
    init(self.actor_linear,actor=True)
  def forward(self, observations):
    raise NotImplementedError

  def act(self, observations, legal_actions):
    v = self.critic_linear(self.critic(observations))
    h_a = self.actor(observations)
    act_probs = F.sigmoid(self.actor_linear(h_a))
    act_probs[legal_actions != 0] = 0
    dist = Categorical(act_probs)
    action = dist.sample().unsqueeze(-1)
    action_log_probs = dist.log_prob(action)
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
  def __init__(self,num_state,num_action,num_hidden,num_belief,num_player,log_dir,
               eps=1e-5,clip_param=0.2,value_loss_coef=0.5,entropy_coef=0.01,max_grad_norm=0.5,
               use_clipped_value_loss=True):
    self.policy_modules = [None for _ in range(num_player)]
    self.old_policy_modules = [None for _ in range(num_player)]
    for i in range(num_player):
      self.policy_modules[i] = Policy(num_state,num_action,num_hidden).cuda()
      self.old_policy_modules[i] = Policy(num_state, num_action, num_hidden).cuda()
    self.num_player = num_player
    self.num_belief = num_belief
    self.num_action = num_action
    self.memory = RolloutStorage(batch_size=64,num_agent=num_player,obs_shape=num_state)
    self.memory.cuda()
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
    self.k = 4
    self.batch_size = 640
    self.iter = [0]*2
  def step(self,reward,current_player,legal_moves,observation_vector,is_done,begin=False):

    self._train()
    # train step
    value, self.action, logprobs  = self._select_action(current_player,observation_vector,legal_moves)
    
    return value, self.action, logprobs

  def end_episode(self,reward_since_last_action):
    self._train(end=True)

  def _train(self,end=False):
    if self.eval_mode:
      return
    if (self.memory.step >= self.batch_size) or (end == True):
      #print('end',end,'memory_count',self.memory.add_count)
      for i in range(self.num_player):
        self.iter[i] += 1
        advantages = self.memory.returns[i][:-1] - self.memory.value_preds[i][:-1]
        advantages = (advantages - advantages.mean())/ (advantages.std()+1e-5)
        # train policy module
        avg_val_loss = 0
        avg_act_loss = 0
        avg_dist = 0
        for _ in range(self.k):
          sampler = self.memory.feed_forward_generator(advantages,self.batch_size)
          for sample in sampler:
            obss,acts,val_preds,rets,masks,old_probs,adv_targ = sample
            vals,log_probs,dist_entropy,_ = self.policy_modules[i].evaluate_actions(obss,masks,acts)
            ratio = tr.exp(log_probs-old_probs.detach())
            surr_loss1 = ratio*adv_targ
            surr_loss2 = tr.clamp(ratio,1-self.clip_param,1+self.clip_param)*adv_targ
            action_loss = -tr.min(surr_loss1, surr_loss2).mean()
            if self.use_clipped_value_loss:
              value_pred_clipped = val_preds + \
                                   (vals - val_preds).clamp(-self.clip_param, self.clip_param)
              value_losses = (vals - rets).pow(2)
              value_losses_clipped = (value_pred_clipped - rets).pow(2)
              value_loss = 0.5 * tr.max(value_losses,value_losses_clipped).mean()
            else:
              value_loss = 0.5 * (rets - vals).pow(2).mean()
            self.p_optim[i].zero_grad()
            (value_loss * self.value_loss_coef + action_loss -
             dist_entropy * self.entropy_coef).backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                     self.max_grad_norm)
            self.p_optim[i].step()
            avg_val_loss += value_loss.item()
            avg_dist += dist_entropy.item()
            avg_act_loss += action_loss.item()
        avg_val_loss /= self.k*self.batch_size
        avg_act_loss /= self.k*self.batch_size
        avg_dist /= self.k * self.batch_size
        self.writer.add_scalar('action_loss/agent{}'.format(i), avg_act_loss,self.iter[i])
        self.writer.add_scalar('value_loss/agent{}'.format(i), avg_val_loss,self.iter[i])
        self.writer.add_scalar('entropy/agent{}'.format(i), avg_dist, self.iter[i])
        #print('agent : {} policy loss : {} belief loss : {}'.format(i,loss.mean().item(),kl_loss.mean().item()))
        self.old_policy_modules[i].load_state_dict(self.policy_modules[i].state_dict())
      self.memory.after_update()
  def _select_action(self,current_player,observation,legal_actions):
    observation = tr.from_numpy(observation).float().cuda()
    with tr.no_grad():
      value,action,log_probs = self.old_policy_modules[current_player].act(observation,legal_actions)
    print(value.size(),action.size(),log_probs.size())
    return value,action,log_probs

