
import torch as tr
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
class ReplayBuffer(object):
    def __init__(self,num_player,capacity=50000):
        # in own turn
        self.obss = [[] for _ in range(num_player)]
        self.acts = [[] for _ in range(num_player)]
        self.rews = [[] for _ in range(num_player)]
        self.dones = [[] for _ in range(num_player)]
        self.legals = [[] for _ in range(num_player)]
        self.pis = [[] for _ in range(num_player)]
        # all turn
        self.capcity = capacity
        self.offset = [[None,0,1,2],[2,None,0,1],[1,2,None,0],[0,1,2,None]]
        self.add_count = 0
        self.num_player = num_player
    def clear(self):
        self.obss = [[] for _ in range(self.num_player)]
        self.acts = [[] for _ in range(self.num_player)]
        self.rews = [[] for _ in range(self.num_player)]
        self.dones = [[] for _ in range(self.num_player)]
        self.legals = [[] for _ in range(self.num_player)]
        self.pis = [[] for _ in range(self.num_player)]
        self.add_count = 0
    def add(self,curr,obs,act,rew,done,legal,p):
        self.obss[curr].append(obs)
        self.acts[curr].append(act)
        self.rews[curr].append(rew)
        self.dones[curr].append(done)
        self.legals[curr].append(legal)
        self.pis[curr].append(p)
    def compute_return(self,next_value,use_gae,gamma,gae_lambda,use_proper_time_limits):
        ## to fix
        if use_proper_time_limits:
            if use_gae:
                self.vals[-1] = next_value
                gae = 0
                for step in reversed(range(self.rews.size(0))):
                    delta = self.rews[step] + gamma*self.vals[step+1]-self.vals[step]
                    gae = delta + gamma*gae_lambda*gae
                    self.rets[step] = gae + self.vals[step]
            else:
                self.rets[-1] = next_value
                for step in reversed(range(self.rews.size(0))):
                    self.rets[step] = self.vals[step]
    def get_batch(self,curr,advantages):
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)),mini_batch_size,drop_last=True)
        for indices in sampler:
            obss = self.obss[curr][:-1].view(-1,*self.obss[curr].size()[2:])[indices]
            acts = self.acts[curr].view(-1,self.acts[curr].size())[indices]
            vals = self.vals[curr][:-1].view(-1,1)[indices]
            rets = self.rets[curr][:-1].view(-1,1)[indices]
            old_log_probs_batch = self.log_probs[curr].view(-1,1)[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.view(-1, 1)[indices]

            yield obss, acts, vals, rets, old_log_probs_batch, adv_targ
