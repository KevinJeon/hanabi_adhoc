import numpy as np
import copy
import torch as tr
class ReplayBuffer(object):
    def __init__(self,num_player,capacity=50000):
        # in own turn
        self.obss = [[] for _ in range(num_player)]
        self.acts = [[] for _ in range(num_player)]
        self.rews = [[] for _ in range(num_player)]
        self.dones = [[] for _ in range(num_player)]
        self.legals = [[] for _ in range(num_player)]
        self.pis = [[] for _ in range(num_player)]
        self.bels = [[] for _ in range(num_player)]
        self.confs = [[] for _ in range(num_player)]
        # all turn
        self.o_obss = [[] for _ in range(num_player)]
        self.o_acts = [[] for _ in range(num_player)]
        self.o_bels = [[] for _ in range(num_player)]
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
        self.bels = [[] for _ in range(self.num_player)]
        self.confs = [[] for _ in range(self.num_player)]
        # all turn
        self.o_obss = [[] for _ in range(self.num_player)]
        self.o_acts = [[] for _ in range(self.num_player)]
        self.o_bels = [[] for _ in range(self.num_player)]
        self.add_count = 0
    def add(self,curr,obs,act,rew,done,legal,belief,p,conf):
        self.obss[curr].append(obs)
        self.acts[curr].append(act)
        self.rews[curr].append(rew)
        self.dones[curr].append(done)
        self.legals[curr].append(legal)
        self.bels[curr].append(belief)
        self.pis[curr].append(p)
        self.confs[curr].append(conf)
    def save_all(self,curr,obs,act,bel):
        self.o_obss[curr].append(obs)
        self.o_acts[curr].append(act)
        self.o_bels[curr].append(bel)
    def get_other_view(self,curr,other,num_player,num_action,num_belief=125):
        '''

        Args:
            curr: view of the agent
            other: target to mimic agent
            num_player:
            num_belief:
        Returns:
            mimics: common knowledge + inference of the curr agent
            labels: card of the target other agent
            acts: act of the other agents (view of the other agent)
        '''
        mimics = []
        real_acts = []
        labels = []
        obss = np.array(self.o_obss[curr])
        acts = np.array(self.o_acts[curr])
        bels = np.array(self.o_bels[curr])
        for n,(obs,act,bel) in enumerate(zip(obss,acts,bels)):
            if n % num_player != other:
                mimic = copy.deepcopy(obs)
                offset = 0
                mimic[num_belief*offset:num_belief*(offset+1)] = np.reshape(bel,(1,-1))
                #print('belief',np.reshape(bel,(1,-1)))
                #print(curr,other)
                #print('offset',offset)
                #print('correct',obs[num_belief*offset:num_belief*(offset+1)])
                mimics.append(mimic)
                act_onehot = np.eye(num_action)[act]
                real_acts.append(act_onehot)
                labels.append(obs[num_belief*offset:num_belief*(offset+1)])
        mimics = np.array(mimics)
        labels = np.array(labels)
        real_acts = np.array(real_acts)
        #print('get_other_view',np.shape(mimics),np.shape(labels),np.shape(real_acts))
        return mimics,labels,real_acts
    def get_batch(self,curr):
        obss = tr.tensor(self.obss[curr]).float()
        acts = tr.tensor(self.acts[curr]).float()
        rews = self.rews[curr]
        dones = self.dones[curr]
        bels = tr.tensor(self.bels[curr]).float()
        pis = tr.tensor(self.pis[curr]).float()
        confs = tr.tensor(self.confs[curr]).float()
        return obss,acts,rews,bels,pis,dones,confs