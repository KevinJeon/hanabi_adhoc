import torch as tr
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class RolloutStorage(object):
    def __init__(self, batch_size , num_agent, obs_shape):
        self.obss = tr.zeros(num_agent, batch_size + 1, obs_shape)
        self.rews = tr.zeros(num_agent, batch_size, 1)
        self.value_preds = tr.zeros(num_agent, batch_size + 1, 1)
        self.rets = tr.zeros(num_agent, batch_size + 1, 1)
        self.action_log_probs = tr.zeros(num_agent, batch_size, 1)
        self.legal_actions = tr.zeros(num_agent,batch_size,1)
        self.acts = tr.zeros(num_agent,batch_size, 1)
        self.acts = self.acts.long()
        self.masks = tr.ones(num_agent,batch_size + 1, 1)

        # Masks that indicate whether it's a true terminal state
        # or time limit end state
        self.bad_masks = tr.ones(num_agent,batch_size + 1, 1)
        self.batch_size = batch_size
        self.step = 0

    def cuda(self):
        self.obss = self.obss.cuda()
        self.rews = self.rews.cuda()
        self.value_preds = self.value_preds
        self.rets = self.rets.cuda()
        self.action_log_probs = self.action_log_probs.cuda()
        self.acts = self.acts.cuda()

    def insert(self, curr, obs, acts, action_log_probs,value_preds, rewards):
        self.obss[curr][self.step + 1].copy_(obs)
        self.acts[curr][self.step].copy_(acts)
        self.action_log_probs[curr][self.step].copy_(action_log_probs)
        self.value_preds[curr][self.step].copy_(value_preds)
        self.rews[curr][self.step].copy_(rewards)

        self.step = (self.step + 1) % self.batch_size

    def after_update(self):
        self.obss[0].copy_(self.obss[-1])

    def compute_returns(self,
                        curr,
                        next_value,
                        use_gae,
                        gamma,
                        gae_lambda,
                        use_proper_time_limits=True):
        if use_proper_time_limits:
            if use_gae:
                self.value_predsp[curr][-1] = next_value
                gae = 0
                for step in reversed(range(self.rews[curr].size(0))):
                    delta = self.rews[curr][step] + gamma * self.value_preds[curr][step + 1] * \
                            self.masks[curr][step + 1] - self.value_preds[curr][step]
                    gae = delta + gamma * gae_lambda * self.masks[curr][step + 1] * gae
                    gae = gae * self.bad_masks[curr][step + 1]
                    self.rets[step] = gae + self.value_preds[step]
            else:
                self.rets[curr][-1] = next_value
                for step in reversed(range(self.rews[curr].size(0))):
                    self.rets[curr][step] = (self.rets[curr][step + 1] * \
                        gamma * self.masks[curr][step + 1] + self.rews[curr][step]) * self.bad_masks[curr][step + 1] \
                        + (1 - self.bad_masks[curr][step + 1]) * self.value_preds[curr][step]
        else:
            if use_gae:
                self.value_preds[curr][-1] = next_value
                gae = 0
                for step in reversed(range(self.rews[curr].size(0))):
                    delta = self.rews[curr][step] + gamma * self.value_preds[curr][step + 1] \
                            * self.masks[curr][step + 1] - self.value_preds[curr][step]
                    gae = delta + gamma * gae_lambda * self.masks[curr][step + 1] * gae
                    self.rets[curr][step] = gae + self.value_preds[curr][step]
            else:
                self.rets[curr][-1] = next_value
                for step in reversed(range(self.rews[curr].size(0))):
                    self.rets[curr][step] = self.rets[curr][step + 1] * \
                        gamma * self.masks[curr][step + 1] + self.rews[curr][step]

    def feed_forward_generator(self,curr,advantages):
        num_agent, batch_size = self.rews[curr].size()[0:2]
        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            batch_size,
            drop_last=True)
        for indices in sampler:
            obs_batch = self.obss[curr][:-1].view(-1, *self.obss[curr].size()[2:])[indices]
            actions_batch = self.acts[curr].view(-1,self.acts[curr].size(-1))[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            return_batch = self.rets[curr][:-1].view(-1, 1)[indices]
            masks_batch = self.masks[curr][:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs[curr].view(-1,1)[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.view(-1, 1)[indices]

            yield obs_batch, actions_batch, value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ
