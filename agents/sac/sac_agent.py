from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os

import gin.tf
import numpy as np
import tensorflow as tf
import replay_memory
import tensorflow_probability as tfp

tfd = tfp.distributions

slim = tf.contrib.slim

Transition = collections.namedtuple(
    'Transition', ['reward', 'observation', 'legal_actions', 'action', 'begin'])

def policy_template(state, num_actions, layer_size=512, num_layers=2):
    weights_initializer = slim.variance_scaling_initializer(
        factor=1.0 / np.sqrt(3.0), mode='FAN_IN', uniform=True)

    net = tf.cast(state, tf.float32)
    net = tf.squeeze(net, axis=2)
    for _ in range(num_layers):
        net = slim.fully_connected(net, layer_size,
                                   activation_fn=tf.nn.relu)
    mean = slim.fully_connected(net, num_actions, activation_fn=None,
                               weights_initializer=weights_initializer)
    log_std = slim.fully_connected(net, num_actions, activation_fn=None,
                               weights_initializer=weights_initializer)
    log_std = tf.clip_by_value(log_std, clip_value_min=0,clip_value_max=2)
    return mean,log_std

def critic_template(state, num_actions, layer_size=512, num_layers=2):
    weights_initializer = slim.variance_scaling_initializer(
        factor=1.0 / np.sqrt(3.0), mode='FAN_IN', uniform=True)

    net = tf.cast(state, tf.float32)
    net = tf.squeeze(net, axis=2)
    for _ in range(num_layers):
        net = slim.fully_connected(net, layer_size,
                                   activation_fn=tf.nn.relu)
    net = slim.fully_connected(net, num_actions, activation_fn=None,
                               weights_initializer=weights_initializer)
    return net

def value_template(state,layer_size=512,num_layers=2):
    weights_initializer = slim.variance_scaling_initializer(
        factor=1.0 / np.sqrt(3.0), mode='FAN_IN', uniform=True)

    net = tf.cast(state, tf.float32)
    net = tf.squeeze(net, axis=2)
    for _ in range(num_layers):
        net = slim.fully_connected(net, layer_size,
                                   activation_fn=tf.nn.relu)
    net = slim.fully_connected(net, 1, activation_fn=None,
                               weights_initializer=weights_initializer)
    return net
@gin.configurable
class SACAgent(object):
    def __init__(self,
                 num_actions=None,
                 observation_size=None,
                 num_players=None,
                 gamma=0.99,
                 update_horizon=1,
                 min_replay_history=500,
                 update_period=4,
                 stack_size=1,
                 target_update_period=500,
                 epsilon_train=0.02,
                 epsilon_eval=0.001,
                 epsilon_decay_period=1000,
                 graph_template=[policy_template,critic_template,value_template],
                 tf_device='/gpu:*',
                 use_staging=True):

        tf.logging.info('Creating %s agent with the following parameters:',
                        self.__class__.__name__)
        tf.logging.info('\t gamma: %f', gamma)
        tf.logging.info('\t update_horizon: %f', update_horizon)
        tf.logging.info('\t min_replay_history: %d', min_replay_history)
        tf.logging.info('\t update_period: %d', update_period)
        tf.logging.info('\t target_update_period: %d', target_update_period)
        tf.logging.info('\t epsilon_train: %f', epsilon_train)
        tf.logging.info('\t epsilon_eval: %f', epsilon_eval)
        tf.logging.info('\t epsilon_decay_period: %d', epsilon_decay_period)
        tf.logging.info('\t tf_device: %s', tf_device)
        tf.logging.info('\t use_staging: %s', use_staging)
        # Global variables.
        self.num_actions = num_actions
        self.observation_size = observation_size
        self.num_players = num_players
        self.gamma = gamma
        self.update_horizon = update_horizon
        self.cumulative_gamma = math.pow(gamma, update_horizon)
        self.min_replay_history = min_replay_history
        self.target_update_period = target_update_period
        self.epsilon_train = epsilon_train
        self.epsilon_eval = epsilon_eval
        self.epsilon_decay_period = epsilon_decay_period
        self.update_period = update_period
        self.eval_mode = False
        self.training_steps = 0
        self.tau = 0.01
        self.batch_staged = False
        self.optimizer = [tf.train.AdamOptimizer(learning_rate=.0025,epsilon=1e-6) for _ in range(4)]
        self.alpha = None
        with tf.device(tf_device):
            online_critic1 = tf.make_template('Online_Critic1', graph_template[1])
            online_critic2 = tf.make_template('Online_Critic2', graph_template[1])
            target_value = tf.make_template('Target_Value', graph_template[2])
            online_value = tf.make_template('Online_Value', graph_template[2])
            online_actor = tf.make_template('Online_Actor', graph_template[0])
            states_shape = (1, observation_size, stack_size)
            actions_shape = (1, num_actions, stack_size)
            self.state = np.zeros(states_shape)
            self.state_ph = tf.placeholder(tf.uint8, states_shape, name='state_ph')
            self.legal_actions_ph = tf.placeholder(tf.float32,
                                                   [self.num_actions],
                                                   name='legal_actions_ph')
            self.action_ph = tf.placeholder(tf.uint8, actions_shape, name='state_ph')
            self._replay = self._build_replay_memory(use_staging)
            # for get action
            self.mu, self.log_std = online_actor(self.state_ph,self.num_actions)
            self._replay_pi,self._replay_logstd = online_actor(self._replay.states,self.num_actions)
            # for train
            self._replay = self._build_replay_memory(use_staging)
            print(self.num_actions)
            self._replay_q1s = online_critic1(self._replay.states,self.num_actions)
            self._replay_q2s = online_critic2(self._replay.states,self.num_actions)
            self._vt = target_value(self._replay.next_states)
            self._v = online_value(self.state_ph)
            self._pi_q1 = online_critic1(self.state_ph,self.num_actions)
            self._pi_q2 = online_critic2(self.state_ph,self.num_actions)
            self._train_op = self._build_train_op()
            self._sync_qt_ops = self._build_sync_op()

        # Set up a session and initialize variables.
        self._sess = tf.Session(
            '', config=tf.ConfigProto(allow_soft_placement=True))
        self._init_op = tf.global_variables_initializer()
        self._sess.run(self._init_op)
        self._saver = tf.train.Saver(max_to_keep=3)
        # This keeps tracks of the observed transitions during play, for each
        # player.
        self.transitions = [[] for _ in range(num_players)]

    def _build_replay_memory(self, use_staging):
        return replay_memory.WrappedReplayMemory(
            num_actions=self.num_actions,
            observation_size=self.observation_size,
            batch_size=32,
            stack_size=1,
            use_staging=use_staging,
            update_horizon=self.update_horizon,
            gamma=self.gamma)

    def _build_train_op(self):
        # train critic
        training_ops = []
        replay_action_one_hot = tf.one_hot(
            self._replay.actions, self.num_actions, 1., 0., name='action_one_hot')
        replay_chosen_q1 = tf.reduce_sum(
            self._replay_q1s * replay_action_one_hot,
            reduction_indices=1,
            name='replay_chosen_q1')
        replay_chosen_q2 = tf.reduce_sum(
            self._replay_q2s * replay_action_one_hot,
            reduction_indices=1,
            name='replay_chosen_q2')
        target = tf.stop_gradient(self._build_target_op())
        critic_loss1 = 0.5 * tf.reduce_mean((target - replay_chosen_q1) ** 2)
        critic_loss2 = 0.5 * tf.reduce_mean((target - replay_chosen_q2) ** 2)
        training_ops.append(self.optimizer[0].minimize(loss=critic_loss1))
        training_ops.append(self.optimizer[1].minimize(loss=critic_loss2))
        # train actor
        ## check one more time
        D_s = self._replay_pi.shape.as_list()[-1]
        policy_prior = tf.contrib.distributions.MultivariateNormalDiag(
            loc=tf.zeros(D_s), scale_diag=tf.ones(D_s))
        policy_prior_log_probs = policy_prior.log_prob(self._replay_pi)
        min_log_target = tf.minimum(self._pi_q1, self._pi_q2)
        policy_kl_loss = tf.reduce_mean(self._replay_logstd * tf.stop_gradient(
            self._replay_logstd - self._pi_q1 + self._v - policy_prior_log_probs))
        policy_regularization_losses = tf.get_collection(
            tf.GraphKeys.REGULARIZATION_LOSSES,
            scope='Policy')
        policy_regularization_loss = tf.reduce_sum(
            policy_regularization_losses)
        policy_loss = (policy_kl_loss
                       + policy_regularization_loss)
        # train value
        value_loss = 0.5 * tf.reduce_mean((self._v - tf.stop_gradient(min_log_target - self._replay_logstd + policy_prior_log_probs)) ** 2)
        training_ops.append(self.optimizer[2].minimize(loss=policy_loss))
        training_ops.append(self.optimizer[3].minimize(loss=value_loss))
        return training_ops

    def _build_target_op(self):
        # make target
        replay_next_vt_max = tf.reduce_max(self._vt + self._replay.next_legal_actions, 1)
        return self._replay.rewards + self.cumulative_gamma * replay_next_vt_max * (
                1. - tf.cast(self._replay.terminals, tf.float32))

    def _build_sync_op(self):
        # Get trainable variables from online and target networks.
        sync_qt_ops = []
        trainables_online = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='Online_Value')
        trainables_target = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='Target_Value')
        for (w_online, w_target) in zip(trainables_online, trainables_target):
            # Assign weights from online to target network.
            sync_qt_ops.append(tf.assign(w_target, (1 - self.tau) * w_target + self.tau * w_online))
        return sync_qt_ops

    def begin_episode(self,current_player, legal_actions, observation):
        self._train_step()
        self.action = self._select_action(observation, legal_actions)
        self._record_transition(current_player, 0, observation, legal_actions,
                                self.action, begin=True)
        return self.action

    def step(self, reward, current_player, legal_actions, observation):
        self._train_step()

        self.action = self._select_action(observation, legal_actions)
        self._record_transition(current_player, reward, observation, legal_actions,
                                self.action)
        return self.action

    def end_episode(self, final_rewards):
        self._post_transitions(terminal_rewards=final_rewards)

    def _record_transition(self, current_player, reward, observation,
                           legal_actions, action, begin=False):

        self.transitions[current_player].append(
            Transition(reward, np.array(observation, dtype=np.uint8, copy=True),
                       np.array(legal_actions, dtype=np.float32, copy=True),
                       action, begin))

    def _post_transitions(self, terminal_rewards):
        # We store each player's episode consecutively in the replay memory.
        for player in range(self.num_players):
            num_transitions = len(self.transitions[player])

            for index, transition in enumerate(self.transitions[player]):
                # Add: o_t, l_t, a_t, r_{t+1}, term_{t+1}
                final_transition = index == num_transitions - 1
                if final_transition:
                    reward = terminal_rewards[player]
                else:
                    reward = self.transitions[player][index + 1].reward

                self._store_transition(transition.observation, transition.action,
                                       reward, final_transition,
                                       transition.legal_actions)

            # Now that this episode has been stored, drop it from the transitions
            # buffer.
            self.transitions[player] = []

    def _select_action(self, observation, legal_actions,with_log=False):
        self.state[0, :, 0] = observation
        std = tf.math.exp(self.log_std)
        print(legal_actions)
        legal_actions[legal_actions<0] = 0
        mu = self.mu * self.legal_actions_ph
        dist = tfd.MultivariateNormalDiag(loc=mu, scale_diag=std)
        x_t = dist.sample()
        log_pi_t = dist.log_prob(x_t)
        action = tf.tanh(x_t)
        if self.eval_mode:
            action = self._sess.run(action, {self.state_ph: self.state, self.legal_actions_ph: legal_actions})
        else:
            action = self._sess.run(action, {self.state_ph: self.state, self.legal_actions_ph: legal_actions})
        if with_log:
            log_pi_t -= tf.reduce_sum(tf.log(1 - tf.tanh(x_t) ** 2 + 1e-6), axis=1)
            return action, log_pi_t
        else:
            return action

    def _train_step(self):
        if self.eval_mode:
          return

        # Run a training op.
        if (self._replay.memory.add_count >= self.min_replay_history and
            not self.batch_staged):
            self._sess.run(self._replay.prefetch_batch)
            self.batch_staged = True
        if (self._replay.memory.add_count > self.min_replay_history and
            self.training_steps % self.update_period == 0):
            # train is list, so fix it
            self._sess.run([self._train_op, self._replay.prefetch_batch])
        # Sync weights.
        if self.training_steps % self.target_update_period == 0:
            self._sess.run(self._sync_qt_ops)
        self.training_steps += 1

    def _store_transition(self, observation, action, reward, is_terminal,
                          legal_actions):
        if not self.eval_mode:
            self._sess.run(
                self._replay.add_transition_op, {
                    self._replay.add_obs_ph: observation,
                    self._replay.add_action_ph: action,
                    self._replay.add_reward_ph: reward,
                    self._replay.add_terminal_ph: is_terminal,
                    self._replay.add_legal_actions_ph: legal_actions
                })

    def bundle_and_checkpoint(self, checkpoint_dir, iteration_number):
        if not tf.gfile.Exists(checkpoint_dir):
            return None
        self._saver.save(
            self._sess,
            os.path.join(checkpoint_dir, 'tf_ckpt'),
            global_step=iteration_number)
        self._replay.save(checkpoint_dir, iteration_number)
        bundle_dictionary = {}
        bundle_dictionary['state'] = self.state
        bundle_dictionary['eval_mode'] = self.eval_mode
        bundle_dictionary['training_steps'] = self.training_steps
        bundle_dictionary['batch_staged'] = self.batch_staged
        return bundle_dictionary

    def unbundle(self, checkpoint_dir, iteration_number, bundle_dictionary):
        try:
            # replay.load() will throw a GOSError if it does not find all the
            # necessary files, in which case we should abort the process.
            self._replay.load(checkpoint_dir, iteration_number)
        except tf.errors.NotFoundError:
            return False
        for key in self.__dict__:
            if key in bundle_dictionary:
                self.__dict__[key] = bundle_dictionary[key]
        self._saver.restore(self._sess, tf.train.latest_checkpoint(checkpoint_dir))
        return True