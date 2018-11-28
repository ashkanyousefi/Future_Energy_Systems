from random import random

import argparse
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
import matplotlib.pyplot as plt
from gym import wrappers

import single_device_env
from multiple_device_env import MultipleDeviceEnvironment

mean_return_log = []

class GymDQNLearner:
    def __init__(self, multiple, num_devices):
        if multiple:
            self.saving_path = './saved_models/dqn/multiple/%s/' % (num_devices)
        else:
            self.saving_path = './saved_models/dqn/single/'

        self.num_devices = num_devices
        self.formatstr = "{0:0" + str(self.num_devices) + "b}"

        self.epochs = 1500
        self.gamma = .9
        self.epsilon = 1.
        self.train_per_epoch = 1
        self.n_generating_trajectories_per_epoch = 1
        self.max_memory_size = 2000
        self.max_trajectory_length = 1000
        self.batch_size = 256

        # self.env = gym.make('CartPole-v0').env
        # self.state_embedding_size = self.env.observation_space.shape[0]
        # self.number_of_actions = self.env.action_space.n
        if not multiple:
            self.env = single_device_env.get_random_env()
        else:
            self.env = MultipleDeviceEnvironment(num_devices)
        self.state_embedding_size = self.env.get_obs_shape()
        self.number_of_actions = self.env.get_action_shape()
        print(self.state_embedding_size, self.number_of_actions)
        self.layer_units = [32, 16, int(2**self.number_of_actions)]
        # self.layer_units = [64, 32, self.number_of_actions]
        self.layer_activations = ['tanh', 'relu', None]
        # self.layer_keep_probs = [.1, .1, 1.]
        # self.layer_regularizers = [tf.contrib.layers.l2_regularizer(1.),
        #                            tf.contrib.layers.l2_regularizer(1.),
        #                            tf.contrib.layers.l2_regularizer(1.)]
        self.layer_keep_probs = [1., 1., 1.]
        self.layer_regularizers = [None,
                                   None,
                                   None]
        self.initialize_experience_replay_memory()

        self.create_model()
        self.load()

    def initialize_experience_replay_memory(self):
        self.experience_replay_memory = np.array([])

    def get_epsilon(self, i):
        # alpha = 1e-5
        # return 1.0 - (i / np.sqrt(1 + alpha * (i ** 2))) * np.sqrt(alpha)
        # return 1.0 - float(i) / epochs
        return max(0.1, self.epsilon * (0.9989 ** i))
        # return 1

    def get_state_weights(self, trajectory):
        # total_reward = len(trajectory)
        total_reward = np.sum([t[2] for t in trajectory])
        # cum_reward = np.cumsum([t[2] for t in trajectory])
        return [total_reward for i, t in enumerate(trajectory)]

    def add_to_memory(self, trajectory):
        weights = self.get_state_weights(trajectory)
        for (from_state, action, reward, to_state, done, q_value), weight in zip(trajectory, weights):
            if self.experience_replay_memory.shape[0] >= self.max_memory_size:
                # self.experience_replay_memory = \
                #     np.delete(self.experience_replay_memory, np.random.randint(0, self.experience_replay_memory.shape[0]))
                # self.experience_replay_memory = self.experience_replay_memory[1:]
                min_element = np.argmin([exp['weight'] for exp in self.experience_replay_memory])
                self.experience_replay_memory = \
                    np.delete(self.experience_replay_memory, min_element)
            self.experience_replay_memory = np.append(self.experience_replay_memory, [
                {'from': from_state, 'action': action,
                 'reward': reward, 'done': done,
                 'to': to_state,
                 'q_value': q_value,
                 'weight': weight}])

    def softmax(self, logits):
        exps = np.exp(logits)
        return exps / np.sum(exps)

    def sample_from_memory(self):
        if self.experience_replay_memory.shape[0] > 1:
            weights = np.array([exp['weight'] for exp in self.experience_replay_memory])
            # p = weights / np.sum(weights)
            p = self.softmax(weights)
            return np.random.choice(self.experience_replay_memory,
                                    np.min([self.batch_size, self.experience_replay_memory.shape[0]]), p=p)
        else:
            return self.experience_replay_memory

    def create_multilayer_dense(self, scope, layer_input, layer_units, layer_activations, keep_probs=None,
                                regularizers=None, reuse_vars=None):
        with tf.variable_scope(scope, reuse=reuse_vars):
            last_layer = None
            if regularizers is None:
                regularizers = [None for _ in layer_units]
            if keep_probs is None:
                keep_probs = [1. for _ in layer_units]
            for i, (layer_size, activation, keep_prob, reg) in enumerate(zip(layer_units, layer_activations,
                                                                             keep_probs, regularizers)):
                if i == 0:
                    inp = layer_input
                else:
                    inp = last_layer
                last_layer = tf.layers.dense(inp, layer_size, activation, activity_regularizer=reg)
                if keep_prob != 1.0:
                    last_layer = tf.nn.dropout(last_layer, keep_prob)
        return last_layer

    def create_model(self):
        self.inputs = tf.placeholder(np.float32, [None, self.state_embedding_size], name='inputs')
        self.outputs = tf.placeholder(np.float32, [None, int(2**self.number_of_actions)], name='outputs')

        self.output_layer = \
            self.create_multilayer_dense('q_func', self.inputs, self.layer_units, self.layer_activations,
                                         self.layer_keep_probs, self.layer_regularizers)
        self.test_output_layer = self.create_multilayer_dense('q_func', self.inputs, self.layer_units,
                                                              self.layer_activations, reuse_vars=True)
        self.loss = tf.losses.mean_squared_error(self.outputs, self.output_layer, scope='q_func')

        trainable_variables = tf.trainable_variables('q_func')
        self.train_op = tf.train.AdamOptimizer(1e-3, name='optimizer').minimize(self.loss, var_list=trainable_variables)
        self.saver = tf.train.Saver()
        self.sess = tf.Session()

    def get_action(self, epoch, q_value):
        if random() < self.get_epsilon(epoch):
            # action = self.env.action_space.sample()
            action = self.env.action_space_sample()
        else:
            action = np.argmax(q_value)
            action = [int(a) for a in self.formatstr.format(action)]
        return action

    def generate_new_trajectories(self, epoch):
        for _ in range(self.n_generating_trajectories_per_epoch):
            observation = self.env.reset()
            done = False
            trajectory = []
            while not done:
                q_value = self.sess.run(self.test_output_layer, {self.inputs: [observation]})[0]
                action = self.get_action(epoch, q_value)
                new_observation, reward, done, info = self.env.step(action)
                trajectory.append((observation, action, reward, new_observation, done, q_value))
                observation = new_observation
                if len(trajectory) > self.max_trajectory_length:
                    break
            self.add_to_memory(trajectory)

    def create_batch(self):
        batch_q_values = []
        batch_observations = []
        for experience in self.sample_from_memory():
            action = experience['action']
            new_q_value = np.copy(experience['q_value'])
            new_q_value[action] = experience['reward']
            if not experience['done']:
                update_value = np.max(self.sess.run(self.output_layer, {self.inputs: [experience['to']]})[0])
                new_q_value[action] += self.gamma * update_value
            batch_q_values.append(new_q_value)
            batch_observations.append(experience['from'])
        return batch_observations, batch_q_values

    def train(self):
        epoch = 0
        # while loss_value > 0.002:
        while epoch < self.epochs:
            self.generate_new_trajectories(epoch)
            epoch_loss = None
            for sub_epoch_id in range(self.train_per_epoch):
                batch_observations, batch_q_values = self.create_batch()
                _, epoch_loss = self.sess.run((self.train_op, self.loss),
                                              {self.inputs: batch_observations, self.outputs: batch_q_values})
            self.save()
            epoch_total_reward = self.play()
            mean_reward = np.mean([s['weight'] for s in self.experience_replay_memory])
            max_reward = np.max([s['weight'] for s in self.experience_replay_memory])
            print(
                "*********** epoch %d ***********\n"
                "memory size: %d, mean-max state weights: %.3f\t%.3f\n"
                "total loss: %f\n"
                "total reward gained: %f\n"
                "epsilon: %.3f" % (epoch, self.experience_replay_memory.shape[0],
                                   mean_reward,
                                   max_reward,
                                   epoch_loss, epoch_total_reward, self.get_epsilon(epoch)))
            mean_return_log.append(mean_reward)
            epoch += 1

    def play(self, render=False, monitor=False, max_timestep=None):
        total_reward = 0
        done = False
        observation = self.env.reset()
        reward = None
        timestep = 0
        if monitor:
            env = wrappers.Monitor(self.env, "./monitors/dqn/", force=True)
        else:
            env = self.env
        while not done:
            if render:
                env.render()
            q_value = self.sess.run(self.test_output_layer, {self.inputs: [observation]})[0]
            # action = env.action_space.sample() # random action
            action = np.argmax(q_value)
            # mapping integer to actual actions
            action = [int(a) for a in self.formatstr.format(action)]
            if timestep == self.max_trajectory_length:
                print(total_reward)
                break
            # mod = total_reward % 100
            # if mod in (0, 1, 2, 3):
            # action = env.action_space.sample()
            # action = 1 - action
            new_observation, reward, done, info = env.step(action)
            #!!! tr is out of scope due to training is in main function now
            #if not tr: 
            #   print(f'action selected: {action}, obs: {observation}, reward: {reward}')
            total_reward += reward
            timestep += 1
            observation = new_observation
            if done:
                break
            if max_timestep is not None:
                if timestep > max_timestep:
                    if monitor:
                        env.close()
                        env.reset()
                    break
        return total_reward

    def save(self):
        self.saver.save(self.sess, self.saving_path)

    def load(self):
        import os
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        if not os.path.exists(self.saving_path):
            os.makedirs(self.saving_path)
        if not tf.train.checkpoint_exists(self.saving_path + 'checkpoint'):
            print('Saved temp_models not found! Randomly initialized.')
        else:
            self.saver.restore(self.sess, self.saving_path)
            print('Model loaded!')


def main(multiple, dnum, fig_path="exp.png"):
    tr = True
    if not multiple:
        dnum = 1
    model = GymDQNLearner(multiple, dnum)
    if tr:
        model.train()
    episode_reward = model.play(False, False, 2000)
    print('total reward: %f' % episode_reward)
    if multiple:
        fig_path = "mul_%s_" % (dnum) + fig_path
    with open(fig_path[:-4]+".txt", "w+") as f:
        for l in mean_return_log:
            f.write(str(l) + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--multiple', action='store_true')
    parser.add_argument('--dnum', default=3)
    parser.add_argument('--path', default="exp.png")
    args = parser.parse_args()
    main(multiple=args.multiple, dnum=args.dnum, fig_path=args.path)

