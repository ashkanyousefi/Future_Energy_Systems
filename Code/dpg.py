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

logs = []


class DPGAgent:

    def __init__(self, multiple=True, dnum=3):
        if multiple:
            self.env = MultipleDeviceEnvironment(dnum)
            self.dnum = dnum
        else:
            self.dnum = 1
            self.env = single_device_env.get_random_env()

        self.action_size = 1
        self.obs_size = 5
        self.max_episodes = 1500
        self.learning_rate = 0.01
        self.gamma = 0.95  # Discount rate
        self.input = tf.placeholder(tf.float32, [None, self.obs_size], name="input")
        self.actions = tf.placeholder(tf.int32, [None, 2**self.action_size], name="actions")
        self.discounted_episode_rewards_ = tf.placeholder(tf.float32, [None, ], name="discounted_episode_rewards")

        self.optimizer, self.loss, self.act_dist = self.build_graph()

    def discount_and_normalize_rewards(self, episode_rewards):
        discounted_episode_rewards = np.zeros_like(episode_rewards)
        cumulative = 0.0
        for i in reversed(range(len(episode_rewards))):
            cumulative = cumulative * self.gamma + episode_rewards[i]
            discounted_episode_rewards[i] = cumulative

        mean = np.mean(discounted_episode_rewards)
        std = np.std(discounted_episode_rewards)
        discounted_episode_rewards = (discounted_episode_rewards - mean) / (std)

        return discounted_episode_rewards

    def build_graph(self):
        with tf.name_scope("inputs"):

            with tf.name_scope("fc1"):
                fc1 = tf.contrib.layers.fully_connected(inputs=self.input,
                                                        num_outputs=128,
                                                        activation_fn=tf.nn.relu,
                                                        weights_initializer=tf.contrib.layers.xavier_initializer())

            with tf.name_scope("fc2"):
                fc2 = tf.contrib.layers.fully_connected(inputs=fc1,
                                                        num_outputs=64,
                                                        activation_fn=tf.nn.relu,
                                                        weights_initializer=tf.contrib.layers.xavier_initializer())

            with tf.name_scope("fc3"):
                fc3 = tf.contrib.layers.fully_connected(inputs=fc2,
                                                        num_outputs=2**self.action_size,
                                                        activation_fn=None,
                                                        weights_initializer=tf.contrib.layers.xavier_initializer())

            with tf.name_scope("softmax"):
                action_distribution = tf.nn.softmax(fc3)

            with tf.name_scope("loss"):
                # tf.nn.softmax_cross_entropy_with_logits computes the cross entropy of the result after applying the softmax function
                # If you have single-class labels, where an object can only belong to one class, you might now consider using
                # tf.nn.sparse_softmax_cross_entropy_with_logits so that you don't have to convert your labels to a dense one-hot array.
                neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=fc3, labels=self.actions)
                loss = tf.reduce_mean(neg_log_prob * self.discounted_episode_rewards_)

            with tf.name_scope("train"):
                train_opt = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        return train_opt, loss, action_distribution

    def train(self):
        mean_reward_ = tf.placeholder(tf.float32, name="mean_reward")
        # Setup TensorBoard Writer
        writer = tf.summary.FileWriter("/tensorboard/pg/1")

        ## Losses
        tf.summary.scalar("Loss", self.loss)

        ## Reward mean
        tf.summary.scalar("Reward_mean", mean_reward_)

        write_op = tf.summary.merge_all()

        allRewards = []
        total_rewards = 0
        maximumRewardRecorded = 0
        episode = 0
        episode_states, episode_actions, episode_rewards = [], [], []

        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for episode in range(self.max_episodes):

                log_e_reward = 0
                count = 0

                # Launch the game
                state = self.env.reset()
                self.log_env.reset()

                while True:

                    # Choose action a, remember WE'RE NOT IN A DETERMINISTIC ENVIRONMENT, WE'RE OUTPUT PROBABILITIES.
                    state = state[:5]
                    action_probability_distribution = sess.run(self.act_dist, feed_dict={self.input: np.reshape(state, (1, len(state)))})

                    actions = []
                    for _ in range(self.dnum):
                        action = np.random.choice(range(action_probability_distribution.shape[1]),
                                              p=action_probability_distribution.ravel())
                        actions.append(action)

                    # Perform a
                    new_state, reward, done, info = self.env.step(actions)
                    # Store s, a, r
                    episode_states.append(state)

                    # For actions because we output only one (the index) we need 2 (1 is for the action taken)
                    # We need [0., 1.] (if we take right) not just the index
                    action_ = np.zeros(2**self.action_size)
                    action_[action] = 1

                    episode_actions.append(action_)

                    episode_rewards.append(reward)
                    if done:
                        # Calculate sum reward
                        episode_rewards_sum = np.sum(episode_rewards)
                        logs.append(log_e_reward)

                        allRewards.append(episode_rewards_sum)

                        total_rewards = np.sum(allRewards)

                        # Mean reward
                        mean_reward = np.divide(episode_rewards_sum, 24)

                        maximumRewardRecorded = np.amax(allRewards)

                        print("==========================================")
                        print("Episode: ", episode)
                        print("Reward: ", episode_rewards_sum)
                        print("Mean Reward", mean_reward)
                        print("Max reward so far: ", maximumRewardRecorded)
                        # Calculate discounted reward
                        discounted_episode_rewards = self.discount_and_normalize_rewards(episode_rewards)

                        # Feedforward, gradient and backpropagation
                        loss_, _ = sess.run([self.loss, self.optimizer], feed_dict={self.input: np.vstack(np.array(episode_states)),
                                                                          self.actions: np.vstack(np.array(episode_actions)),
                                                                          self.discounted_episode_rewards_: discounted_episode_rewards
                                                                          })

                        # Write TF Summaries
                        summary = sess.run(write_op, feed_dict={self.input: np.vstack(np.array(episode_states)),
                                                                self.actions: np.vstack(np.array(episode_actions)),
                                                                self.discounted_episode_rewards_: discounted_episode_rewards,
                                                                mean_reward_: mean_reward
                                                                })

                        writer.add_summary(summary, episode)
                        writer.flush()

                        # Reset the transition stores
                        episode_states, episode_actions, episode_rewards = [], [], []

                        break

                    state = new_state

                # Save Model
                if episode % 100 == 0:
                    saver.save(sess, "./models/model.ckpt")
                    print("Model saved")


def main():
    agent = DPGAgent()
    agent.train()
    print(logs)
    with open("dpg_log.txt", "w+") as f:
        for l in logs:
            f.write("%s\n" % (l))

if __name__ == '__main__':
    main()
