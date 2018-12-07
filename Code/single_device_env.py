# import the required modules
import random

import numpy as np


# Class Environment


class Environment:
    def __init__(self, appliances_number, appliances_consumption, electricity_cost, udc, schedule_start, schedule_stop
                 , usage_duration, penalty, encourage):
        self.advise = False
        self.appliances_number = appliances_number
        self.appliances_consumption = appliances_consumption
        self.electricity_cost = electricity_cost
        self.udc = udc
        self.schedule_start = schedule_start
        self.schedule_stop = schedule_stop
        self.usage_duration = usage_duration
        self.penalty = penalty
        self.encourage = encourage
        self.preferences_satisfied = True
        self.done = False
        self.time_stamp = 0
        self.state_accumulation = 0
        self.episode_rewards = []
        self.history_actions = []
        print(f'schedule: {self.schedule_start} - {self.schedule_stop}, duration: {self.usage_duration}')

    def get_action_shape(self):
        return 1

    def reset(self):
        self.done = False
        self.time_stamp = 0
        self.state_accumulation = 0
        self.history_actions = []
        return self.get_obs()

    def action_space_sample(self):
        return [random.randint(0, 1)]

    def get_obs_shape(self):
        return len(self.get_obs())

    def get_obs(self):
        obs = [self.time_stamp, self.state_accumulation, self.schedule_start, self.usage_duration, self.schedule_stop]
        if self.advise:
            # Giving optimal advise currently
            if self.time_stamp >= 5 and self.time_stamp <= 8:
                a = 1
            else:
                a = 0
            obs.append(a)
        return obs

    def reward(self, action):
        in_schedule_condition = self.schedule_start <= self.time_stamp < self.schedule_stop
        at_schedule_stop = self.time_stamp == self.schedule_stop

        reward_function = (1 - in_schedule_condition) * \
                          (at_schedule_stop * self.encourage * (self.usage_duration - \
                                                                np.abs(self.usage_duration - self.state_accumulation)) + \
                           action * self.penalty +
                           (1 - action) * self.encourage) + \
                          in_schedule_condition * (
                                  action * (self.electricity_cost[self.time_stamp] * self.appliances_consumption) + \
                                  (1 - action) * self.udc * self.appliances_consumption)
        reward_function *= -1
        self.episode_rewards.append(reward_function)
        return reward_function

    def old_reward(self, action):
        in_schedule_condition = self.schedule_start <= self.time_stamp <= self.schedule_stop
        at_schedule_stop = self.time_stamp == self.schedule_stop

        reward_function = (1 - in_schedule_condition) * \
                          (action * self.penalty +
                           (1 - action) * self.encourage) + \
                          in_schedule_condition * (
                                  at_schedule_stop * ((not self.preferences_satisfied) *
                                                      self.penalty *
                                                      np.abs(self.usage_duration - self.state_accumulation) +
                                                      self.preferences_satisfied *
                                                      self.encourage * self.usage_duration) + \
                                  (1 - at_schedule_stop) *
                                  (action * self.electricity_cost[self.time_stamp] * self.appliances_consumption + \
                                   (1 - action) * self.udc * self.appliances_consumption))
        reward_function *= -1
        self.episode_rewards.append(reward_function)
        return reward_function

    def step(self, action):
        # Compatible to multiple device
        action = action[0]

        self.history_actions.append(action)
        self.state_accumulation += action
        if self.state_accumulation != self.usage_duration and self.time_stamp == self.schedule_stop:
            self.preferences_satisfied = False
        reward = self.reward(action)
        self.time_stamp += 1
        self.done = self.time_stamp == 24
        return self.get_obs(), reward, self.done, None


def get_random_env():
    appliances_number = 1
    udc = 0.
    electricity_cost = np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 12, 12, 5, 5, 5, 5, 10, 10, 10, 5, 5, 5])
    # appliances_consumption = np.random.randint(1, 10) / 10
    # schedule_start = np.random.randint(0, 12)
    # schedule_stop = np.random.randint(13, 23)
    # usage_duration = np.random.randint(0, schedule_stop - schedule_start)
    appliances_consumption = 1
    schedule_start = 5
    schedule_stop = 20
    usage_duration = 4
    penalty = 10.
    encourage = -10.

    return Environment(appliances_number, appliances_consumption, electricity_cost, udc, schedule_start, schedule_stop,
                       usage_duration, penalty, encourage)

def get_random_env_alter():
    appliances_number = 1
    udc = 0.
    electricity_cost = np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 12, 12, 5, 5, 5, 5, 10, 10, 10, 5, 5, 5])
    # appliances_consumption = np.random.randint(1, 10) / 10
    schedule_start = np.random.randint(0, 12)
    schedule_stop = np.random.randint(13, 23)
    usage_duration = np.random.randint(0, schedule_stop - schedule_start)
    appliances_consumption = 1
    penalty = 10.
    encourage = -10.

    return Environment(appliances_number, appliances_consumption, electricity_cost, udc, schedule_start, schedule_stop,
                       usage_duration, penalty, encourage)

if __name__ == '__main__':
    env = get_random_env()
    rewards = []
    count = 0
    while not env.done:
        a = np.random.randint(0, 2)
        a = 0
        if count >= 5 and count <= 8:
            a = 1
        count += 1
        ob, r, done, _ = env.step([a])
        rewards.append(r)
        print(f'action: {a}, reward: {r}, obs: {ob}')
    print("Sum reward of episode: %s" % (sum(rewards)))
    print("Mean reward of random actions: %s " % (sum(rewards) / len(rewards)))
