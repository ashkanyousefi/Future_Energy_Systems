# import the required modules
import random

import numpy as np


# Class Environment


class Environment:
    def __init__(self, appliances_number, appliances_consumption, electricity_cost, udc, schedule_start, schedule_stop
                 , usage_duration, penalty, encourage):
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
        return 2

    def reset(self):
        self.done = False
        self.time_stamp = 0
        self.state_accumulation = 0
        self.history_actions = []
        return self.get_obs()

    def action_space_sample(self):
        return random.randint(0, 1)

    def get_obs_shape(self):
        return np.shape(self.get_obs())

    def get_obs(self):
        return [self.time_stamp, self.state_accumulation]

    def reward(self, action):
        in_schedule_condition = self.schedule_start <= self.time_stamp <= self.schedule_stop
        at_schedule_stop = self.time_stamp == self.schedule_stop

        reward_function = (1 - in_schedule_condition) * \
                          (action * self.penalty +
                           (1 - action) * self.encourage) + \
                          in_schedule_condition * (
                                  at_schedule_stop * ((not self.preferences_satisfied) *
                                                      self.penalty * np.abs(self.usage_duration - self.state_accumulation) +
                                                      self.preferences_satisfied * self.encourage * self.usage_duration) + \
                                  (1 - at_schedule_stop) *
                                  (action * self.electricity_cost[self.time_stamp] * self.appliances_consumption + \
                                   (1 - action) * self.udc * self.appliances_consumption))
        reward_function *= -1
        self.episode_rewards.append(reward_function)
        return reward_function

    def step(self, action):
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


if __name__ == '__main__':
    # load_details = np.array(
    #     [['load number', 'Appliances', 's', 'f', 'l', 'r', 'udc'], [1, 'Lighting', 18, 20, 3, 0.36, 8],
    #      [2, 'Washing machine', 8, 14, 2, 0.5, 1], [3, 'Cloth dryer', 11, 17, 1, 1.8, 0]
    #         , [4, 'Dish washer', 14, 19, 2, 1.2, 9], ['Vacuum cleaner', 9, 14, 1, 0.65, 1],
    #      [6, 'Iron', 6, 9, 1, 1.1, 1], [7, 'Rice cooker', 7, 10, 1, 0.30, 0]])
    #
    # electricity_price = np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 12, 12, 5, 5, 5, 5, 10, 10, 10, 5, 5, 5])
    # print(Environment.time_stamp)
    env = get_random_env()

    # for i in range(0, int(1e5)):
    #     ob, r, done = env.step(1)
    # if done:
    #     env.reset()
    # if i % 1000 == 0:
    #     print(np.mean(env.episode_rewards[-100:]))
    while not env.done:
        a = np.random.randint(0, 2)
        ob, r, done, _ = env.step(a)
        print(f'action: {a}, reward: {r}, obs: {ob}')
