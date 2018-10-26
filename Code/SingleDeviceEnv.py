# import the required modules
import numpy as np
import random

# Class Environment

class Action:
    def __init__(self, action):
        self.action = action


class Environment:
    def __init__(self, appliances_number, appliances_consumption, electricity_cost, udc, schedule_start, schedule_stop
                 , usage_duration, penalty):
        self.appliances_number = appliances_number
        self.appliances_consumption = appliances_consumption
        self.electricity_cost = electricity_cost
        self.udc = udc
        self.schedule_start = schedule_start
        self.schedule_stop = schedule_stop
        self.usage_duration = usage_duration
        self.penalty = penalty
        self.done = False
        self.time_stamp = 0
        self.state_accumulation = 0
        self.episode_rewards = []

    def get_action_shape(self):
        return 1

    def reset(self):
        self.appliances_number = 1
        self.udc = 1
        self.appliances_consumption = np.random.randint(1, 10) / 10
        self.schedule_start = np.random.randint(0, 12)
        self.schedule_stop = np.random.randint(12, 23)
        self.usage_duration = np.random.randint(1, 12)
        self.done = False
        self.time_stamp = 0
        self.state_accumulation = 0
        self.episode_rewards = []

    def action_space_sample(self):
        return random.randint(0, 1)

    def get_obs_shape(self):
        return len(self.get_obs())

    def get_obs(self):
        return [self.time_stamp, self.schedule_start, self.schedule_stop, self.state_accumulation, self.usage_duration]

    def reward(self, action):
        reward_function = (1-self.done) * self.penalty + (action*self.electricity_cost[self.time_stamp]*self.appliances_consumption)
        if self.time_stamp <= self.schedule_stop and self.time_stamp >= self.schedule_start:
            if action == 1:
                reward_function += 20
            else:
                reward_function -= self.udc
                
        self.episode_rewards.append(reward_function)
        return reward_function

    def step(self, action):

        #self.sta.state_accumulation += action
        if action == 1 and self.time_stamp <= self.schedule_stop and self.time_stamp >= self.schedule_start:
            self.state_accumulation += 1

        self.time_stamp += 1
        self.done = self.time_stamp == 23 or self.state_accumulation >= self.usage_duration
        return self.get_obs(), self.reward(action), self.done

def get_random_env():

    appliances_number = 1
    udc = 1
    appliances_consumption = np.random.randint(1, 10) / 10
    electricity_cost = np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 12, 12, 5, 5, 5, 5, 10, 10, 10, 5, 5, 5]) / 10
    schedule_start = np.random.randint(0, 12)
    schedule_stop = np.random.randint(13, 23)
    usage_duration = np.random.randint(0, schedule_stop - schedule_start)

    return Environment(appliances_number, appliances_consumption, electricity_cost, udc, schedule_start, schedule_stop, usage_duration, -.2)


if __name__ == '__main__':
    # load_details = np.array(
    #     [['load number', 'Appliances', 's', 'f', 'l', 'r', 'udc'], [1, 'Lighting', 18, 20, 3, 0.36, 8],
    #      [2, 'Washing machine', 8, 14, 2, 0.5, 1], [3, 'Cloth dryer', 11, 17, 1, 1.8, 0]
    #         , [4, 'Dish washer', 14, 19, 2, 1.2, 9], ['Vacuum cleaner', 9, 14, 1, 0.65, 1],
    #      [6, 'Iron', 6, 9, 1, 1.1, 1], [7, 'Rice cooker', 7, 10, 1, 0.30, 0]])
    #
    # electricity_price = np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 12, 12, 5, 5, 5, 5, 10, 10, 10, 5, 5, 5])
    # print(Environment.time_stamp)
    appliances_number = 1
    udc = np.array([1, 1, 1])
    appliances_consumption = np.array([3, 4, 5])
    electricity_cost = np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 12, 12, 5, 5, 5, 5, 10, 10, 10, 5, 5, 5]) / 10
    schedule_start = [10, 11, 10]
    schedule_stop = [19, 17, 21]
    usage_duration = [2, 3, 1]
    action = 1

    for start, stop, duration, consumption in zip(schedule_start, schedule_stop, usage_duration, appliances_consumption):
        env = Environment(appliances_number, consumption, electricity_cost, udc, start, stop, duration, -.2)
    done = False

    while not done:
        state, reward, done = env.step(action)
        print("time stamp %s, current %s, needed %s, reward %s, done %s" % (state[0], state[3], usage_duration, reward, done))


    # The setting for the Environment class presented in the following

    # The setting for the state class is set in the following