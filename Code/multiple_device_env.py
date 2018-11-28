# import the required modules
import random
import single_device_env
import numpy as np


# Class Environment


class MultipleDeviceEnvironment:
    def __init__(self, num_devices, devices=None):
        if devices is not None:
            self.devices = devices
        else:
            self.devices = [single_device_env.get_random_env() for _ in range(num_devices)]
        self.history_actions = []
        self.done = False
        self.time_stamp = 0
        for d in self.devices:
            print(f'schedule: {d.schedule_start} - {d.schedule_stop}, duration: {d.usage_duration}')

    def get_action_shape(self):
        return len(self.devices)

    def reset(self):
        self.done = False
        self.time_stamp = 0
        for d in self.devices:
            d.reset()
        return self.get_obs()

    def action_space_sample(self):
        return [random.randint(0, 1) for _ in self.devices]

    def get_obs_shape(self):
        return len(self.get_obs())

    def get_obs(self):
        obs = [self.time_stamp]
        for d in self.devices:
            ob = d.get_obs()
            obs.extend(ob[1:])
        return np.array(obs)

    def reward(self, action):
        r = 0
        for d, a in zip(self.devices, action):
            r += d.reward(a)
        return r

    def step(self, action):
        if len(action) != len(self.devices):
            raise AssertionError("Need %s actions, but %s actions are provided" % (len(self.devices), len(action)))

        self.history_actions.append(action)

        obs = [self.time_stamp]
        reward = 0

        for d, a in zip(self.devices, action):
            ob, r, done, _ = d.step([a])
            obs.extend(ob[1:])
            reward += r
        self.time_stamp += 1
        self.done = self.time_stamp == 24

        return obs, reward, self.done, None

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

    return MultipleDeviceEnvironment(appliances_number, appliances_consumption, electricity_cost, udc, schedule_start, schedule_stop,
                       usage_duration, penalty, encourage)


if __name__ == '__main__':
    # print(Environment.time_stamp)
    env = MultipleDeviceEnvironment(3)

    # for i in range(0, int(1e5)):
    #     ob, r, done = env.step(1)
    # if done:
    #     env.reset()
    # if i % 1000 == 0:
    #     print(np.mean(env.episode_rewards[-100:]))
    while not env.done:
        a = env.action_space_sample()
        ob, r, done, _ = env.step(a)
        print(f'action: {a}, reward: {r}, obs: {ob}')
