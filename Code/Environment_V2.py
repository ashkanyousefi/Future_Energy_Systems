# import the required modules
import numpy as np

# Class Environment

class State:
    def __init__(self, time_stamp, current_state, state_accumulation):
        self.time_stamp = time_stamp
        self.current_state = current_state
        self.state_accumulation = state_accumulation


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

        # Generate an instance from the State
        time_stamp = 0
        current_state = np.array([0, 0, 0])
        state_accumulation = np.array([0, 0, 0])
        self.sta = State(time_stamp, current_state, state_accumulation)

    def reward(self, action):
        condition = (self.sta.state_accumulation < self.usage_duration) and \
                    (self.sta.time_stamp == self.schedule_stop)

        reward_function = condition * self.penalty + (1 - condition) * \
                          (action * self.electricity_cost * self.appliances_consumption + \
                           (1 - action) * self.udc * self.appliances_consumption)
        return reward_function

    def step(self, action):
        self.sta.current_state = action
        self.sta.state_accumulation += action
        self.sta.time_stamp += 1

        return self.sta, self.reward(action), self.sta.time_stamp == 24


if __name__ == '__main__':
    # load_details = np.array(
    #     [['load number', 'Appliances', 's', 'f', 'l', 'r', 'udc'], [1, 'Lighting', 18, 20, 3, 0.36, 8],
    #      [2, 'Washing machine', 8, 14, 2, 0.5, 1], [3, 'Cloth dryer', 11, 17, 1, 1.8, 0]
    #         , [4, 'Dish washer', 14, 19, 2, 1.2, 9], ['Vacuum cleaner', 9, 14, 1, 0.65, 1],
    #      [6, 'Iron', 6, 9, 1, 1.1, 1], [7, 'Rice cooker', 7, 10, 1, 0.30, 0]])
    #
    # electricity_price = np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 12, 12, 5, 5, 5, 5, 10, 10, 10, 5, 5, 5])
    print(Environment.time_stamp)
    appliances_number = 3
    udc = np.array([1, 1, 1])
    appliances_consumption = np.array([3, 4, 5])
    electricity_cost = np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 12, 12, 5, 5, 5, 5, 10, 10, 10, 5, 5, 5]) / 10
    schedule_start = [10, 11, 10]
    schedule_stop = [19, 17, 21]
    usage_duration = [2, 3, 1]
    action = np.array([1, 0, 1])

    env = Environment(appliances_number, udc, appliances_consumption, electricity_cost, schedule_start,
                      schedule_stop, usage_duration, -.2)

    print(env.step(action))
    print(env.step(action))
    print(env.step(action))
    print(env.step(action))
    print(env.step(action))

    # The setting for the Environment class presented in the following

    # The setting for the state class is set in the following
