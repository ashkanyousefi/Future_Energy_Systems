import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# state class definition

class State:
    def __init__(self, appliance__initial_status, total_uptime):
        self.appliance__initial_status = appliance__initial_status
        self.total_uptime = total_uptime


class Action:
    '''
    The DQN algorithm result will provide the list of actions
    which is required to change the state.
    '''

    def __init__(self, input_action, output_action):
        self.input_action = input_action
        self.output_action = output_action


class Environment:
    def __init__(self, price_hourly, customer_preferences, number_appliances):
        self.number_appliances = number_appliances
        self.price_hourly = price_hourly
        self.customer_preferences = customer_preferences

    def step(initial_state, action):
        '''
        Moves environment one step forward based on input action given by the agent
        :param action: The new action
        :return: (new state: State, reward: Int, is episode finished (done): Boolean)
        '''

        if action[i] == 0:
            new_state[i] = initial_state[i]

        elif action[i] == 1:
            new_state[i] = not (initial_state[i])

        return new_state

    def reward_function(action, electricity_cost, appliance_consumption, udc, penalty):
        if action == 1:
            reward = np.matmul(electricity_cost, appliance_consumption)

        elif action == 0:
            reward = np.matmul(udc, appliance_consumption)

        else:
            reward = penalty

        return reward


if __name__ == '__main__':
    # create an instance of the class and initialize the parameters
    env = Environment(price_hourly, customer_preferences, number_appliances)
    sta = State(appliance__initial_status)
    act = Action(input_action, output_action)

    env.price_hourly = [i / 100 for i in range(1, 25)]
    env.customer_preferences = [0, 0, 0]
    env.number_appliances = 3
    sta.appliance_initial_status = [0, 0, 0]
    appliance_consumption = [3, 4, 5]
    penalty = -0.2

    for i in range(25):
        new_state = step(old_state, action)
        old_state = new_state
        total_reward = reward_function(action, price_hourly, appliance_consumption, udc, penalty)
