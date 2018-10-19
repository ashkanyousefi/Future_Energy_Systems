import matplotlib.pyplot as plt
import numpy as np


# state class definition
class State:
    def __init__(self, number_of_appliances):
        self.value = np.zeros(
            number_of_appliances + 1,
            dtype=int)  # first element is the current timestep, the others are total consumption


class Action:
    '''
    The DQN algorithm result will provide the list of actions
    which is required to change the state.
    '''

    def __init__(self, number_of_appliances):
        self.value = np.zeros((number_of_appliances,), dtype=int)


class Environment:
    def __init__(self, c, udc, r, s, l, f, penalty):
        self.c = c
        self.udc = udc
        self.r = r
        self.s = s
        self.l = l
        self.f = f
        self.penalty = penalty
        self.number_appliances = self.r.shape[0]
        self.current_state = State(self.number_appliances)

    def step(self, actions: Action):
        '''
        Moves environment one step forward based on input action given by the agent
        :param action: The new action
        :return: (new state: State, reward: Int, is episode finished (done): Boolean)
        '''

        reward = self.reward_function(actions)
        self.current_state.value[0] += [1]
        self.current_state.value[1:] += actions.value
        return self.current_state.value.copy(), reward, self.current_state.value[0] == 24

    def reward_function(self, actions: Action):
        t = self.current_state.value[0]
        c_t = self.c[t]
        a1_case = c_t * self.r * actions.value
        a0_case = self.udc * self.r * (1 - actions.value)
        penalty_mask = (self.f == t) * (self.current_state.value[1:] < self.l)
        return penalty_mask * self.penalty + (1 - penalty_mask) * (a0_case + a1_case)


if __name__ == '__main__':
    # create an instance of the class and initialize the parameters

    number_appliances = 3
    price_hourly = np.array([i / 100 for i in range(1, 25)])
    customer_preferences = np.array([0 for _ in range(number_appliances)])
    appliance_consumption = np.array([3 + np.random.randint(-2, 3) for _ in range(number_appliances)])
    starts = np.array([0 for _ in range(number_appliances)])
    durations = np.array([2 for _ in range(number_appliances)])
    finals = np.array([10 for _ in range(number_appliances)])
    penalty = -0.2

    env = Environment(price_hourly, customer_preferences, appliance_consumption, starts, durations, finals, penalty)
    total_reward = 0
    done = False
    print(env.current_state.value)
    while not done:
        act = Action(number_appliances)
        appliance_id = np.random.randint(0, number_appliances)
        act.value[appliance_id] = 1
        new_state, action_reward, done = env.step(act)
        print(f'new state: {new_state}, action: {act.value}, reward: {action_reward}')
        total_reward += action_reward
    print(f'total reward: {total_reward}')
