from torch import nn

from environment import Environment
from path_configs import ROOT_PATH


class DQNModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *input):
        pass

    def save(self):
        pass

    def load(self):
        pass


class DQNTrainer:
    def __init__(self, environment: Environment, dqn_model: DQNModel):
        pass

    def train(self):
        pass

######### pseudo code #########
# env = Environment()
# model = DQNModel()
# trainer = DQNTrainer(env, model)
# trainer.train()
