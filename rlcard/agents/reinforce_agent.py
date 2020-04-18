import torch
from torch.distributions import Categorical
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PolicyNetwork(torch.nn.Module):
    def __init__(self,
                 input_size,
                 output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, 32)
        self.linear2 = nn.Linear(32, output_size)

    def forward(self, x):
        out = F.relu(self.linear1(x))
        out = F.softmax(self.linear2(out))
        return out

class Policy:
    def __init__(self, action_num, state_shape, learning_rate, device):
        self.learning_rate = learning_rate
        self.device = device
        self._init_policy_network(action_num, state_shape)
        self.optimizer = torch.optim.Adam(self.policy_network.parameters)
        self.log_probs = []
        self.rewards = []

    def _init_policy_network(self, action_num, state_shape):
        policy_network = PolicyNetwork(np.prod(state_shape), action_num)
        self.policy_network = policy_network.to(self.device)

    def select_action(self, state):
        state = torch.from_numpy(state).float()  # To check that the shape is correct
        probs = self.policy_network(state)
        m = Categorical(probs)
        action = m.sample()
        self.log_probs.append(m.log_prob(action))
        return action.item()

    def terminate_episode(self):
        G = 0


class ReinforceAgent:
    def __init__(self,
                 discount_factor=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.1,
                 epsilon_decay_steps=20000,
                 batch_size=32,
                 learning_rate=1e-5,
                 device=None):
        self.discount_factor = discount_factor
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self._init_device(device)

    def _init_device(self, device):
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

