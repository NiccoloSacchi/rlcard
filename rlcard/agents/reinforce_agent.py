import torch
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
    def __init__(self, action_num, state_shape):
        self.policy_network = PolicyNetwork(np.prod(state_shape), action_num)

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

