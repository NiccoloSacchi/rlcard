import torch
from torch.distributions import Categorical
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from rlcard.utils.utils import remove_illegal


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
    eps = np.finfo(np.float32).eps.item()

    def __init__(self,
                 action_num,
                 state_shape,
                 learning_rate,
                 discount_factor,
                 device):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.device = device
        self._init_policy_network(action_num, state_shape)
        self.optimizer = torch.optim.Adam(self.policy_network.parameters)
        self.log_probs = []
        self.rewards = []

    def _init_policy_network(self, action_num, state_shape):
        policy_network = PolicyNetwork(np.prod(state_shape), action_num)
        self.policy_network = policy_network.to(self.device)

    def predict(self, observed_state):
        state = torch.from_numpy(observed_state).float()  # To check that the shape is correct
        probs = self.policy_network(state)
        return probs

    def step(self, state):
        probs = self.predict(state["obs"])
        # TODO: check removing the actions like this is fine for the computation of the gradient
        probs = remove_illegal(probs, state["legal_actions"])
        m = Categorical(probs)
        action = m.sample()
        self.log_probs.append(m.log_prob(action))
        return action.item()

    def eval_step(self, state):
        with torch.no_grad():
            probs = self.predict(state["obs"])
            probs = remove_illegal(probs, state["legal_actions"])
            # TODO: check that a greedy policy is obtained correctly with this step
            best_action = np.argmax(probs)
        return best_action

    def terminate_episode(self):
        G = 0
        returns = []
        for r in self.rewards[::-1]:
            G = r + self.discount_factor * G
            returns.insert(0, G)
        returns = torch.tensor(returns)

        # TODO: check why it is better to normalize the returns
        returns = (returns - returns.mean()) / (returns.std() + self.eps)

        # TODO: check that this operation is correctly done element-wise
        # TODO: verify that the data are in the correct device - still not clear to me
        policy_loss = - (torch.tensor(self.log_probs) * returns).sum()

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        self.rewards = []
        self.log_probs = []


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

