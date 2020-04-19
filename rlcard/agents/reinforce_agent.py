import torch
from torch.distributions import Categorical
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from rlcard.utils.utils import remove_illegal_torch


class PolicyNetwork(torch.nn.Module):
    def __init__(self,
                 input_size,
                 output_size):
        super().__init__()
        self.input_size = input_size
        self.linear1 = nn.Linear(input_size, 32)
        self.linear2 = nn.Linear(32, output_size)

    def forward(self, x):
        x = x.view((-1, self.input_size))
        out = F.relu(self.linear1(x))
        # TODO check that the dimension along which to do compute the softmax is correct
        out = F.softmax(self.linear2(out), dim=1)
        return out


class Policy:
    eps = np.finfo(np.float32).eps.item()

    def __init__(self,
                 action_num,
                 state_shape,
                 learning_rate,
                 discount_factor,
                 device):
        self.discount_factor = discount_factor
        self.device = device
        self._init_policy_network(action_num, state_shape)
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.log_probs = []
        self.rewards = []

    def _init_policy_network(self, action_num, state_shape):
        policy_network = PolicyNetwork(np.prod(state_shape), action_num)
        self.policy_network = policy_network.to(self.device)

    def predict(self, observed_state):
        state = torch.from_numpy(observed_state).float().to(self.device)  # To check that the shape is correct
        probs = self.policy_network(state)
        return probs

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
        policy_loss = - (torch.cat(self.log_probs) * returns).sum()

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        self.rewards = []
        self.log_probs = []

        # TODO: verify this loss is the real training loss
        return policy_loss


class ReinforceAgent:
    def __init__(self,
                 scope,
                 action_num,
                 state_shape,
                 discount_factor=0.99,
                 learning_rate=1e-5,
                 device=None):
        # TODO: check that it is correct to have use_raw == False
        self.use_raw = False
        self.scope = scope
        self._init_device(device)
        self.policy = Policy(action_num=action_num, state_shape=state_shape, learning_rate=learning_rate,
                             discount_factor=discount_factor, device=device)

    def _init_device(self, device):
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    def feed(self, ts):
        (_, _, reward, _, _) = tuple(ts)
        self.policy.rewards = reward

    def step(self, state):
        probs = self.policy.predict(state["obs"])
        # TODO: check removing the actions like this is fine for the computation of the gradient
        probs = remove_illegal_torch(probs, state["legal_actions"])
        m = Categorical(probs)
        action = m.sample()
        self.policy.log_probs.append(m.log_prob(action))
        return action.item()

    def eval_step(self, state):
        with torch.no_grad():
            probs = self.policy.predict(state["obs"])
            probs = remove_illegal_torch(probs, state["legal_actions"])
            # TODO: could be also good to keep the policy stochastic also at evaluation
            best_action = np.argmax(probs)
        return best_action, probs

    def train(self):
        return self.policy.terminate_episode()

    def get_state_dict(self):
        ''' Get the state dict to save models

        Returns:
            (dict): A dict of model states
        '''
        policy_key = self.scope + 'policy_network'
        policy = self.policy.policy_network.state_dict()
        return {policy_key: policy}

    def load(self):
        raise NotImplementedError()
