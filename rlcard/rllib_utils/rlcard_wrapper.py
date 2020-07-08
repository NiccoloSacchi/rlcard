import rlcard
from gym.spaces import Dict, Discrete, Tuple, Box
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
import random


class RLCardWrapper(MultiAgentEnv):
    """
    This class wraps a RLCard environment so to make it compatible to RLlib.
    As RLCard environments have N-player and a variable action space we have to:
    - Create a RLlib multi-agent environment: https://docs.ray.io/en/latest/rllib-env.html?multi-agent-and-hierarchical#multi-agent-and-hierarchical
    - Make the environment return a mask for the actions: https://docs.ray.io/en/master/rllib-models.html#variable-length-parametric-action-spaces

    Of the rlcard environment you have to pass the name, the low and high values of the observation space
    """
    def __init__(self, env_config={}):
        # create the rlcard environment
        self.rlcard_env = rlcard.make(env_config['rlcard_env_id'])

        # state and action spaces
        self.action_space = Discrete(self.rlcard_env.action_num)  # number of actions in this game
        self.observation_space = Dict({
            "real_obs": Box(low=-float('inf'), high=float('inf'), shape=self.rlcard_env.state_shape),

            # we have to handle changing action spaces
            "action_mask": Box(0, 1, shape=(self.rlcard_env.action_num,)),
        })

        # these players will have to be randomized for evaluation purposes
        self.randomize_agents_eval = []
        if "randomize_agents_eval" in env_config:
            self.randomize_agents_eval = env_config["randomize_agents_eval"]

        # instantiate the player names
        self.players = ["player_{}".format(i+1) for i in range(self.rlcard_env.player_num)]

    def reset(self):

        # get state and player pointer from rlcard env
        state, player_id = self.rlcard_env.init_game()

        # --------------------------------------------
        # state = {
        #     'legal_actions': [0, 1, 2],
        #     'obs': array([
        #         0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        #         0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        #         0., 0.])  # len 36
        # }  # self.rlcard_env.state_shape = [36]
        # --------------------------------------------

        # !! not needed as RLcard environment return the reward only a the end of the game !!
        # reward is given to the last player with 1 delay
        # self.reward_buffer = {p: 0 for p in self.players}

        return {self.players[player_id]: self.get_state(state)}

    def step(self, action_dict):
        # There is always only one player playing per turn: take and execute the action of the current player
        assert len(action_dict) == 1
        player, action = action_dict.popitem()
        if player in self.randomize_agents_eval:
            # randomize for evaluation purposes
            action = random.sample(self.rlcard_env._get_legal_actions(), 1)[0]
            if isinstance(action, str):
                # convert to integer
                assert hasattr(self.rlcard_env, 'actions'), 'The environment is returning string actions and does not have the list of possible actions...'
                action = self.rlcard_env.actions.index(action)
        next_state, next_player_id = self.rlcard_env.step(action, raw_action=False)

        # Get the state for the next player, reward is defaulted to 0 until the end of the game
        next_player_name = self.players[next_player_id]
        reward = {next_player_name: 0}
        obs = {next_player_name: self.get_state(next_state)}
        done = {"__all__": False}

        # if the game is done we get the rewards for all the players
        if self.rlcard_env.game.is_over():
            reward = {name: r for name, r in zip(self.players, self.rlcard_env.get_payoffs())}
            obs = {self.players[player_id]: self.get_state(self.rlcard_env.get_state(player_id)) for player_id in range(len(self.players))}
            done = {"__all__": True}

        return obs, reward, done, {}

    def get_state(self, rlcard_state):
        mask = np.zeros(self.action_space.n)
        mask[rlcard_state['legal_actions']] = 1
        return {
            'real_obs': rlcard_state['obs'],
            'action_mask': mask
        }
