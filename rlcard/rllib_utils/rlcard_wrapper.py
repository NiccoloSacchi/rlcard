import rlcard
from gym.spaces import Dict, Discrete, Tuple, Box
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np


class RLCard2RLlib(MultiAgentEnv):
    """
    This class wraps a RLCard environment so to make it compatible to RLlib.
    As RLCard environments have N-player and a variable action space we have to:
    - Create a RLlib multi-agent environment: https://docs.ray.io/en/latest/rllib-env.html?multi-agent-and-hierarchical#multi-agent-and-hierarchical
    - Make the environment return a mask for the actions: https://docs.ray.io/en/master/rllib-models.html#variable-length-parametric-action-spaces
    """

    def __init__(self, rlcard_env_id, config=None):
        # create the rlcard environment
        self.rlcard_env = rlcard.make(rlcard_env_id)

        # state and action spaces
        self.action_space = Discrete(self.rlcard_env.action_num)  # number of actions in this game

        # TODO: are the values in the state all in [0-1]?
        self.observation_space = Dict({
            "real_obs": Box(low=0, high=1, shape=self.rlcard_env.state_shape),  # example: self.rlcard_env.state_shape = [36]

            # we have to handle changing action spaces
            "action_mask": Box(0, 1, shape=(self.rlcard_env.action_num,)),
        })

        # rlcard_env.num_players has the number of players in this game
        self.players = ["player_{}".format(i+1) for i in range(self.rlcard_env.num_players)]

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

        return {self.curr_player: self.get_state(state)}

    def step(self, action_dict):
        # There is always only one player playing per turn: take and execute the action of the current player
        curr_player_id = self.rlcard_env.active_player
        action = action_dict[self.players[curr_player_id]]
        next_state, next_player_id = self.rlcard_env.step(action, self.agents[curr_player_id].use_raw)

        # Get atste for the next player, reward is defaulted to 0 until the end of the game
        next_player_name = self.players[next_player_id]
        reward = {next_player_name: 0}
        obs = {next_player_name: self.get_state(next_state)}
        done = {}

        # if the game is done we get the rewards for all the players
        if self.rlcard_env.game.is_over():
            reward = {name: r for name, r in zip(self.players, self.rlcard_env.get_payoffs())}
            obs = {self.players[player_id]: self.get_state(self.rlcard_env.get_state(player_id)) for player_id in range(len(self.players))}
            done = {"__all__": False}

        return obs, reward, done, {}

    def get_state(self, rlcard_state):
        mask = np.zeros(self.action_space.n)
        mask[rlcard_state['legal_actions']] = 1
        return {
            'real_obs': rlcard_state['obs'],
            'action_mask': mask
        }
