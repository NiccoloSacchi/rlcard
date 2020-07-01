# Implementation of the Multi Agent Env. game

from gym.spaces import Dict, Discrete, Tuple, Box
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import random


class Actions:
    # number of actions
    SIZE = 3

    # types o actions
    ROCK = 0
    PAPER = 1
    SCISSORS = 2
    NA = 3  # Not Available, hand not yet played


class RockPaperScissors(MultiAgentEnv):
    """
    Two-player environment for the famous rock paper scissors game, modified:
    - There are two agents which alternate, the action of one agent provides the
        state for the next agent. Since one of the two players begins, the agent
        which starts second should learn to always win! The startign player
        is drawn randomly.
    - The action space changes. The game is divided in three rounds across
        which you can't re-use the same action.
    """

    # Action/State spaces
    ACTION_SPACE = Discrete(Actions.SIZE)

    OBSERVATION_SPACE = Dict({
        "real_obs": Tuple((
            # First round
            Tuple((Discrete(4), Discrete(4))),

            # Second round
            Tuple((Discrete(4), Discrete(4))),

            # Third round
            Tuple((Discrete(4), Discrete(4))),
        )),

        # we have to handle changing action spaces
        "action_mask": Box(0, 1, shape=(Actions.SIZE,)),
    })

    # Reward mapping
    rewards = {
        (Actions.ROCK, Actions.ROCK): (0, 0),
        (Actions.ROCK, Actions.PAPER): (-1, 1),
        (Actions.ROCK, Actions.SCISSORS): (1, -1),
        (Actions.PAPER, Actions.ROCK): (1, -1),
        (Actions.PAPER, Actions.PAPER): (0, 0),
        (Actions.PAPER, Actions.SCISSORS): (-1, 1),
        (Actions.SCISSORS, Actions.ROCK): (-1, 1),
        (Actions.SCISSORS, Actions.PAPER): (1, -1),
        (Actions.SCISSORS, Actions.SCISSORS): (0, 0),
    }

    def __init__(self, config=None):

        # state and action spaces
        self.action_space = self.ACTION_SPACE
        self.observation_space = self.OBSERVATION_SPACE

        self.players = ["player_1", "player_2"]

    def reset(self):
        self.player_scores = {p: 0 for p in self.players}  # just used to collect the scores
        self.curr_round = 0
        self.player_pointer = random.randint(0, 1)
        self.state = [
            [3, 3],
            [3, 3],
            [3, 3],
        ]

        # reward is given to the last player with 1 delay
        self.reward_buffer = {p: 0 for p in self.players}

        # actions cannot be reused across one game, we keep a mask for each player
        self.action_mask = {p: [1 for _ in range(self.action_space.n)] for p in self.players}

        return {self.players[self.player_pointer]: self.get_state(self.players[self.player_pointer])}

    def step(self, action_dict):
        # Get current player
        curr_player_pointer = self.player_pointer
        curr_player = self.players[self.player_pointer]

        # Get next player
        next_player_pointer = (self.player_pointer + 1) % 2
        next_player = self.players[next_player_pointer]

        # Make sure you have the ation only for the current player
        assert curr_player in action_dict and len(action_dict) == 1, \
            "{} should be playing but action {} was received.".format(curr_player, action_dict)

        # Play the action
        curr_action = action_dict[curr_player]
        assert self.action_space.contains(curr_action), 'Action {} is not valid'.format(curr_action)
        assert self.state[self.curr_round][curr_player_pointer] == Actions.NA, \
            "Player {} has already played in round {}. Here the current state: {}".format(
                curr_player_pointer,
                self.curr_round,
                self.state
            )
        assert self.action_mask[curr_player][curr_action] == 1, \
            '{} has already played action {}. State: {}'.format(curr_player, curr_action, self.state)
        self.action_mask[curr_player][curr_action] = 0  # mask out this action
        self.state[self.curr_round][curr_player_pointer] = curr_action

        # We might be not done yet
        done = {"__all__": False}

        # If the next player has already played, the round is done
        game_done = False
        round_done = self.state[self.curr_round][next_player_pointer] != Actions.NA
        if round_done:
            # If the round is done we compute the rewards
            curr_rewards = self.rewards[tuple(self.state[self.curr_round])]
            self.player_scores["player_1"] += curr_rewards[0]
            self.player_scores["player_2"] += curr_rewards[1]
            self.reward_buffer[curr_player] = curr_rewards[curr_player_pointer]

            self.curr_round += 1
            if self.curr_round == 3:
                done = {"__all__": True}
                # Return reward and state for all players
                reward = self.reward_buffer
                obs = {p: self.get_state(next_player) for p in self.players}
                game_done = True

        # Get the state and reward for the next player
        if not game_done:
            obs = {next_player: self.get_state(next_player)}
            reward = {next_player: self.reward_buffer[next_player]}

        # Move pointer to next player
        self.player_pointer = next_player_pointer
        return obs, reward, done, {}

    def get_state(self, player):
        return {
            'real_obs': self.state,
            'action_mask': self.action_mask[player]
        }
