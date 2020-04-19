from .player import Player
import numpy as np
from scipy.special import comb


class SARSAgent(Player):
    def __init__(self, player_id):
        super(self).__init__(player_id)
    
    
    def set_game(self, game):
        self.game = game

        game_size = game.game_size
        hand_size = game.hand_size
        # all possible cards in hand * all posible cards on the table * whether it is possible not to play any card
        state_size = [game_size for _ in range(hand_size)] + [game_size for _ in range(4)] + [2]
        
        # each card in hand can be put on one of the 4 deck (+1 None action)
        action_size = [hand_size, 4]
        
        # Initialize all the rewards to 0
        self.q_s_a = {
            # for eaxh state decide which card to put where
            'action': np.zeros(state_size + action_size)

            # for each state you might decide (if possible to do no action)
            'no-action': np.zeros(state_size)
        }

    
     def get_action(self, state, legal_actions):


    def give_reward(self, next_state, reward):
        ''' Give the reward for the last action. '''
        pass