import sys
import json

class TheGamePlayer(object):

    def __init__(self, player_id):
        ''' Initialize a player.

        Args:
            player_id (int): The id of the player
        '''
        self.player_id = player_id
        self.hand = []
    
    def set_game(self, game):
        '''
        Args:
            game (TheGameGame): pointer to the game that this player will play
        '''
        self.game = game
    
    def reset(self):
        ''' Reset the status of the player. '''
        self.hand = []
        
    def is_alive(self):
        ''' Return the status of the player. '''
        return len(self.hand) != 0
    
    def add_card(self, card):
        ''' Add the given card to the player's hand. '''
        # We keep the hand sorted
        for i, c in enumerate(self.hand):
            if card.rank < c.rank:
                self.hand.insert(i, card)
                return
        # it is bigger than all the cards in the hand
        self.hand.append(card)
            

    def get_player_id(self):
        ''' Return the id of the player
        '''
        return self.player_id
    
#     def get_hand(self):
#         ''' Return the list containing the hand of the player. '''
#         return [c.get_index() for c in self.hand]
    
    def get_action(self, state, legal_actions):
        ''' Returns the next action. If there is no possible action returns None. '''
        
        # if there is no possible action either we have lost or the player finisched the cards
        if len(legal_actions) == 0:
            return None
        
        # The user is expected to provide the action
        while True:
            choice = input('\nPlayer: {}.\nState:{}.\nLegal actions:\n\t{}\nProvide the index of the action or the action itself: '.format(self.player_id, self.state_to_str(state), legal_actions))
            if choice == 'exit':
                sys.exit()

            if choice.isdigit():
                choice = int(choice)
                if choice >= 0 and choice < len(legal_actions):
                    return legal_actions[choice]
            else:
                try:
                    if choice == 'None':
                        action = None
                    else:
                        action = tuple(json.loads('[' + choice + ']'))
                    if action in legal_actions:
                        return action
                except Exception as e:
                    pass
            print('The choice is not valid.')
            
    def state_to_str(self, state):
        ''' Converts the state to a string. '''
        state_str = ''
        state_str += '\n\tPublic Cards: {}'.format([c.get_index() for c in state.public_cards])
        state_str += '\n\tPlayer Hand: {}'.format([c.get_index() for c in self.hand])
        state_str += '\n\t#Cards Played: {}'.format(state.num_cards_played)
        return state_str
    
    def __str__(self):
        ''' Convert to string the id and the hand of the player. '''
        return '{} hand: {}'.format(self.player_id, [c.rank for c in self.hand])
    
    def give_reward(self, next_state, reward):
        ''' Give the reward for the last action. '''
        pass
