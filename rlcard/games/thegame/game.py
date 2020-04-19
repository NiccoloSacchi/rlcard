from copy import deepcopy, copy
import numpy as np
import random

from rlcard.games.thegame import TheGameDealer as Dealer, TheGamePlayer as Player, Card

# TODO: manage seed

class TheGameState(object):
    ''' The state of the game. This will be passed to the player to select the next action. '''
    def __init__(self, game_pointer=0, public_cards=[], num_cards_played=0, players=[], dealer=None, legal_actions=[]):
        
        # Pointer to the current player
        self.game_pointer = game_pointer
        
        # Public cards (4 decks in total)
        self.public_cards = public_cards

        # #cards played by the current player in the current round
        self.num_cards_played = num_cards_played
        
        # Players playing to the game
        self.players = players
        
        # The dealer giving the cards
        self.dealer = dealer
        
        # Actions that can be executed on the currect state by the current player
        self.legal_actions = []


    def clone(self):
        ''' Clone the state of the game. '''
        return deepcopy(self)


    def get_player(self):
        ''' Return the current player. '''
        return self.players[self.game_pointer]


class TheGameGame(object):

    def __init__(self, game_size=100, hand_size=6, min_cards=2, players=[], verbose=0):  # , allow_step_back=False):
        '''
        Initialize The Game
        
        Args:
            game_size (int): the size of the game, e.g. if game_size=100 the game is composed by 98 cards (from 2 to 99)
            hand_size (int): number of cards per hand
            min_cards (int): minimum number of cards to be played per player
            players (list of TheGamePlayer): list of players that will take part to The Game
            verbose (int): level of verbosity of the game (TODO)
        '''
#         self.allow_step_back = allow_step_back

        assert len(players) <= 6, 'Maximum 6 players.'

        self.game_size = game_size
        self.hand_size = hand_size
        self.min_cards = min_cards
        self.verbose = verbose
        
        self.history = []
        self.seed = None

        self.state = TheGameState(
            players=players,
            dealer=Dealer(n=self.game_size)
        )

        for p in players:
            p.set_game(self)

        self.reset()

    def reset(self):
        ''' Resets The Game for a new game '''
        
        # Reset the card of the dealer
        self.state.dealer.reset()
        
        # Mix the players, the first one is in index 0
        random.shuffle(self.state.players)
        self.state.game_pointer = 0

        # Deal cards to each  player to prepare for the first round
        for i in range(len(self.state.players)):
            self.state.players[i].reset()
            self.fill_hand(i)

        # Initialize the game state
        self.state.public_cards = [Card(1), Card(1), Card(self.game_size), Card(self.game_size)]  # first two go up, last two go down
        self.state.num_cards_played = 0
        self.state.legal_actions = self.get_legal_actions()

        # Save the hisory for stepping back to the last state.
        self.history = []


    def set_seed(self, seed):
        ''' Set the seed that will be used by dealers and players (TODO: players are not deterministic only during training). Pass None to remove the current seed. '''
        self.seed = seed
        self.state.dealer.set_seed(seed)


    def run(self):
        ''' Run a complete The Game '''
        
        # Reset a new game
        self.reset()
        
        # Go on until we do not win
        while not self.is_over():
            # Let the player pick the action
            player = self.state.get_player()
            # TODO: possibly pass to the player only its view of the game (e.g. he cannot see other's hands)
            action = player.get_action(self.state, self.state.legal_actions)
            
            # Execute the action and update the game's state
            next_state, reward = self.step(action)
            
            # Give the reward to the player
            player.give_reward(next_state, reward)

            if self.verbose >= 1:
                print('Reward:', reward)
        
        if self.verbose >= 1:
            if self.have_won():
                print('You have won!')
            elif self.have_lost():
                print('You have lost...')
            else:
                print('You have... wait what?')


    def step(self, action, state=None, inplace=True):
        '''
        Execute the action on the state.

        Args:
            action (tuple of int): a specific action.
            state (TheGameState, optional): if given, the action is executed on it. If not given the actions is
                executed on the game's state.
            inplace (bool): if True, the state is update in place, otherwise a new state is returned.
        
        Resturns:
            dict: next state
            int: reward
        '''

        # If no state is given then the game's state is used
        if state is None:
            state = self.state

        # If it is not in place then we clone the given state so that we do not modify it
        if not inplace:
            state = state.clone()

        if action is None:
            # Move the pointer to the next player
            self.move_to_next_player()
        else:
            # Execute the action
            player = state.get_player()
            player_card_index, public_cards_index = action
            state.public_cards[public_cards_index] = player.hand[player_card_index]
            del player.hand[player_card_index]
            state.num_cards_played += 1
        
        # Get the possible actions for the next player
        state.legal_actions = self.get_legal_actions(state)

        # The reward is 0 if we are still playing, otherwise it grows exponentially with the number of played cards
        reward = 0
        num_cards_remaining = sum([len(p.hand) for p in state.players], len(self.state.dealer.deck))
        if (len(state.legal_actions) == 0) or (num_cards_remaining == 0):
            perc_game_done = (self.game_size - num_cards_remaining)/self.game_size
            reward = 2**(10*perc_game_done)-1
        return state, reward


    def move_to_next_player(self):
        ''' Move the pointer to the next player and update the game's state accordingly. '''
        while True:
            self.state.game_pointer = (self.state.game_pointer + 1) % len(self.state.players)
            if self.state.players[self.state.game_pointer].is_alive():
                self.state.num_cards_played = 0
                break


    def fill_hand(self, game_pointer):
        ''' Fill the hand of the player game_pointer. '''
        player = self.state.players[game_pointer]
        dealer = self.state.dealer
        # Draw the cards
        while len(player.hand) < self.hand_size and dealer.has_cards():
            player.add_card(dealer.deal_card())


    def validate_state(self, state):
        ''' Check that the given state is valid. '''
        
        assert isinstance(state, TheGameState), 'The state must be of type TheGameState but is instead of type {}.'.format(type(state))
        
        assert len(state.public_cards) == 4, "There must be only 4 cards on the table but len(state.public_cards) != 4."
        assert all([(c.rank < self.game_size) and (c.rank > 1) for c in state.public_cards]), 'The given public cards are not valid: {}'.format(state.public_cards)


    def validate_action(self, action, state):
        ''' Check that the given action is executable on the given state. '''
        
        player_card_index, public_cards_index = action
        player = state.get_player()
        play_card = player.hand[player_card_index]
        public_card = state.public_cards[public_cards_index]
        assert public_cards_index < 4 and public_cards_index >= 0, 'The action must put a card on one of the 4 decks on the table, {} is not in [0, 3]'.format(public_cards_index)

        if public_cards_index < 2:
            assert (public_card < play_card) or (play_card == public_card - 10), 'You cannot play {} on {} (which goes up)'.format(play_card, public_card)
        else:
            assert (public_card > play_card) or (play_card == public_card + 10), 'You cannot play {} on {} (which goes down)'.format(play_card, public_card)
            
        

#     def step_to_state(self, state):
#         ''' Return to the previous state of the game

#         Returns:
#             (bool): True if the game steps back successfully
#         '''
#         if len(self.history) > 0:
#             self.state = self.history.pop()
#             return True
#         return False

    def get_player_num(self):
        ''' Return the number of players in The Game

        Returns:
            (int): The number of players in the game
        '''
        return len(self.state.players)


    def get_player_id(self):
        ''' Return the current player's id

        Returns:
            (int): current player's id
        '''
        return self.state.players[self.state.game_pointer].player_id


    def is_over(self):
        ''' Check if the game is over

        Returns:
            (boolean): True if the game is over
        '''
        
        return self.have_won() or self.have_lost()


    def have_won(self):
        ''' Return True if the players have won against The Game (i.e. there are no cards to be played/drawed), False otherwise. '''
        return len(self.state.dealer.deck) == 0 and all(len(p.hand)==0 for p in self.state.players)


    def have_lost(self):
        ''' Return True if the players have lost against The Game (i.e. the current player cannot play any card). '''
        return len(self.state.legal_actions) == 0


    def get_legal_actions(self, state=None):
        ''' Return the legal actions for current player
        
        Args:
            state (TheGameState, optional): if given, it is used to compute the legal actions. If not given the game's state is used instead.

        Returns:
            (list or tuples): A list of legal actions represented as tuples (player card index, public_cards index). None indicates
                that the player might take no action.

            Examples:
                (3, 1): play player.hand[3] in public_cards[1]
                (1, 0): play player.hand[1] in public_cards[0]
        '''

        if state is None:
            state = self.state

        def playable(hand_card_rank, public_card_rank):
            if hand_card_rank > public_card_rank:
                return True
            if hand_card_rank == public_card_rank - 10:
                return True
            return False
        
        actions = []
        player = self.state.get_player()
        for i, hand_card in enumerate(player.hand):
            # Can hand_card be played in the decks that go up?
            for j in range(2):
                if playable(hand_card.rank, state.public_cards[j].rank):
                    actions.append((i, j))
            
            # Can hand_card be played in the decks that go down?
            for j in range(2, 4):
                if playable(-hand_card.rank, -state.public_cards[j].rank):
                    actions.append((i, j))
        
        # If the player already played the minimum number of cards, he might chose to stop (take no action)
        min_cards = self.min_cards if self.state.dealer.has_cards() else 1
        if state.num_cards_played >= min_cards:
            actions.append(None)

        return actions
        