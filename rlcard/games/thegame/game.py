# from copy import deepcopy, copy
import numpy as np
import random

from rlcard.games.thegame import TheGameDealer as Dealer, TheGamePlayer as Player, Card

# TODO: manage seed

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
        self.players = players
        
        self.history = []
        self.game_pointer = 0
        self.num_cards_played = 0  # number of cards played by the current player
        self.seed = None
        self.dealer = Dealer(n=self.game_size)
        
#         for p in players:
#             p.set_game(self)

        self.reset()

    def reset(self):
        ''' Resets The Game for a new game '''
        
        # Reset the card of the dealer
        self.dealer.reset()
        
        # Mix the players, the first one is in index 0
        random.shuffle(self.players)
        self.game_pointer = 0

        # Deal cards to each  player to prepare for the first round
        for p in self.players:
            p.reset()
            for _ in range(self.hand_size):
                p.add_card(self.dealer.deal_card())

        # Initialize public cards, first two go up, last two go down
        self.public_cards = [Card(1), Card(1), Card(self.game_size), Card(self.game_size)]

        # Save the hisory for stepping back to the last state.
        self.history = []

    def set_seed(self, seed):
        ''' Set the seed that will be used by dealers and players (TODO: players are not deterministic only during training). Pass None to remove the current seed. '''
        self.seed = seed
        self.dealer.set_seed(seed)
        
    def run(self):
        ''' Run a complete The Game '''
        
        # Reset a new game
        self.reset()
        
        # Go on until we do not win
        while not self.have_won():
            # Get the current state (the hand of the player, the public cards, ...) and the possible actions for the player
            curr_state = self.get_state()
            legal_actions = self.get_legal_actions()
            
            # Check if we have lost
            if len(legal_actions) == 0:
                print('We have lost, the player cannot play anything')
                return
        
            # Let the player pick the action
            action = self.players[self.game_pointer].get_action(curr_state, legal_actions)
            
            # Execute the action
            self.step(action)
            
            # TODO: sent the rewards to the agents
            # curr_player.give_reward(...)
            
#         return state, reward           

    def step(self, action):
        '''
        Get the next state

        Args:
            action (str): a specific action. (call, raise, fold, or check)
        '''
#         if self.allow_step_back:
#             # First snapshot the current state
#             r = deepcopy(self.round)
#             b = self.game_pointer
#             r_c = self.round_counter
#             d = deepcopy(self.dealer)
#             p = deepcopy(self.public_cards)
#             ps = deepcopy(self.players)
#             rn = copy(self.history_raise_nums)
#             self.history.append((r, b, r_c, d, p, ps, rn))

        curr_player = self.players[self.game_pointer]
        if action == None:
            # The player stops playing

            # Draw the cards
            while len(curr_player.hand) < self.hand_size and self.dealer.has_cards():
                curr_player.add_card(self.dealer.deal_card())

            # Move the pointer to the next player
            while True:
                self.game_pointer = (self.game_pointer + 1) % len(self.players)
                if self.players[self.game_pointer].is_alive():
                    break
            self.num_cards_played = 0
        else:
            # Update the state of the game
            player_card_index, public_cards_index = action
            self.public_cards[public_cards_index] = curr_player.hand[player_card_index]
            del curr_player.hand[player_card_index]
            self.num_cards_played += 1

    def get_next_state(self, action):
        ''' Executes the action and returns the next state and the reward. '''
        raise NotImplementedError

#     def step_back(self):
#         ''' Return to the previous state of the game

#         Returns:
#             (bool): True if the game steps back successfully
#         '''
#         if len(self.history) > 0:
#             self.round, self.game_pointer, self.round_counter, self.dealer, self.public_cards, self.players, self.history_raises_nums = self.history.pop()
#             return True
#         return False

    def get_player_num(self):
        ''' Return the number of players in The Game

        Returns:
            (int): The number of players in the game
        '''
        return len(self.players)
    
    def get_state(self):
        ''' Return the current state of the game. '''
        
        return {
            # public cards (4 decks in total)
            'public_cards': self.public_cards,
            
            # #cards played by the current player in the current round
            'num_cards_played': self.num_cards_played,

            # hand of the current player
            'hand': self.players[self.game_pointer].hand
        }

    def get_player_id(self):
        ''' Return the current player's id

        Returns:
            (int): current player's id
        '''
        return self.players[self.game_pointer].player_id

    def is_over(self):
        ''' Check if the game is over

        Returns:
            (boolean): True if the game is over
        '''
        
        return self.have_won() or self.have_lost()
    
    def have_won(self):
        ''' Return True if the players have won against The Game (i.e. there are no cards to be played/drawed), False otherwise. '''
        return len(self.dealer.deck) == 0 and all(len(p.hand)==0 for p in self.players)
    
    def have_lost(self):
        ''' Return True if the players have lost against The Game (i.e. the current player cannot play any card). '''
        curr_player = self.players[self.game_pointer]
        return len(curr_player.get_legal_actions(self.get_state())) == 0

#     def get_payoffs(self):
#         ''' Return the payoffs of the game

#         Returns:
#             (list): Each entry corresponds to the payoff of one player
#         '''
#         hands = [p.hand + self.public_cards if p.status=='alive' else None for p in self.players]
#         chips_payoffs = self.judger.judge_game(self.players, hands)
#         payoffs = np.array(chips_payoffs) / (self.big_blind)
#         return payoffs        
        
    def get_legal_actions(self):
        ''' Return the legal actions for current player

        Returns:
            (list or tuples): A list of legal actions represented as tuples (player card index, public_cards index). None indicates
                that the player might take no action.

            Examples:
                (3, 1): play player.hand[3] in public_cards[1]
                (1, 0): play player.hand[1] in public_cards[0]
        '''

        def playable(hand_card_rank, public_card_rank):
            if hand_card_rank > public_card_rank:
                return True
            if hand_card_rank == public_card_rank - 10:
                return True
            return False
        
        actions = []
        
        hand = self.players[self.game_pointer].hand
        for i, hand_card in enumerate(hand):
            # Can hand_card be played in the deck that go up?
            for j in range(2):
                if playable(hand_card.rank, self.public_cards[j].rank):
                    actions.append((i, j))
            
            # Can hand_card be played in the deck that go down?
            for j in range(2, 4):
                if playable(-hand_card.rank, -self.public_cards[j].rank):
                    actions.append((i, j))
        
        # If the player already played the minimum number of cards, he might chose to stop (take no action)
        min_cards = self.min_cards if self.dealer.has_cards() else 1
        if self.num_cards_played >= min_cards:
            actions.append(None)

        return actions
        