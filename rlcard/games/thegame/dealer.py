import random

from rlcard.games.thegame.utils import Card

class TheGameDealer(object):

    def __init__(self, n=100):
        ''' Initialize a TheGame dealer class for a game of size n. '''
        super().__init__()
        self.n = n

        self.seed = None
        self.reset()

    def reset(self):
        ''' Create and shuffle the deck. '''
        self.deck = [Card(i) for i in range(2, self.n)]
        self.shuffle()

    def shuffle(self):
        ''' Shuffle the deck
        '''
        if self.seed is not None:
            random.seed(self.seed)
        random.shuffle(self.deck)
        
    def has_cards(self):
        ''' Returns True is there are still cards in the deck. '''
        return len(self.deck) > 0

    def deal_card(self):
        ''' Deal one card from the deck

        Returns:
            (Card): The drawn card from the deck
        '''
        return self.deck.pop()
    
    def set_seed(self, seed):
        ''' Set the seed that will be used by the dealer to shuffle the cards. '''
        self.seed = seed
    
    def __str__(self):
        return ', '.join([str(c) for c in self.deck])
            
