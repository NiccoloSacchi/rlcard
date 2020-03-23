class Card(object):
    '''
    Card stores the number and the effect (todo) of the card
    '''
    rank = None

    def __init__(self, rank):
        ''' Initialize the rank of a card

        Args:
            rank: int, rank of the card
        '''
        self.rank = rank

    def get_index(self):
        ''' Get index of a card.

        Returns:
            string: the combination of rank and effect of a card. Eg: 1, 2, 3, ...
        '''
        return str(self.rank)
    
    def __str__(self):
        return 'Card {}'.format(self.rank)