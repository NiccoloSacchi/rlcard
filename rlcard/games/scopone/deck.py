class Card:
    def __init__(self, value, suit, card_id, index):
        self.value = value
        self.suit = suit
        self.id = card_id
        self.index = index


class Deck:
    def __init__(self):
        self._init_cards()

    def _init_cards(self):
        self.cards = []
        for i, suit in enumerate(["S", "H", "D", "C"]):
            for j, printed in enumerate(["A", "2", "3", "4", "5", "6", "7", "J", "Q", "K"]):
                card_id = suit + printed
                index = i * 10 + j
                value = j + 1
                self.cards.append(Card(value, suit, card_id, index))

    def get_card(self, index):
        return self.cards[index]