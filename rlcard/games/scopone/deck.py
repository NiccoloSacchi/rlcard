class Card:
    def __init__(self, value, suit, card_id, index):
        self.value = value
        self.suit = suit
        self.id = card_id
        self.index = index
        self._compute_primiera_value()

    def _compute_primiera_value(self):
        if self.value == 7:
            self.primiera_value = 21
        if self.value == 6:
            self.primiera_value = 18
        if self.value == 1:
            self.primiera_value = 16
        if self.value == 5:
            self.primiera_value = 15
        if self.value == 4:
            self.primiera_value = 14
        if self.value == 3:
            self.primiera_value = 13
        if self.value == 2:
            self.primiera_value = 12
        else:
            self.primiera_value = 10


class Deck:
    SUITS = ["S", "H", "D", "C"]
    VALUES = ["A", "2", "3", "4", "5", "6", "7", "J", "Q", "K"]

    def __init__(self):
        self._init_cards()

    def _init_cards(self):
        self.cards = []
        for i, suit in enumerate(self.SUITS):
            for j, printed in enumerate(self.VALUES):
                card_id = suit + printed
                index = i * 10 + j
                value = j + 1
                self.cards.append(Card(value, suit, card_id, index))

    def get_card(self, index):
        return self.cards[index]