import random


class Card:
    def __init__(self, value, suit, card_id, index):
        self.value = value
        self.suit = suit
        self.id = card_id
        self.index = index
        self._compute_primiera_value()

    def __eq__(self, other):
        return self.id == other.id

    def __str__(self):
        return f"id: {self.id}, index: {self.index}"

    def __hash__(self):
        return self.index

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
        self.cards = self._create_cards()

    def _create_cards(self):
        cards = []
        for i, suit in enumerate(self.SUITS):
            for j, printed in enumerate(self.VALUES):
                card_id = suit + printed
                index = i * 10 + j
                value = j + 1
                cards.append(Card(value, suit, card_id, index))
        return cards

    def get_card(self, index):
        return self.cards[index]

    def distribute_cards(self, seed=None):
        random.seed(seed)
        cards_to_distribute = self._create_cards()
        random.shuffle(cards_to_distribute)
        return [cards_to_distribute[i: i+10] for i in range(0, 40, 10)]
