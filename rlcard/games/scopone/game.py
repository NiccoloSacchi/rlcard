from rlcard.games.scopone.deck import Deck
from rlcard.games.scopone.player import ScoponePlayer


class ScoponeGame:
    def __init__(self):
        self.num_players = 4
        self.num_rounds = 10
        self.current_round = 0
        self.current_player_id = 0
        self.table = set()
        self.players = [ScoponePlayer(i) for i in range(self.num_players)]
        self.deck = Deck()

    def step(self, action):
        """
        Executes the action for one player, consisting in moving a card from his hand to the table.
        :param action: the id of the card to play
        :type action: int
        :return:
        :rtype:
        """
        assert self.current_round < self.num_rounds

        player = self.players[self.current_player_id]
        card = self.deck.get_card(action)

        if card not in player.hand:
            raise ValueError("Action not allowed because the card is not in the player's hand")

        player.hand.remove(card)
        best_combination_on_the_table = self.get_best_combination(card)
        if best_combination_on_the_table:
            self.table.remove(card)
            player.captured.add(card)
            for c in best_combination_on_the_table:
                self.table.remove(c)
                player.captured.add(c)
        else:
            self.table.add(card)

        if self.current_player == self.num_players - 1:
            self.current_player = 0
        else:
            self.current_player += 1
        self.current_round += 1

    # TODO: make this more rigorous - e.g, then give priority to 6, 5, ...
    def get_best_combination(self, played_card):
        compatible_combinations = self.get_compatible_combinations(played_card)

        if not compatible_combinations:
            return []
        if len(compatible_combinations) == 1:
            return compatible_combinations[0]

        # heuristic 1: get the 7 of diamonds
        for combination in compatible_combinations:
            if any([card.value == 7 and card.suit == "D" for card in combination]):
                return combination

        # heuristic 2: get a 7
        for combination in compatible_combinations:
            if any([card.value == 7 for card in combination]):
                return combination

        # heuristic 3: get most diamonds
        diamonds_count = [len([card for card in combination if card.suit == "D"]) for combination in compatible_combinations]
        max_diamonds = max(diamonds_count)
        combinations_max_diamonds = [c for c, count in zip(compatible_combinations, diamonds_count) if count == max_diamonds]
        if len(combinations_max_diamonds) == 1:
            return combinations_max_diamonds[0]
        else:
            # heuristic 4: get most cards
            cards_count = [len(combination) for combination in combinations_max_diamonds]
            max_cards = max(cards_count)
            combinations_max_cards = [c for c, count in zip(compatible_combinations, cards_count) if count == max_cards]
            return combinations_max_cards[0]

    def get_compatible_combinations(self, played_card):
        table_card_list = [c for c in self.table]
        possible_combinations = self.get_combinations_lower_then(played_card.value, table_card_list)
        return [c for c in possible_combinations if sum([el.value for el in c]) == played_card.value]

    def get_combinations_lower_then(self, value, card_list):
        possible_cards = [c for c in card_list if c.value <= value]
        if not possible_cards:
            return []
        else:
            result = [[c] for c in possible_cards]
            for idx, c in enumerate(possible_cards):
                other_combinations = self.get_combinations_lower_then(value - c.value, possible_cards[idx + 1:])
                for combination in other_combinations:
                    result.append([c] + combination)
            return result

    # def compatible_cards(self, played_card):
    #     final_combinations_list = [[c] for c in self.table if c.value == played_card.value]
    #     tentative_combinations_list = [[c] for c in self.table if c.value < played_card.value]
    #     next_tentative_combinations_list = []
    #     while tentative_combinations_list:
    #         for combination in tentative_combinations_list:
    #             for card in self.table:
    #                 if card not in combination:
    #                     new_combination = [c for c in combination] + [card]
    #                     if sum(combination) + card.value < played_card.value:
    #                         next_tentative_combinations_list.append(new_combination)
    #                     elif sum(combination) == played_card.value:
    #                         final_combinations_list.append(new_combination)
    #         tentative_combinations_list = next_tentative_combinations_list


if __name__ == "__main__":
    game = ScoponeGame()
    deck = game.deck
    game.table = {deck.get_card(0), deck.get_card(3), deck.get_card(6), deck.get_card(12), deck.get_card(21)}
    played_card = deck.get_card(9)

    compatible_combinations = game.get_compatible_combinations(played_card)

    for combination in compatible_combinations:
        print([c.id for c in combination])



