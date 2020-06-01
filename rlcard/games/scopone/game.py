from collections import Set

from rlcard.games.scopone.deck import Deck, Card
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
        self.last_player_capturing_id = None

    def get_state(self):
        state = {}
        state["table"] = self.table
        state["current_player"] = self.current_player_id
        state["current_round"] = self.current_round
        for idx in range(self.num_players):
            state[f"player_{idx}"] = self.players[idx].get_state()
        return state

    def is_over(self):
        is_last_round = (self.current_round == self.num_rounds - 1)
        is_last_player = (self.current_player_id == self.num_players - 1)
        return is_last_round and is_last_player

    def get_payoffs(self):
        captured_0_2 = self.players[0].captured.union(self.players[2].captured)
        scope_0_2 = self.players[0].scope + self.players[2].scope

        captured_1_3 = self.players[1].captured.union(self.players[3].captured)
        scope_1_3 = self.players[1].scope + self.players[3].scope

        return self._compute_payoff(captured_0_2, scope_0_2, captured_1_3, scope_1_3)

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
        best_combination_on_the_table = self._get_best_combination(card)
        if best_combination_on_the_table:
            self.last_player_capturing_id = self.current_player_id
            self.table.remove(card)
            player.captured.add(card)
            for c in best_combination_on_the_table:
                self.table.remove(c)
                player.captured.add(c)
        else:
            self.table.add(card)

        if self.is_over():
            last_player_capturing = self.players[self.last_player_capturing_id]
            for card in self.table:
                last_player_capturing.captured.add(card)

        if self.current_player_id == self.num_players - 1:
            self.current_player_id = 0
        else:
            self.current_player_id += 1
        self.current_round += 1

    # TODO: make this more rigorous - e.g, then give priority to 6, 5, ...
    def _get_best_combination(self, played_card):
        compatible_combinations = self._get_compatible_combinations(played_card)

        if not compatible_combinations:
            return []
        if len(compatible_combinations) == 1:
            return compatible_combinations[0]

        # heuristic 1: get the 7 of diamonds
        combinations_with_7_D = [combination for combination in compatible_combinations if
                                 any([card.value == 7 and card.suit == "D" for card in combination])]
        if combinations_with_7_D:
            best_combinations = combinations_with_7_D
            if len(best_combinations) == 1:
                return best_combinations[0]
        else:
            best_combinations = compatible_combinations

        # heuristic 2: get a 7
        combinations_with_7 = [combination for combination in best_combinations if
                             any([card.value == 7  for card in combination])]
        if combinations_with_7:
            best_combinations = combinations_with_7
            if len(best_combinations) == 1:
                return best_combinations[0]

        # heuristic 3: get most diamonds
        diamonds_count = [len([card for card in combination if card.suit == "D"]) for combination in best_combinations]
        max_diamonds = max(diamonds_count)
        combinations_max_diamonds = [c for c, count in zip(best_combinations, diamonds_count) if count == max_diamonds]
        if len(combinations_max_diamonds) == 1:
            return combinations_max_diamonds[0]
        else:
            # heuristic 4: get most cards
            cards_count = [len(combination) for combination in combinations_max_diamonds]
            max_cards = max(cards_count)
            combinations_max_cards = [c for c, count in zip(compatible_combinations, cards_count) if count == max_cards]
            return combinations_max_cards[0]

    def _get_compatible_combinations(self, played_card):
        table_card_list = [c for c in self.table]
        possible_combinations = self._get_combinations_lower_then(played_card.value, table_card_list)
        return [c for c in possible_combinations if sum([el.value for el in c]) == played_card.value]

    def _get_combinations_lower_then(self, value, card_list):
        possible_cards = [c for c in card_list if c.value <= value]
        if not possible_cards:
            return []
        else:
            result = [[c] for c in possible_cards]
            for idx, c in enumerate(possible_cards):
                other_combinations = self._get_combinations_lower_then(value - c.value, possible_cards[idx + 1:])
                for combination in other_combinations:
                    result.append([c] + combination)
            return result

    def _compute_payoff(self, captured_0: Set[Card], scope_0: int, captured_1: Set[Card], scope_1: int):
        count_0 = scope_0
        count_1 = scope_1

        # settebello
        if any([c.value == 7 and c.suit == "D" for c in captured_0]):
            count_0 += 1
        else:
            assert any([c.value == 7 and c.suit == "D" for c in captured_1])
            count_1 += 1

        # carte
        assert len(captured_0) + len(captured_1) == 40
        if len(captured_0) > 20:
            count_0 += 1
        elif len(captured_1) > 20:
            count_1 += 1

        # ori
        diamonds_count_0 = len([c for c in captured_0 if c.suit == "D"])
        diamonds_count_1 = len([c for c in captured_1 if c.suit == "D"])
        assert diamonds_count_0 + diamonds_count_1 == 10
        if diamonds_count_0 > 5:
            count_0 += 1
        elif diamonds_count_1 > 5:
            count_1 += 1

        # primiera
        primiera_sum_0 = 0
        primiera_sum_1 = 0
        for suit in Deck.SUITS:
            best_val_0 = max([c.primiera_value for c in captured_0 if c.suit == suit])
            primiera_sum_0 += best_val_0
            best_val_1 = max([c.primiera_value for c in captured_1 if c.suit == suit])
            primiera_sum_1 += best_val_1
        if primiera_sum_0 > primiera_sum_1:
            count_0 += 1
        elif primiera_sum_1 > primiera_sum_0:
            count_1 += 1

        return count_0, count_1

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

    compatible_combinations = game._get_compatible_combinations(played_card)

    for combination in compatible_combinations:
        print([c.id for c in combination])

    print("Best combination:")
    print([c.id for c in game._get_best_combination(played_card)])



