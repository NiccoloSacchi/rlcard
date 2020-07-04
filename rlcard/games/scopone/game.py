from rlcard.games.scopone.deck import Deck
from rlcard.games.scopone.player import ScoponePlayer


class ScoponeGame:
    player_starting = 0

    def __init__(self):
        self.num_players = 4
        self.num_rounds = 10
        self.current_round = None
        self.current_player_id = None
        self.table = None
        self.players = None
        self.deck = Deck()
        self.last_player_capturing_id = None
        self.first_player_this_game = None

    def get_player_num(self):
        return 4

    def get_action_num(self):
        return 40

    def init_game(self):
        self.current_player_id = self.player_starting
        self.first_player_this_game = self.current_player_id
        self.player_starting = (self.player_starting + 1) % self.num_players
        self.current_round = 0
        self.table = set()
        self.players = [ScoponePlayer(i) for i in range(self.num_players)]
        for player, cards in zip(self.players, self.deck.distribute_cards()):
            player.give_cards(cards)
        self.last_player_capturing_id = None
        return self.get_state(), self.current_player_id

    def get_state(self, player_id=None):
        if player_id is None:
            player_id = self.current_player_id
        state = {}
        state["table"] = self.table
        state["current_player"] = player_id
        state["current_round"] = self.current_round
        for idx in range(self.num_players):
            state[f"player_{idx}"] = self.players[idx].get_state()
        return state

    def get_legal_actions(self):
        current_player_hand = self.players[self.current_player_id].hand
        return [card.id for card in current_player_hand]

    def is_over(self):
        last_round_is_over = (self.current_round == self.num_rounds)
        last_player_has_played = (self.current_player_id == self.first_player_this_game)
        return last_round_is_over and last_player_has_played

    def get_payoffs(self):
        captured_0_2 = self.players[0].captured.union(self.players[2].captured)
        scope_0_2 = self.players[0].scope + self.players[2].scope

        captured_1_3 = self.players[1].captured.union(self.players[3].captured)
        scope_1_3 = self.players[1].scope + self.players[3].scope

        score_team_0, score_team_1 = self._compute_payoff(captured_0_2, scope_0_2, captured_1_3, scope_1_3)

        adv_team_0 = score_team_0 - score_team_1
        return adv_team_0, -adv_team_0, adv_team_0, -adv_team_0

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
        card = action

        if card not in player.hand:
            raise ValueError("Action not allowed because the card is not in the player's hand")

        player.hand.remove(card)
        player.played.add(card)
        # print(f"Player {self.current_player_id} with hand {[c.id for c in player.hand]} played the card {card.id}")
        best_combination_on_the_table = self._get_best_combination(card)
        if best_combination_on_the_table:
            self.last_player_capturing_id = self.current_player_id
            player.captured.add(card)
            for c in best_combination_on_the_table:
                self.table.remove(c)
                player.captured.add(c)
                if not self.table:
                    player.scope += 1
        else:
            self.table.add(card)
        # print(f"Cards on the table after play: {[c.id  for c in self.table]}")

        if self.current_player_id == (self.first_player_this_game + self.num_players - 1) % self.num_players:
            self.current_round += 1
            # print(f"=========== Round {self.current_round} completed ============")
        self.current_player_id = (self.current_player_id + 1) % self.num_players

        if self.is_over():
            last_player_capturing = self.players[self.last_player_capturing_id]
            # print(f"Giving the remaining cards to player {last_player_capturing.player_id}")
            for card in self.table:
                last_player_capturing.captured.add(card)
                self.table = set()
            assert all([len(p.played) == 10 for p in self.players])
            assert all([len(p.hand) == 0 for p in self.players])
        return self.get_state(), self.current_player_id

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
        return combinations_max_diamonds[0]

    def _get_compatible_combinations(self, played_card):
        table_card_list = [c for c in self.table]
        possible_combinations = self._get_combinations_lower_then(played_card.value, table_card_list)
        correct_sum_combinations = [c for c in possible_combinations if
                                    sum([el.value for el in c]) == played_card.value]
        if not correct_sum_combinations:
            return []
        else:
            shortest_combination_len = min([len(c) for c in correct_sum_combinations])
            return [c for c in correct_sum_combinations if len(c) == shortest_combination_len]

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

    def _compute_payoff(self, captured_0, scope_0, captured_1, scope_1):
        count_0 = scope_0
        count_1 = scope_1

        # settebello
        if any([c.value == 7 and c.suit == "D" for c in captured_0]):
            assert all([c.value != 7 or c.suit != "D" for c in captured_1])
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
            primiera_value_suit_0 = [c.primiera_value for c in captured_0 if c.suit == suit]
            if primiera_value_suit_0:
                primiera_sum_0 += max(primiera_value_suit_0)
            primiera_value_suit_1 = [c.primiera_value for c in captured_1 if c.suit == suit]
            if primiera_value_suit_1:
                primiera_sum_1 += max(primiera_value_suit_1)
        if primiera_sum_0 > primiera_sum_1:
            count_0 += 1
        elif primiera_sum_1 > primiera_sum_0:
            count_1 += 1

        return count_0, count_1


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



