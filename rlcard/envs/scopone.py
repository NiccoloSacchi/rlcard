import numpy as np
from rlcard.envs.env import Env
from rlcard.games.scopone.game import ScoponeGame


class ScoponeEnv(Env):
    def __init__(self, config):
        """
        Initialize the environment for Scopone
        """
        self.game = ScoponeGame()
        super().__init__(config)
        self.actions = list(range(self.game.get_action_num()))
        # + 2 because of table and hand, plus captured cards per each player
        self.state_shape = [self.game.get_player_num() + 2, self.game.get_action_num()]

    def _load_model(self):
        pass

    def _extract_state(self, state):
        """Extract the state representation for the agent which is now required to choose an action (the one
        corresponding to the current player)

        :param state: game state and each of the players' state
        :type state: dict
        :return: the extracted state
        :rtype: dict
        """
        extracted_state = {}
        player_id = state["current_player"]
        player_hand = state[f"player_{player_id}"]["hand"]
        available_card_ids = [card.id for card in player_hand]
        extracted_state["allowed_actions"] = available_card_ids

        allowed_actions_vec = np.zeros(self.game.get_action_num())
        allowed_actions_vec[available_card_ids] = 1
        cards_on_table_id = [card.id for card in state["table"]]
        cards_on_table_vec = np.zeros(self.game.get_action_num())
        cards_on_table_vec[cards_on_table_id] = 1

        captured_cards_vec = np.zeros(self.game.get_player_num(), self.game.get_action_num())
        for player_id in self.game.get_player_num():
            captured_cards = state[f"player_{player_id}"]["captured"]
            captured_cards_ids = [c.id for c in captured_cards]
            captured_cards_vec[player_id, captured_cards_ids] = 1

        extracted_state["tabular_state"] = np.concatenate((cards_on_table_vec, allowed_actions_vec, captured_cards_vec),
                                                          axis=0)

    def get_payoffs(self):
        self.game.get_payoffs()

    def get_perfect_information(self):
        self.game.get_state()

    def _decode_action(self, action_id):
        return self.game.deck.get_card(action_id)

    def _get_legal_actions(self):
        self.game.get_legal_actions()
