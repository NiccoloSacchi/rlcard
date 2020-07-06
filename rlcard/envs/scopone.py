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
        # + 2 because of table and hand, plus captured cards and played cards per each player
        self.state_shape = [2 * self.game.get_player_num() + 2, self.game.get_action_num()]

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
        available_card_idx = [card.index for card in player_hand]
        extracted_state["legal_actions"] = available_card_idx

        allowed_actions_vec = np.zeros(self.game.get_action_num(), dtype=int)
        allowed_actions_vec[available_card_idx] = 1
        cards_on_table_idx = [card.index for card in state["table"]]
        cards_on_table_vec = np.zeros(self.game.get_action_num(), dtype=int)
        cards_on_table_vec[cards_on_table_idx] = 1

        captured_cards_vec = np.zeros((self.game.get_player_num(), self.game.get_action_num()), dtype=int)
        played_cards_vec = np.zeros((self.game.get_player_num(), self.game.get_action_num()), dtype=int)
        for player_id in range(self.game.get_player_num()):
            captured_cards = state[f"player_{player_id}"]["captured"]
            captured_cards_idx = [c.index for c in captured_cards]
            captured_cards_vec[player_id, captured_cards_idx] = 1
            played_cards = state[f"player_{player_id}"]["played"]
            played_cards_idx = [c.index for c in played_cards]
            played_cards_vec[player_id, played_cards_idx] = 1

        extracted_state["obs"] = np.concatenate((allowed_actions_vec.reshape(1, 40),
                                                 cards_on_table_vec.reshape(1, 40),
                                                 captured_cards_vec,
                                                 played_cards_vec),
                                                axis=0)

        return extracted_state

    def get_payoffs(self):
        return self.game.get_payoffs()

    def get_perfect_information(self):
        return self.game.get_state()

    def _decode_action(self, action_id):
        return self.game.deck.get_card(action_id)

    def _get_legal_actions(self):
        return self.game.get_legal_actions()
