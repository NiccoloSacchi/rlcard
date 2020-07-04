
class ScoponePlayer:
    def __init__(self, player_id):
        self.player_id = player_id
        self.hand = set()
        self.captured = set()
        self.played = set()
        self.scope = 0

    def get_state(self):
        """Encode the state for the player in a dictionary
        :param table: cards on the table
        :type table: List[String]
        :param legal_actions: cards that are still in the hand of the player and which can be played
        :type legal_actions: List[int]
        :return: the encoded state of the player
        :rtype: dict(string: List[Int])
        """
        state = {}
        state["hand"] = self.hand
        state["captured"] = self.captured
        state["played"] = self.played
        state["scope"] = self.scope
        return state

    def give_cards(self, cards):
        for card in cards:
            self.hand.add(card)
