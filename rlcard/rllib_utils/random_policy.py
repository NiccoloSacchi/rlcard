from ray.rllib import Policy


def create_RandomPolicy(seed=0):

    # a hand-coded policy that acts at random in the env (doesn't learn)
    class RandomPolicy(Policy):
        """Hand-coded policy that returns random actions."""
        def __init__(self, observation_space, action_space, config):
            self.observation_space = observation_space
            self.action_space = action_space
            self.action_space.seed(seed)

        def compute_actions(
                self,
                obs_batch,
                state_batches,
                prev_action_batch=None,
                prev_reward_batch=None,
                info_batch=None,
                episodes=None,
                **kwargs
        ):
            """Compute actions on a batch of observations."""
            return [self.action_space.sample() for _ in obs_batch], [], {}

        def learn_on_batch(self, samples):
            """No learning."""
            pass

        def get_weights(self):
            pass

        def set_weights(self, weights):
            pass

    return RandomPolicy
