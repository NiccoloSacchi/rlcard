import random
from ray.rllib import Policy


class RandomPolicy(Policy):
    """Hand-coded policy that returns random actions."""

    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)
        # self.preprocessor = get_preprocessor(observation_space)(observation_space)
        self.action_space.seed(0)

    def compute_actions(
            self,
            obs_batch,
            state_batches=None,
            prev_action_batch=None,
            prev_reward_batch=None,
            info_batch=None,
            episodes=None,
            **kwargs
    ):
        """Compute actions on a batch of observations."""
        mask_len = self.action_space.n
        action_mask_list = [obs[: mask_len] for obs in obs_batch]
        actions_list = []
        for mask in action_mask_list:
            allowed_actions = [idx for idx, allowed in enumerate(mask) if allowed]
            actions_list.append(random.choice(allowed_actions))
        return actions_list, [], {}

    def learn_on_batch(self, samples):
        """No learning."""
        pass

    def get_weights(self):
        pass

    def set_weights(self, weights):
        pass

    def compute_log_likelihoods(self, actions, obs_batch, state_batches=None, prev_action_batch=None,
                                prev_reward_batch=None):
        pass

    def compute_gradients(self, postprocessed_batch):
        pass

    def apply_gradients(self, gradients):
        pass

    def export_model(self, export_dir):
        pass

    def export_checkpoint(self, export_dir):
        pass

    def import_model_from_h5(self, import_file):
        pass
