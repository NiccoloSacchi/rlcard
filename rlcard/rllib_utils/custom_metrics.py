"""Example of using RLlib's debug callbacks.
Here we use callbacks to track the average CartPole pole angle magnitude as a
custom metric.
"""

from typing import Dict
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.agents.callbacks import DefaultCallbacks


class PlayerScoreCallbacks(DefaultCallbacks):
    def on_episode_start(self, worker: RolloutWorker, base_env: BaseEnv,
                         policies: Dict[str, Policy],
                         episode: MultiAgentEpisode, **kwargs):
        # print("episode {} started".format(episode.episode_id))
        pass

    def on_episode_step(self, worker: RolloutWorker, base_env: BaseEnv,
                        episode: MultiAgentEpisode, **kwargs):
        pass

    def on_episode_end(self, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       **kwargs):
        # player_scores = base_env.rlcard_env.get_payoffs()
        player_scores_raw = episode.agent_rewards
        player_scores = {f"{p}_score": v for (p, _), v in player_scores_raw.items()}
        episode.custom_metrics.update(player_scores)
        print(dict(player_scores))

    def on_sample_end(self, worker: RolloutWorker, samples: SampleBatch,
                      **kwargs):
        # print("returned sample batch of size {}".format(samples.count))
        pass

    def on_train_result(self, trainer, result: dict, **kwargs):
        # print("trainer.train() result: {} -> {} episodes".format(
        #     trainer, result["episodes_this_iter"]))
        # you can mutate the result dict to add new fields to return
        result["callback_ok"] = True

    def on_postprocess_trajectory(
            self, worker: RolloutWorker, episode: MultiAgentEpisode,
            agent_id: str, policy_id: str, policies: Dict[str, Policy],
            postprocessed_batch: SampleBatch,
            original_batches: Dict[str, SampleBatch], **kwargs):
        # print("postprocessed {} steps".format(postprocessed_batch.count))
        # if "num_batches" not in episode.custom_metrics:
        #     episode.custom_metrics["num_batches"] = 0
        # episode.custom_metrics["num_batches"] += 1
        pass
