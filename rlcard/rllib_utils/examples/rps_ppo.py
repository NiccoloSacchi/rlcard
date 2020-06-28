import ray
from ray.rllib.agents.ppo import PPOTrainer

from rlcard.rllib_utils.model import ParametricActionsModel
from ray.rllib.models import ModelCatalog

from rlcard.rllib_utils.examples.envs.rps_env import RockPaperScissors
from ray.tune.registry import register_env


# Register env and model to be used by rllib
register_env("ParametricRPS", lambda _: RockPaperScissors())
ModelCatalog.register_custom_model("parametric_model_tf", ParametricActionsModel)

# Initialize ray
ray.init(num_cpus=4)

# Train the ParametricActionsModel on RockPaperScissors with PPO
ppo_trainer_config = {
    "env": "ParametricRPS",  # RockPaperScissors
    "model": {
        "custom_model": "parametric_model_tf",  # ParametricActionsModel,
    },
}
trainer = PPOTrainer(config=ppo_trainer_config)
for i in range(5):
    res = trainer.train()
    print("Iteration {}. episode_reward_mean: {}".format(i, res['episode_reward_mean']))
