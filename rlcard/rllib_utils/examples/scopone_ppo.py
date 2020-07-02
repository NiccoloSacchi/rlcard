import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog
from ray.tune import register_env

from rlcard.rllib_utils.model import ParametricActionsModel
from rlcard.rllib_utils.rlcard_wrapper import RLCardWrapper

rlcard_env_id = 'scopone'

register_env("ParametricScopone", lambda _: RLCardWrapper(rlcard_env_id=rlcard_env_id))
ModelCatalog.register_custom_model("parametric_model_tf", ParametricActionsModel)

ray.init(num_cpus=4)

ppo_trainer_config = {
    "env": "ParametricScopone",
    "model": {
        "custom_model": "parametric_model_tf",
    },
}
trainer = PPOTrainer(config=ppo_trainer_config)
for i in range(5):
    res = trainer.train()
    print("Iteration {}. episode_reward_mean: {}".format(i, res['episode_reward_mean']))

print('Training finished, check the results in ~/ray_results/<dir>/')