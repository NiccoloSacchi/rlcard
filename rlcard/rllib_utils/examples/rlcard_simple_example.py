import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.dqn import DQNTrainer

from rlcard.rllib_utils.model import ParametricActionsModel
from ray.rllib.models import ModelCatalog

from rlcard.rllib_utils.rlcard_wrapper import RLCardWrapper
from ray.tune.registry import register_env

# Decide which RLcard environment to use
# rlcard_env_id = 'blackjack'
# rlcard_env_id = 'doudizhu'
# rlcard_env_id = 'gin-rummy'
# rlcard_env_id = 'leduc-holdem'
# rlcard_env_id = 'limit-holdem'
# rlcard_env_id = 'mahjong'
# rlcard_env_id = 'no-limit-holdem'
# rlcard_env_id = 'simple-doudizhu'
rlcard_env_id = 'uno'

# Decide with which algorithm to train
# Trainer = DQNTrainer
Trainer = PPOTrainer

# Register env and model to be used by rllib
register_env(rlcard_env_id, lambda _: RLCardWrapper(rlcard_env_id=rlcard_env_id))
ModelCatalog.register_custom_model("parametric_model_tf", ParametricActionsModel)

# Initialize ray
ray.init(num_cpus=4)

# Train the ParametricActionsModel on rlcard_env_id with Trainer
trainer_config = {
    "env": rlcard_env_id,
    "model": {
        "custom_model": "parametric_model_tf",  # ParametricActionsModel,
    },
}

trainer = Trainer(config=trainer_config)
for i in range(5):
    res = trainer.train()
    print("Iteration {}. episode_reward_mean: {}".format(i, res['episode_reward_mean']))
    print(res)

print('Training finished, check the results in ~/ray_results/<dir>/')
