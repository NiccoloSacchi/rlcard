import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
from ray.rllib.agents.dqn.dqn_tf_policy import DQNTFPolicy
from rlcard.rllib_utils.random_policy import RandomPolicy

from rlcard.rllib_utils.model import ParametricActionsModel
from ray.rllib.models import ModelCatalog

from rlcard.rllib_utils.rlcard_wrapper import RLCardWrapper
from ray.tune.registry import register_env

import time

# Decide which RLcard environment to use
# rlcard_env_id = 'blackjack'
# rlcard_env_id = 'doudizhu'
# rlcard_env_id = 'gin-rummy'
rlcard_env_id = 'leduc-holdem'
# rlcard_env_id = 'limit-holdem'
# rlcard_env_id = 'mahjong'
# rlcard_env_id = 'no-limit-holdem'
# rlcard_env_id = 'simple-doudizhu'
# rlcard_env_id = 'uno'


# Register env and model to be used by rllib
RLCardWrapped = lambda _: RLCardWrapper(rlcard_env_id=rlcard_env_id)
register_env(rlcard_env_id, RLCardWrapped)
ModelCatalog.register_custom_model("parametric_model_tf", ParametricActionsModel)

# Initialize ray
# ray.init(num_cpus=4, num_gpus=1)
ray.init(num_cpus=4)
#
# Define the policies
ppo_trainer_config = {
    # "env": rlcard_env_id,
    "model": {
        "custom_model": "parametric_model_tf",
    },
}
env_tmp = RLCardWrapped(None)
policies = {
    "ppo_policy_1": (PPOTFPolicy,
                     env_tmp.observation_space,
                     env_tmp.action_space,
                     ppo_trainer_config),
    "rand_policy": (RandomPolicy,
                    env_tmp.observation_space,
                    env_tmp.action_space,
                    {}),
}

# Instantiate the PPO trainer eval
trainer_eval = PPOTrainer(config={
    "env": rlcard_env_id,
    "multiagent": {
        "policies_to_train": ['ppo_policy_1'],
        "policies": policies,
        "policy_mapping_fn": lambda agent_id: "ppo_policy_1" if agent_id == "player_1" else "rand_policy",
    },
    # "num_gpus": 0.5,
})

trainer = PPOTrainer(config={
    "env": rlcard_env_id,
    "multiagent": {
        "policies_to_train": ['ppo_policy_1'],
        "policies": policies,
        "policy_mapping_fn": lambda agent_id: "ppo_policy_1",
    },
    # "num_gpus": 0.5,
    # "num_gpus_per_worker": 0,
})

start = time.time()
for i in range(10):
    trainer.set_weights(trainer_eval.get_weights(["ppo_policy_1"]))
    trainer.train()

    trainer_eval.set_weights(trainer.get_weights(["ppo_policy_1"]))
    res = trainer_eval.train()

    policy_rewards = sorted(['{}: {}'.format(k, v) for k, v in res['policy_reward_mean'].items()])
    print("Iteration {}. policy_reward_mean: {}".format(i, policy_rewards))
stop = time.time()
train_duration = time.strftime('%H:%M:%S', time.gmtime(stop-start))
print('Training finished ({}), check the results in ~/ray_results/<dir>/'.format(train_duration))
