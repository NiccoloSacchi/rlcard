import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy

from rlcard.rllib_utils.custom_metrics import PlayerScoreCallbacks
from rlcard.rllib_utils.random_policy import RandomPolicy

from rlcard.rllib_utils.model import ParametricActionsModel
from ray.rllib.models import ModelCatalog

from rlcard.rllib_utils.rlcard_wrapper import RLCardWrapper
from ray.tune.registry import register_env

import time

rlcard_env_id = 'scopone'
checkpoint_path = r"C:\Users\chiappal\ray_results\PPO_scopone_2020-07-07_23-25-17_training_2\checkpoint_5202\checkpoint-5202"
restore_checkpoint = True
num_iter = 10000
checkpoint_every = 200
eval_every = 10

# Register env and model to be used by rllib
scopone_env = RLCardWrapper(rlcard_env_id=rlcard_env_id)
register_env(rlcard_env_id, lambda _: scopone_env)
ModelCatalog.register_custom_model("parametric_model_tf", ParametricActionsModel)

# Initialize ray
ray.init(num_cpus=4, num_gpus=0)

# Define the policies
ppo_policy_config = {
    # "env": rlcard_env_id,
    "model": {
        "custom_model": "parametric_model_tf",
        "fcnet_hiddens": [256, 256, 256],
    },
}
policies = {
    "ppo_policy": (PPOTFPolicy,
                   scopone_env.observation_space,
                   scopone_env.action_space,
                   ppo_policy_config),
    "rand_policy": (RandomPolicy,
                    scopone_env.observation_space,
                    scopone_env.action_space,
                    {}),
}

# Instantiate the PPO trainer eval
trainer_eval = PPOTrainer(config={
    "env": rlcard_env_id,
    "multiagent": {
        "policies_to_train": ['ppo_policy'],
        "policies": policies,
        "policy_mapping_fn": lambda agent_id: "ppo_policy" if agent_id in ("player_1", "player_3") else "rand_policy",
    },
    # "num_gpus": 0,
    "callbacks": PlayerScoreCallbacks
})

trainer = PPOTrainer(config={
    "env": rlcard_env_id,
    "multiagent": {
        "policies_to_train": ['ppo_policy'],
        "policies": policies,
        "policy_mapping_fn": lambda agent_id: "ppo_policy",
    },
    # "num_gpus": 0,
    # "num_gpus_per_worker": 0,
    "callbacks": PlayerScoreCallbacks
})

if restore_checkpoint:
    trainer.restore(checkpoint_path)

start = time.time()

try:
    for i in range(num_iter):
        res = trainer.train()
        print("Iteration {}. policy result: {}".format(i, res))
        if i % eval_every == 0:
            trainer_eval.set_weights(trainer.get_weights(["ppo_policy"]))
            res = trainer_eval.train()
        if i % checkpoint_every == 0:
            trainer.save()
except:
    trainer.save()

stop = time.time()
train_duration = time.strftime('%H:%M:%S', time.gmtime(stop-start))
print('Training finished ({}), check the results in ~/ray_results/<dir>/'.format(train_duration))
