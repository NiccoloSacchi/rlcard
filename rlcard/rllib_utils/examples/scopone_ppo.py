import ray
from ray.rllib.agents.ppo import PPOTrainer, PPOTFPolicy
from ray.rllib.models import ModelCatalog
from ray.tune import register_env

from rlcard.rllib_utils.model import ParametricActionsModel
from rlcard.rllib_utils.rlcard_wrapper import RLCardWrapper

rlcard_env_id = 'scopone'
checkpoint_path = r"C:\Users\chiappal\ray_results\PPO_ParametricScopone_2020-07-03_17-32-1762krv6zp\checkpoint_20\checkpoint-20"
restore_checkpoint = True

scopone_env = RLCardWrapper(rlcard_env_id=rlcard_env_id)
register_env("ParametricScopone", lambda _: scopone_env)
ModelCatalog.register_custom_model("parametric_model_tf", ParametricActionsModel)

ray.init(num_cpus=4)

ppo_policy_config = {
    "env": "ParametricScopone",
    "model": {
        "custom_model": "parametric_model_tf",
        "fcnet_hiddens": [512, 256, 128],
        "use_lstm": False,
    },
}

policies = {
    "ppo_policy_albi": (PPOTFPolicy,
                        scopone_env.observation_space,
                        scopone_env.action_space,
                        ppo_policy_config),
    "ppo_policy_nico": (PPOTFPolicy,
                        scopone_env.observation_space,
                        scopone_env.action_space,
                        ppo_policy_config)
}

ppo_trainer_config = {
    "env": "ParametricScopone",
    "multiagent": {
        "policies_to_train": ["ppo_policy_albi"],
        "policies": policies,
        "policy_mapping_fn": lambda agent_id: "ppo_policy_albi" if agent_id in ("player_1", "player_3") else "ppo_policy_nico",
    },
    "observation_filter": "NoFilter"
}

trainer = PPOTrainer(config=ppo_trainer_config)
if restore_checkpoint:
    trainer.restore(checkpoint_path)

trainer.get_policy("ppo_policy_albi").model.base_model.summary()
trainer.get_policy("ppo_policy_nico").model.base_model.summary()

for i in range(200):
    res = trainer.train()
    print("Iteration {}. policy_reward_mean: {}".format(i, res['policy_reward_mean']))
    trainer.save()

print('Training finished, check the results in ~/ray_results/<dir>/')