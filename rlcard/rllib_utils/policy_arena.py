import ray
from ray.rllib.agents.a3c.a3c_tf_policy import A3CTFPolicy
from ray.rllib.agents.dqn import DQNTFPolicy
from ray.rllib.agents.ppo import PPOTFPolicy

from rlcard.rllib_utils.random_policy import RandomPolicy
from rlcard.rllib_utils.trainer import RLTrainer


def get_policy_weights(policy, checkpoint, env_id, agent_to_policy, policy_to_class):
    trainer_cls, config = RLTrainer(
        experiment_name="temp",
        rlcard_env_id=env_id,
        agent_to_policy=agent_to_policy,
        policy_to_class=policy_to_class,
        randomize_agents_eval=[],
        resources={}
    ).trainer_to_config.popitem()
    trainer = trainer_cls(config)
    trainer.restore(checkpoint)
    policy_weights = trainer.get_weights(policy)
    return policy_weights


def create_arena_trainer(env_id, first_policy, second_policy, policy_to_class):
    trainer_cls, config = RLTrainer(
        experiment_name="arena",
        rlcard_env_id=env_id,
        agent_to_policy={
            "player_1": first_policy,
            "player_2": second_policy,
            "player_3": first_policy,
            "player_4": second_policy
        },
        policy_to_class=policy_to_class,
        randomize_agents_eval=[],
        resources={}
    ).trainer_to_config.popitem()
    return trainer_cls(config)


if __name__ == "__main__":

    num_iter = 1000

    env_id = "limit-holdem"

    policy_0 = "a2c_policy"
    policy_to_class_0 = {
        'ppo_policy_1': PPOTFPolicy,
        'dqn_policy_1': DQNTFPolicy,
        'a2c_policy': A3CTFPolicy,
        'random_policy': RandomPolicy
    }
    agent_to_policy_0 = {
        'player_1': policy_0,
        'player_2': policy_0,
        'player_3': policy_0,
        'player_4': policy_0,
    }

    checkpoint_0 = r"C:\Users\chiappal\ray_results\limit-holdem\A2C_limit-holdem_0_2020-07-11_10-20-24ewvwlolq\checkpoint_400\checkpoint-400"

    policy_1 = "dqn_policy_1"
    policy_to_class_1 = {
        'ppo_policy_1': PPOTFPolicy,
        'dqn_policy_1': DQNTFPolicy,
        'random_policy': RandomPolicy
    }
    agent_to_policy_1 = {
        'player_1': policy_1,
        'player_2': policy_1,
        'player_3': policy_1,
        'player_4': policy_1,
    }

    checkpoint_1 = r"C:\Users\chiappal\ray_results\limit-holdem\DQN_limit-holdem_0_2020-07-11_17-33-563qv8j_bf\checkpoint_516\checkpoint-516"

    # Initialize ray
    ray.init(num_cpus=4, num_gpus=0)

    policy_weights_0 = get_policy_weights(policy_0, checkpoint_0, env_id, agent_to_policy_0, policy_to_class_0)
    policy_weights_1 = get_policy_weights(policy_1, checkpoint_1, env_id, agent_to_policy_1, policy_to_class_1)

    policy_to_class = {
        'ppo_policy_1': PPOTFPolicy,
        'dqn_policy_1': DQNTFPolicy,
        'a2c_policy': A3CTFPolicy,
        'random_policy': RandomPolicy
    }

    trainer = create_arena_trainer(env_id, policy_0, policy_1, policy_to_class)

    for i in range(num_iter):
        trainer.set_weights(policy_weights_0)
        trainer.set_weights(policy_weights_1)
        res = trainer.train()
        print("Iteration {}. policy result: {}".format(i, res))


