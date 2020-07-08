from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
from ray.rllib.agents.dqn.dqn_tf_policy import DQNTFPolicy
from rlcard.rllib_utils.random_policy import RandomPolicy
from rlcard.rllib_utils.trainer import RLTrainer

# rlcard_env_id = 'blackjack'
# rlcard_env_id = 'doudizhu'
# rlcard_env_id = 'gin-rummy'
# rlcard_env_id = 'leduc-holdem'
rlcard_env_id = 'limit-holdem'
# rlcard_env_id = 'mahjong'
# rlcard_env_id = 'no-limit-holdem'
# rlcard_env_id = 'simple-doudizhu'
# rlcard_env_id = 'uno'
# rlcard_env_id = 'scopone'

for rlcard_env_id in [
    'blackjack',
    'doudizhu',
    # 'gin-rummy',
    'leduc-holdem',
    'limit-holdem',
    'mahjong',
    'no-limit-holdem',
    # 'simple-doudizhu',
    'uno',
    'scopone',
]:
    try:
        trainer = RLTrainer(
            experiment_name=rlcard_env_id,
            rlcard_env_id=rlcard_env_id,
            agent_to_policy={
                'player_1': 'ppo_policy_1',
                'player_2': 'ppo_policy_1',
                'player_3': 'ppo_policy_1',
                'player_4': 'ppo_policy_1',
                # random_policy
                # 'player_2': 'dqn_policy_1',
            },
            policy_to_class={
                'ppo_policy_1': PPOTFPolicy,
                'dqn_policy_1': DQNTFPolicy,
                'random_policy': RandomPolicy
            },
            randomize_agents_eval=['player_2', 'player_4'],
            resources={
                'num_workers': 8,
                # 'num_gpus': 1,
                # 'num_gpus_per_worker': 0.25,
                # 'num_cpus_per_worker': 1,
            },
        )

        trainer.train(
            # If these values are surpassed then the training stops.
            stop={
                # 'episodes_total': 50000,
                # "timesteps_total": 20000,
                'time_total_s': 60*60,
                # "training_iteration": 60,
                # "policy_reward_mean/ppo_policy_1": 1.5,
                # "episode_reward_mean": 2.90,

            },
            # Continue training from parameters previously trained. TODO: how to merge the results in a single graph?
            # restore="~/ray_results/leduc-holdem/PPO_leduc-holdem_0_2020-07-05_22-22-02h1g897qx/checkpoint_9/checkpoint-9",
        )
    except Exception as e:
        print('!!! Failed training {}: {} !!!'.format(rlcard_env_id, e))
