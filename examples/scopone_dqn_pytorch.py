import rlcard
import torch


# Make environment
from rlcard.agents.dqn_agent_pytorch import DQNAgent
from rlcard.utils.logger import Logger
from rlcard.utils.utils import set_global_seed, tournament

env = rlcard.make("scopone")
eval_env = rlcard.make('leduc-holdem')

# Set the iterations numbers and how frequently we evaluate the performance
evaluate_every = 100
evaluate_num = 1000
episode_num = 100000

# The intial memory size
memory_init_size = 1000

# Train the agent every X steps
train_every = 1

# The paths for saving the logs and learning curves
log_dir = './experiments/scopone_result/'

# Set a global seed
set_global_seed(0)

agents = [DQNAgent(scope=f'dqn_{i}',
                 action_num=env.action_num,
                 replay_memory_init_size=memory_init_size,
                 train_every=train_every,
                 state_shape=env.state_shape,
                 mlp_layers=[128, 128],
                 device=torch.device('cpu')) for i in range(env.player_num)]
env.set_agents(agents)
eval_env.set_agents(agents)

logger = Logger(log_dir)

for episode in range(episode_num):
    # Generate data from the environment
    trajectories, _ = env.run(is_training=True)

    # Feed transitions into agent memory, and train the agent
    for agent_idx, traj in enumerate(trajectories):
        for ts in traj:
            agents[agent_idx].feed(ts)

    # Evaluate the performance.
    if episode % evaluate_every == 0:
        logger.log_performance(env.timestep, tournament(eval_env, evaluate_num)[0])

