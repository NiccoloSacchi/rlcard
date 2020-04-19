import os
import rlcard
import torch
from rlcard.agents.reinforce_agent import ReinforceAgent
from rlcard.utils.utils import set_global_seed, tournament
from rlcard.utils.logger import Logger

episode_num = 10
evaluate_num = 10
evaluate_every = 1

env = rlcard.make("blackjack")
eval_env = rlcard.make("blackjack")

log_dir = './experiments/blackjack_reinforce_result/'

set_global_seed(42)

agent = ReinforceAgent(scope="reinforce_agent",
                       action_num=env.action_num,
                       state_shape=env.state_shape,
                       discount_factor=0.99,
                       learning_rate=1e-6,
                       device=None)
env.set_agents([agent])
eval_env.set_agents([agent])

logger = Logger(log_dir)

for episode in range(episode_num):
    trajectories, _ = env.run(is_training=True)

    for ts in trajectories[0]:
        agent.feed(ts)

    loss = agent.train
    logger.log(f"Loss: {loss}")

    if episode % evaluate_every == 0:
        logger.log_performance(env.timestep, tournament(eval_env, evaluate_num)[0])

logger.close_files()
logger.plot("REINFORCE")

# Save model
save_dir = 'models/blackjack_reinforce'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
torch.save(agent.get_state_dict(), os.path.join(save_dir, 'model.pth'))






