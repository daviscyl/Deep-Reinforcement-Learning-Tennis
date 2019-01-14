import torch
import numpy as np
from unityagents import UnityEnvironment
from ddpg_agent import Agent


# instantiate the environment and agent
env = UnityEnvironment(file_name='Tennis.app')
agent = Agent(state_size=24, action_size=2, random_seed=2)

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# get the number of agents in the environment
env_info = env.reset(train_mode=False)[brain_name]
num_agents = len(env_info.agents)


agent.actor_local.load_state_dict(torch.load('best_actor.pth'))
agent.critic_local.load_state_dict(torch.load('best_critic.pth'))

env_info = env.reset(train_mode=False)[brain_name]     # reset the environment
states = env_info.vector_observations                  # get the current state (for each agent)
score = np.zeros(num_agents)                           # initialize the score (for each agent)
while True:
    actions = agent.act(states, noise_level=0)         # select an action (for each agent)
    env_info = env.step(actions)[brain_name]           # send all actions to tne environment
    next_states = env_info.vector_observations         # get next state (for each agent)
    rewards = env_info.rewards                         # get reward (for each agent)
    dones = env_info.local_done                        # see if episode finished
    score += rewards                                   # update the score (for each agent)
    states = next_states                               # roll over states to next time step
    if np.any(dones):                                  # exit loop if episode finished
        break
print('Total score (averaged over agents) this episode: {}'.format(np.max(score)))

env.close()
