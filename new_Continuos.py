#!/usr/bin/env python
# coding: utf-8

# # Continuous Control
# 
# ---
# 
# In this notebook, you will learn how to use the Unity ML-Agents environment for the second project of the [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) program.
# 
# ### 1. Start the Environment
# 
# We begin by importing the necessary packages.  If the code cell below returns an error, please revisit the project instructions to double-check that you have installed [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md) and [NumPy](http://www.numpy.org/).

# In[1]:


from unityagents import UnityEnvironment
import numpy as np


# Next, we will start the environment!  **_Before running the code cell below_**, change the `file_name` parameter to match the location of the Unity environment that you downloaded.
# 
# - **Mac**: `"path/to/Reacher.app"`
# - **Windows** (x86): `"path/to/Reacher_Windows_x86/Reacher.exe"`
# - **Windows** (x86_64): `"path/to/Reacher_Windows_x86_64/Reacher.exe"`
# - **Linux** (x86): `"path/to/Reacher_Linux/Reacher.x86"`
# - **Linux** (x86_64): `"path/to/Reacher_Linux/Reacher.x86_64"`
# - **Linux** (x86, headless): `"path/to/Reacher_Linux_NoVis/Reacher.x86"`
# - **Linux** (x86_64, headless): `"path/to/Reacher_Linux_NoVis/Reacher.x86_64"`
# 
# For instance, if you are using a Mac, then you downloaded `Reacher.app`.  If this file is in the same folder as the notebook, then the line below should appear as follows:
# ```
# env = UnityEnvironment(file_name="Reacher.app")
# ```

# In[2]:


env = UnityEnvironment(file_name='Reacher_20.app')


# Environments contain **_brains_** which are responsible for deciding the actions of their associated agents. Here we check for the first brain available, and set it as the default brain we will be controlling from Python.

# In[3]:


# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]


# ### 2. Examine the State and Action Spaces
# 
# In this environment, a double-jointed arm can move to target locations. A reward of `+0.1` is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.
# 
# The observation space consists of `33` variables corresponding to position, rotation, velocity, and angular velocities of the arm.  Each action is a vector with four numbers, corresponding to torque applicable to two joints.  Every entry in the action vector must be a number between `-1` and `1`.
# 
# Run the code cell below to print some information about the environment.

# In[4]:


# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])


# ### 3. Take Random Actions in the Environment, Pre-training
# 
# In the next code cell, you will learn how to use the Python API to control the agent and receive feedback from the environment.
# 
# Once this cell is executed, you will watch the agent's performance, if it selects an action at random with each time step.  A window should pop up that allows you to observe the agent, as it moves through the environment.  
# 
# Of course, as part of the project, you'll have to change the code so that the agent is able to use its experience to gradually choose better actions when interacting with the environment!

# In[5]:


env_info = env.reset(train_mode=False)[brain_name]     # reset the environment    
states = env_info.vector_observations                  # get the current state (for each agent)
scores = np.zeros(num_agents)                          # initialize the score (for each agent)
while True:
    actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
    actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
    env_info = env.step(actions)[brain_name]           # send all actions to tne environment
    next_states = env_info.vector_observations         # get next state (for each agent)
    rewards = env_info.rewards                         # get reward (for each agent)
    dones = env_info.local_done                        # see if episode finished
    scores += env_info.rewards                         # update the score (for each agent)
    states = next_states                               # roll over states to next time step
    if np.any(dones):                                  # exit loop if episode finished
        break
print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))


# When finished, you can close the environment.

# In[6]:


env.close()


# ### 4. Training the Agents

# In[5]:


from Agent import DDPG_Agent
from collections import deque
# import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np
import torch
import time

# sns.set()
# get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


agent = DDPG_Agent(state_size=state_size, action_size=action_size, random_seed=8)

def DDPG(n_episodes=1000, max_t=2000, print_every=10):
    """
    DDPG 
    
    Parameters
    ==========
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        
    """
    # list containing scores from each episode
    scores_global = []      
    # last 100 scores
    scores_deque = deque(maxlen=100)
    
    start_time = time.time()
    
    for i_episode in range(1, n_episodes+1):
        # reset the environment
        env_info = env.reset(train_mode=True)[brain_name]
        # obtain the current state for each agent
        states = env_info.vector_observations
        # initialize the score (for each agent)
        scores = np.zeros(num_agents)
        agent.reset()
                
        for timestep in range(max_t):
            # select an action
            actions = agent.act(states, add_noise=True)
            # send all actions to the environment
            env_info = env.step(actions)[brain_name]        
            # get the next states for each agent
            next_states = env_info.vector_observations
            # obtain the rewards for each agent
            rewards = env_info.rewards
            # see whether or not the episode is finished
            dones = env_info.local_done
            agent.step(states, actions, rewards, next_states, dones, timestep)
            # roll over to the next timestep
            states = next_states 
            # update the scores for each agent
            scores += rewards
            
            if np.any(dones):
                break 
                        
        # average the scores of each agent
        score = np.mean(scores)
        # save most recent score        
        scores_deque.append(score) 
        # save most recent score
        scores_global.append(score)             
        
        # print out some statistics
        print('Episode:{}, Score: {:.2f}, Max_score: {:.2f}, Min_score: {:.2f}'.format(
              i_episode, score, np.max(scores), np.min(scores)))
        
        if i_episode % print_every == 0 or (len(scores_deque) == 100 and np.mean(scores_deque) >= 30) :
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            duration = (int)(time.time() - start_time) 
            print('Episodes {}, AVG Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))

        
        if len(scores_deque) == 100 and np.mean(scores_deque) >= 30:  
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            torch.save(agent.actor_local.state_dict(), 'solved_checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'solved_checkpoint_critic.pth')
            break
            
    np.save('scores.npy', scores_global)        
    return scores_global


# In[7]:


scores = DDPG()


# In[11]:


# fig = plt.figure(figsize=[12,8])
# plt.plot(np.arange(1, len(scores)+1), scores)
# plt.ylabel('Score', fontsize=14)
# plt.xlabel('Episode', fontsize=14)
# plt.title('Plot of Rewards', fontsize=16)
# plt.show()


# ### 5. Observe the Trained Agents Perform!

# In[7]:


# New_Agent = DDPG_Agent(33, 4, 8)


# # In[8]:


# def load(New_Agent, actor_file, critic_file):
#     New_Agent.actor_local.load_state_dict(torch.load(actor_file))
#     New_Agent.actor_target.load_state_dict(torch.load(actor_file))
#     New_Agent.critic_local.load_state_dict(torch.load(critic_file))
#     New_Agent.critic_target.load_state_dict(torch.load(critic_file))
    
# load(New_Agent, 'checkpoint_actor.pth', 'checkpoint_critic.pth')   


# # In[9]:


# def play(agent, episodes=5):
#     action_size=4
#     num_agents=20
#     for i_episode in range(episodes):
#         env_info = env.reset(train_mode=False)[brain_name]     # reset the environment    
#         states = env_info.vector_observations                  # get the current state (for each agent)
#         scores = np.zeros(num_agents)                          # initialize the score (for each agent)
#         while True:
#             actions = agent.act(states, add_noise=False)       # all actions between -1 and 1
#             env_info = env.step(actions)[brain_name]           # send all actions to tne environment
#             next_states = env_info.vector_observations         # get next state (for each agent)
#             rewards = env_info.rewards                         # get reward (for each agent)
#             dones = env_info.local_done                        # see if episode finished
#             scores += env_info.rewards                         # update the score (for each agent)
#             states = next_states                               # roll over states to next time step
#             if np.any(dones):                                  # exit loop if episode finished
#                 break
#             #break
#         print('Episode: {} Average Score (over agents): {}'.format(i_episode, np.mean(scores)))


# # In[10]:


# play(New_Agent, 3)


# In[ ]:




