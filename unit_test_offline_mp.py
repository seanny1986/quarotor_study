import matplotlib.pyplot as plt
import numpy as np
import modules as mod
import offline_agents as offag
import online_agents as onag
import training_loops as tl
import utilities as utils
import gym
import os
import torch
import mp_envs as mp
from multiprocessing_env import SubprocVecEnv


##############################################################################################
########################## REINFORCE-BASED ALGS ##############################################
##############################################################################################
print()
print("Gym location: ")
print(gym.__file__)
print()

num_envs = 32

envs = [mp.make_half_cheetah() for i in range(num_envs)]
tenv = gym.make("HalfCheetah-v1")
envs = SubprocVecEnv(envs)
state_dim = tenv.observation_space.shape[0]
hidden_dim = 64
action_dim = tenv.action_space.shape[0]
bsize = 2048


path = os.getcwd()
print("Working directory: ", path)
print("Observation space dim: ", tenv.observation_space.shape)
print("Action space dim: ", tenv.action_space.shape)

"""
# A2C
v_fn = mod.ValueNet(state_dim, hidden_dim, 1, num_heads=1)
beta = mod.IndependentGaussianPolicy(state_dim, hidden_dim, action_dim)
agent = offag.A2Cmp(beta, v_fn)
pol_opt = torch.optim.Adam(beta.parameters(), lr=1e-4)
v_opt = torch.optim.Adam(v_fn.parameters(), lr=1e-4)
a2c_data = tl.train_offline_mp(envs, tenv, agent, pol_opt, None, v_opt, batch_size=bsize, iterations=500)
"""

# PPO
v_fn = mod.ValueNet(state_dim, hidden_dim, 1, num_heads=1)
beta = mod.IndependentGaussianPolicy(state_dim, hidden_dim, action_dim)
agent = offag.PPOmp(beta, v_fn)
pol_opt = torch.optim.Adam(beta.parameters(), lr=3e-4)
v_opt = torch.optim.Adam(v_fn.parameters(), lr=3e-4)
ppo_data = tl.train_offline_mp(envs, tenv, agent, pol_opt, None, v_opt, batch_size=bsize, iterations=500)

# TRPO
v_fn = mod.ValueNet(state_dim, hidden_dim, 1, num_heads=1)
beta = mod.IndependentGaussianPolicy(state_dim, hidden_dim, action_dim)
pi = mod.IndependentGaussianPolicy(state_dim, hidden_dim, action_dim)
fvp = utils.gaussian_fvp
agent = offag.TRPOmp(beta, pi, v_fn, fvp)
pol_opt = torch.optim.Adam(beta.parameters(), lr=1e-4)
v_opt = torch.optim.Adam(v_fn.parameters(), lr=1e-4)
trpo_data = tl.train_offline_mp(envs, tenv, agent, pol_opt, None, v_opt, batch_size=bsize, iterations=500)

epochs = np.arange(0, 510, 10)
plt.figure(figsize=(12,12))
plt.plot(epochs, np.array(a2c_data))
plt.plot(epochs, np.array(ppo_data))
plt.plot(epochs, np.array(trpo_data))
plt.title("Offline Algorithm Unit Tests on HalfCheetah-v1")
plt.xlabel("Epochs")
plt.ylabel("Reward")
plt.legend(["a2c", "ppo", "trpo"])
plt.savefig("offline_unit_test.pdf")