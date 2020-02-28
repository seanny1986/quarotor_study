import matplotlib.pyplot as plt
import numpy as np
import pid
import modules as mod
import offline_agents as offag
import online_agents as onag
import training_loops as tl
import utilities as utils
import gym
import gym_aero
import os
import torch



##############################################################################################
########################## PATHWISE DERIVATIVE ALGS ##########################################
##############################################################################################
print()
print("Gym location: ")
print(gym.__file__)
print()

env = gym.make("HalfCheetah-v1")
state_dim = env.observation_space.shape[0]
hidden_dim = 64
action_dim = env.action_space.shape[0]

path = os.getcwd()
print("Working directory: ", path)
print("Observation space dim: ", env.observation_space.shape)
print("Action space dim: ", env.action_space.shape)

# DDPG
ou_noise = utils.OrnsteinUhlenbeckProcess(theta=0.15, mu=0., sigma=0.2, size=action_dim)
replay_memory = utils.ReplayMemory(1e6)
q_fn = mod.ValueNet(state_dim + action_dim, hidden_dim, 1, num_heads=1)
q_fn_targ = mod.ValueNet(state_dim + action_dim, hidden_dim, 1, num_heads=1)
beta = mod.DeterministicPolicy(state_dim, hidden_dim, action_dim)
agent = onag.DDPG(beta, q_fn, q_fn_targ, replay_memory, ou_noise)
pol_opt = torch.optim.Adam(beta.parameters(), lr=1e-4)
q_opt = torch.optim.Adam(q_fn.parameters(), lr=1e-4)
ddpg_data = tl.train_online(env, agent, pol_opt, q_opt, None, batch_size=128, iterations=500, log_interval=10)

# TD3
replay_memory = utils.ReplayMemory(1e6)
q_fn = mod.ValueNet(state_dim + action_dim, hidden_dim, 1, num_heads=2)
q_fn_targ = mod.ValueNet(state_dim + action_dim, hidden_dim, 1, num_heads=2)
beta = mod.DeterministicPolicy(state_dim, hidden_dim, action_dim)
agent = onag.TD3(beta, q_fn, q_fn_targ, replay_memory)
pol_opt = torch.optim.Adam(beta.parameters(), lr=1e-4)
q_opt = torch.optim.Adam(q_fn.parameters(), lr=1e-4)
td3_data = tl.train_online(env, agent, pol_opt, q_opt, None, batch_size=128, iterations=500, log_interval=10)

# SVG(0)
replay_memory = utils.ReplayMemory(1e6)
q_fn = mod.ValueNet(state_dim + action_dim, hidden_dim, 1, num_heads=1)
q_fn_targ = mod.ValueNet(state_dim + action_dim, hidden_dim, 1, num_heads=1)
beta = mod.SquashedGaussianPolicy(state_dim, hidden_dim, action_dim, pwd=True)
agent = onag.SVG(beta, q_fn, q_fn_targ, replay_memory)
pol_opt = torch.optim.Adam(beta.parameters(), lr=1e-4)
q_opt = torch.optim.Adam(q_fn.parameters(), lr=1e-4)
svg_data = tl.train_online(env, agent, pol_opt, q_opt, None, batch_size=128, iterations=500, log_interval=10)

# SAC
replay_memory = utils.ReplayMemory(1e6)
q_fn = mod.ValueNet(state_dim + action_dim, hidden_dim, 1, num_heads=1)
v_fn = mod.ValueNet(state_dim, hidden_dim, 1, num_heads=1)
v_fn_targ = mod.ValueNet(state_dim, hidden_dim, 1, num_heads=1)
beta = mod.SquashedGaussianPolicy(state_dim, hidden_dim, action_dim, pwd=True)
agent = onag.SAC(beta, q_fn, v_fn, v_fn_targ, replay_memory)
pol_opt = torch.optim.Adam(beta.parameters(), lr=1e-4)
q_opt = torch.optim.Adam(q_fn.parameters(), lr=1e-4)
v_opt = torch.optim.Adam(v_fn.parameters(), lr=1e-4)
sac_data = tl.train_online(env, agent, pol_opt, q_opt, v_opt, batch_size=128, iterations=500, log_interval=10)

epochs = np.arange(0, 510, 10)
plt.figure(figsize=(12,12))
plt.plot(epochs, np.array(ddpg_data))
plt.plot(epochs, np.array(td3_data))
plt.plot(epochs, np.array(svg_data))
plt.plot(epochs, np.array(sac_data))
plt.title("Online Algorithm Unit Tests on HalfCheetah-v1")
plt.xlabel("Epochs")
plt.ylabel("Reward")
plt.legend(["ddpg", "td3", "svg(0)", "sac"])
plt.savefig("online_unit_test.pdf")

