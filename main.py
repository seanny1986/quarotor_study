import matplotlib.pyplot as plt
import numpy as np
import modules as mod
import offline_agents as offag
import online_agents as onag
import training_loops as tl
import utilities as utils
import gym
import gym_aero as aero
import os
import torch
import csv
import settings as cfg

"""
Runs the main experiment loop
"""

def write_to_file(env_name, alg_name, af, data, batch_size, update_steps):
    # process data
    fp = path + "/data/" + env_name + "_" + alg_name + "_" + str(af) + ".csv"
    data = np.array(data).T
    mean = np.mean(data, axis=1, keepdims=True)
    std = np.std(data, axis=1, keepdims=True)
    data = np.append(data, mean, axis=1)
    data = np.append(data, std, axis=1)
    iterations = np.arange(0, iters+li, li).reshape(-1, 1)
    timesteps = batch_size * iterations.copy()
    updates = update_steps * iterations.copy()
    proc = np.append(iterations, timesteps, axis=1)
    proc = np.append(proc, updates, axis=1)
    full_data = np.append(proc, data, axis=1)
    rows = ["iterations", "timesteps", "gradient updates"] + ["reward " + str(i) for i in range((data.shape[1]-2))] + ["mean", "stdev"]
    with open(fp, "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(rows)
        for i in range(full_data.shape[0]):
            writer.writerow(full_data[i,:].tolist())

path = os.getcwd()
print()
print("Gym location: ")
print(gym.__file__)
print("Gym-Aero location: ")
print(aero.__file__)
print("Working directory: ", path)
print()

offline_batch_size = cfg.offline_batch_size
online_batch_size = cfg.online_batch_size
pol_lr = cfg.pol_lr
q_fn_lr = cfg.q_lr
v_fn_lr = cfg.v_lr
iters = cfg.training_iterations
hidden_dim = cfg.hidden_dim
offline_update_iters = cfg.offline_update_iters
warmup = cfg.warmup
online_steps = cfg.online_steps
li = cfg.log_freq
memory_size = cfg.memory_size

def run(envs, action_freqs, reps):
    for e in envs:
        env = gym.make(e)
        state_dim = env.observation_space.shape[0]        
        action_dim = env.action_space.shape[0]
        for af in action_freqs:
            
            print("Running A2C in {} at action selection frequency {}".format(e, af))
            env.set_action_frequency(af)
            dat = []
            for i in range(reps):
                # A2C
                v_fn = mod.ValueNet(state_dim, hidden_dim, 1, num_heads=1)
                beta = mod.IndependentGaussianPolicy(state_dim, hidden_dim, action_dim)
                agent = offag.A2C(beta, v_fn)
                pol_opt = torch.optim.Adam(beta.parameters(), lr=pol_lr)
                v_opt = torch.optim.Adam(v_fn.parameters(), lr=v_fn_lr)
                reinforce_data = tl.train_offline(env, agent, pol_opt, None, v_opt, batch_size=offline_batch_size, iterations=iters, log_interval=li)
                dat.append(reinforce_data)
            write_to_file(e, "a2c", af, dat, offline_batch_size, offline_update_iters)
            print("A2C data written.")
            print()
            print()

            print("Running PPO in {} at action selection frequency {}".format(e, af))
            dat = []
            for i in range(reps):
                # PPO
                v_fn = mod.ValueNet(state_dim, hidden_dim, 1, num_heads=1)
                beta = mod.IndependentGaussianPolicy(state_dim, hidden_dim, action_dim)
                agent = offag.PPO(beta, v_fn)
                pol_opt = torch.optim.Adam(beta.parameters(), lr=pol_lr)
                v_opt = torch.optim.Adam(v_fn.parameters(), lr=v_fn_lr)
                ppo_data = tl.train_offline(env, agent, pol_opt, None, v_opt, batch_size=offline_batch_size, iterations=iters, log_interval=li)
                dat.append(ppo_data)
            write_to_file(e, "ppo", af, dat, offline_batch_size, offline_update_iters)
            print("PPO data written.")
            print()
            print()
            
            print("Running TRPO in {} at action selection frequency {}".format(e, af))
            dat = []
            for i in range(reps):
                # TRPO
                v_fn = mod.ValueNet(state_dim, hidden_dim, 1, num_heads=1)
                beta = mod.IndependentGaussianPolicy(state_dim, hidden_dim, action_dim)
                pi = mod.IndependentGaussianPolicy(state_dim, hidden_dim, action_dim)
                fvp = utils.gaussian_fvp
                agent = offag.TRPO(beta, pi, v_fn, fvp)
                pol_opt = torch.optim.Adam(beta.parameters(), lr=pol_lr)
                v_opt = torch.optim.Adam(v_fn.parameters(), lr=v_fn_lr)
                trpo_data = tl.train_offline(env, agent, pol_opt, None, v_opt, batch_size=offline_batch_size, iterations=iters, log_interval=li)
                dat.append(trpo_data)
            write_to_file(e, "trpo", af, dat, offline_batch_size, offline_update_iters)
            print("TRPO data written.")
            print()
            print()
            
            
            print("Running DDPG in {} at action selection frequency {}".format(e, af))
            dat = []
            for i in range(reps):
                # DDPG
                ou_noise = utils.OrnsteinUhlenbeckProcess(theta=cfg.ou_theta, mu=cfg.ou_mu, sigma=cfg.ou_sigma, size=action_dim)
                replay_memory = utils.ReplayMemory(memory_size)
                q_fn = mod.ValueNet(state_dim + action_dim, hidden_dim, 1, num_heads=1)
                q_fn_targ = mod.ValueNet(state_dim + action_dim, hidden_dim, 1, num_heads=1)
                beta = mod.DeterministicPolicy(state_dim, hidden_dim, action_dim)
                agent = onag.DDPG(beta, q_fn, q_fn_targ, replay_memory, ou_noise)
                pol_opt = torch.optim.Adam(beta.parameters(), lr=pol_lr)
                q_opt = torch.optim.Adam(q_fn.parameters(), lr=q_fn_lr)
                ddpg_data = tl.train_online(env, agent, pol_opt, q_opt, None, batch_size=online_batch_size, iterations=iters, log_interval=li)
                dat.append(ddpg_data)
            write_to_file(e, "ddpg", af, dat, online_steps, online_steps)
            print("DDPG data written.")
            print()
            print()
            
            print("Running TD3 in {} at action selection frequency {}".format(e, af))
            dat = []
            for i in range(reps):
                # TD3
                replay_memory = utils.ReplayMemory(memory_size)
                q_fn = mod.ValueNet(state_dim + action_dim, hidden_dim, 1, num_heads=2)
                q_fn_targ = mod.ValueNet(state_dim + action_dim, hidden_dim, 1, num_heads=2)
                beta = mod.DeterministicPolicy(state_dim, hidden_dim, action_dim)
                agent = onag.TD3(beta, q_fn, q_fn_targ, replay_memory)
                pol_opt = torch.optim.Adam(beta.parameters(), lr=pol_lr)
                q_opt = torch.optim.Adam(q_fn.parameters(), lr=q_fn_lr)
                td3_data = tl.train_online(env, agent, pol_opt, q_opt, None, batch_size=online_batch_size, iterations=iters, log_interval=li)
                dat.append(td3_data)
            write_to_file(e, "td3", af, dat, online_steps, online_steps)
            print("TD3 data written.")
            print()
            print()

            print("Running SVG in {} at action selection frequency {}".format(e, af))
            dat = []
            for i in range(reps):
                # SVG(0)
                replay_memory = utils.ReplayMemory(memory_size)
                q_fn = mod.ValueNet(state_dim + action_dim, hidden_dim, 1, num_heads=1)
                q_fn_targ = mod.ValueNet(state_dim + action_dim, hidden_dim, 1, num_heads=1)
                beta = mod.SquashedGaussianPolicy(state_dim, hidden_dim, action_dim, pwd=True)
                agent = onag.SVG(beta, q_fn, q_fn_targ, replay_memory)
                pol_opt = torch.optim.Adam(beta.parameters(), lr=pol_lr)
                q_opt = torch.optim.Adam(q_fn.parameters(), lr=q_fn_lr)
                svg_data = tl.train_online(env, agent, pol_opt, q_opt, None, batch_size=online_batch_size, iterations=iters, log_interval=li)
                dat.append(svg_data)
            write_to_file(e, "svg", af, dat, online_steps, online_steps)
            print("SVG data written.")
            print()
            print()

            print("Running SAC in {} at action selection frequency {}".format(e, af))
            dat = []
            for i in range(reps):
                # SAC
                replay_memory = utils.ReplayMemory(memory_size)
                q_fn = mod.ValueNet(state_dim + action_dim, hidden_dim, 1, num_heads=1)
                v_fn = mod.ValueNet(state_dim, hidden_dim, 1, num_heads=1)
                v_fn_targ = mod.ValueNet(state_dim, hidden_dim, 1, num_heads=1)
                beta = mod.SquashedGaussianPolicy(state_dim, hidden_dim, action_dim, pwd=True)
                agent = onag.SAC(beta, q_fn, v_fn, v_fn_targ, replay_memory)
                pol_opt = torch.optim.Adam(beta.parameters(), lr=pol_lr)
                q_opt = torch.optim.Adam(q_fn.parameters(), lr=q_fn_lr)
                v_opt = torch.optim.Adam(v_fn.parameters(), lr=v_fn_lr)
                sac_data = tl.train_online(env, agent, pol_opt, q_opt, v_opt, batch_size=online_batch_size, iterations=iters, log_interval=li)
                dat.append(sac_data)
            write_to_file(e, "sac", af, dat, online_steps, online_steps)
            print("SAC data written.")
            print()

if __name__ == "__main__":
    # execute only if run as a script
    #action_freqs = [0.005, 0.02, 0.035, 0.05, 0.065, 0.08, 0.095, 0.11, 0.125]
    action_freqs = [0.05]
    reps = 3

    envs = ["Hover-v0", "RandomWaypointFH-v0", "RandomWaypointNH-v0", "Land-v0"]
    run(envs, action_freqs, reps)

    pid_envs = ["PIDHover-v0", "PIDRandomWaypointFH-v0", "PIDRandomWaypointNH-v0", "PIDLand-v0"]
    run(pid_envs, action_freqs, reps)