import torch
import numpy as np
import gym
import gym_aero as ga
import utilities as utils
import settings as cfg

device = cfg.device
render = cfg.render

def test(env, agent):
    state = torch.Tensor(env.reset()).to(device)
    done = False
    r = 0.
    while not done:
        if render:
            env.render()
        action, _, _ = agent.select_action(state)
        next_state, reward, done, _ = env.step(action.cpu().data.numpy())
        r += reward
        next_state = torch.Tensor(next_state).to(device)
        state = next_state
    return r


def rollout(env, agent, batch_size):
    s_, a_, ns_, r_, lp_, masks = [], [], [], [], [], []
    T = 0
    while T < batch_size:
        t = 0
        state = torch.Tensor(env.reset()).to(device)
        done = False
        while not done:
            action, log_prob, entropy = agent.select_action(state)
            next_state, reward, done, info = env.step(action.cpu().data.numpy())
            reward = torch.Tensor([reward]).to(device)
            next_state = torch.Tensor(next_state).to(device)
            s_.append(state)
            a_.append(action)
            ns_.append(next_state)
            r_.append(reward)
            lp_.append(log_prob)
            masks.append(torch.Tensor([not done]).to(device))
            state = next_state
            t += 1
        T += t
    trajectory = {
                "states" : s_,
                "actions" : a_,
                "rewards" : r_,
                "next_states" : ns_,
                "masks" : masks,
                "log_probs" : lp_,
                }
    return trajectory


def train_offline(env, agent, pol_opt, q_opt, v_opt, batch_size=1024, iterations=500, log_interval=10, t_runs=10):
    test_rew_best = np.mean([test(env, agent) for _ in range(t_runs)])
    data = []
    data.append(test_rew_best)
    print()
    print("Iterations: ", 0)
    print("Time steps: ", 0)
    print("Reward: ", test_rew_best)
    print()
    for ep in range(1, iterations+1):
        trajectory = rollout(env, agent, batch_size)
        rewards = torch.stack(trajectory["rewards"]).to(device)
        masks = torch.stack(trajectory["masks"]).to(device)
        R = torch.sum(rewards)/(masks.size(0) - torch.sum(masks))
        print("Batch {},  mean reward: {:.4f}".format(ep, R))        
        agent.update(pol_opt, q_opt, v_opt, trajectory)
        if ep % log_interval == 0:
            test_rew = np.mean([test(env, agent) for _ in range(t_runs)])
            data.append(test_rew)
            print("------------------------------")
            print("Iterations: ", ep)
            print("Time steps: ", batch_size*ep)
            print("Reward: ", test_rew)
            print("------------------------------")
    return data


def train_online(env, agent, pol_opt, q_opt, v_opt, steps=1000, warmup=1000, batch_size=128, iterations=500, log_interval=10, t_runs=10):
    # warmup to add transitions to replay memory
    T = 0
    while T < warmup:
        state = torch.Tensor(env.reset()).to(device)
        done = False
        t = 0
        while not done:
            action, _, _ = agent.select_action(state)
            action = action.detach()
            next_state, reward, done, info = env.step(action.cpu().data.numpy())
            mask = 1 if t == env._max_episode_steps else float(not done)
            mask = torch.Tensor([mask]).to(device)
            reward = torch.Tensor([reward]).to(device)
            next_state = torch.Tensor(next_state).to(device)
            agent.replay_memory.push(state, action, next_state, reward, mask)
            state = next_state
            t += 1
        T += t
    
    print("Warmup finished, training agent.")
    
    test_rew_best = np.mean([test(env, agent) for _ in range(t_runs)])
    data = []
    data.append(test_rew_best)
    print()
    print("Iterations: ", 0)
    print("Time steps: ", 0)
    print("Reward: ", test_rew_best)
    print()
    
    # run training loop
    for ep in range(1, int(iterations + 1)):
        T = 0
        R = 0
        n = 1
        while T < steps:
            done = False
            state = torch.Tensor(env.reset()).to(device)
            r = 0
            t = 0
            while not done:
                action, _, _ = agent.select_action(state)
                action = action.detach()
                next_state, reward, done, info = env.step(action.cpu().data.numpy())
                r += reward
                mask = 1 if t == env._max_episode_steps else float(not done)
                mask = torch.Tensor([mask]).to(device)
                reward = torch.Tensor([reward]).to(device)
                next_state = torch.Tensor(next_state).to(device)
                agent.replay_memory.push(state, action, next_state, reward, mask)
                state = next_state
                agent.update(pol_opt, q_opt, v_opt, batch_size)
                t += 1
            T += t
            R = (R*(n-1)+r)/n
            n += 1
        print("Batch {},  mean reward: {:.4f}".format(ep, R))        
        if ep % log_interval == 0:
            test_rew = np.mean([test(env, agent) for _ in range(t_runs)])
            print("------------------------------")
            print("Iterations: ", ep)
            print("Time steps: ", ep * steps)
            print("Test Reward: ", test_rew)
            print("------------------------------")
            data.append(test_rew)
    return data