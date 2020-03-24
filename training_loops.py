import torch
import numpy as np
import gym
import gym_aero as ga
import utilities as utils
import settings as cfg

device = cfg.device
render = cfg.render
save_video = cfg.save_video

def test(env, agent, running_state=None):
    if running_state is not None:
        state = torch.Tensor(running_state(env.reset())).to(device)
    else:
        state = torch.Tensor(env.reset()).to(device)
    done = False
    r = 0.
    while not done:
        if render:
            try:
                env.render(video=save_video)
            except:
                env.render()
        action, _, _ = agent.select_action(state)
        next_state, reward, done, _ = env.step(action.cpu().data.numpy())
        r += reward
        if running_state is not None:
            next_state = torch.Tensor(running_state(next_state)).to(device)
        else:
            next_state = torch.Tensor(next_state).to(device)
        state = next_state
    return r


def rollout(env, agent, batch_size, running_state=None):
    s_, a_, ns_, r_, lp_, masks = [], [], [], [], [], []
    T = 0
    while T < batch_size:
        t = 0
        if running_state is not None:
            state = torch.Tensor(running_state(env.reset())).to(device)
        else:
            state = torch.Tensor(env.reset()).to(device)
        done = False
        while not done:
            action, log_prob, entropy = agent.select_action(state)
            next_state, reward, done, info = env.step(action.cpu().data.numpy())
            reward = torch.Tensor([reward]).to(device)
            if running_state is not None:
                next_state = torch.Tensor(running_state(next_state)).to(device)
            else:
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


def rollout_mp(states, envs, agent, batch_size):
    s_, a_, ns_, r_, lp_, masks = [], [], [], [], [], []
    t = 0
    while t < batch_size:
        actions, log_probs, entropies = agent.select_action(states)
        next_states, rewards, dones, infos = envs.step(actions.cpu().data.numpy())
        rewards = torch.Tensor(rewards).to(device)
        ms = torch.Tensor([not d for d in dones]).to(device)
        next_states = torch.Tensor(next_states).to(device)
        s_.append(states)
        a_.append(actions)
        ns_.append(next_states)
        r_.append(rewards)
        lp_.append(log_probs)
        masks.append(ms)
        states = next_states
        t += 1
    trajectory = {
                "states" : s_,
                "actions" : a_,
                "rewards" : r_,
                "next_states" : ns_,
                "masks" : masks,
                "log_probs" : lp_,
                }
    return states, trajectory


def train_offline(env, agent, pol_opt, q_opt, v_opt, batch_size=1024, iterations=500, log_interval=10, t_runs=10):
    global save_video
    running_state = None#utils.ZFilter((agent.beta.input_dim,), clip=5)
    test_rew_best = np.mean([test(env, agent, running_state) for _ in range(t_runs)])
    if save_video: 
        try:
            fname = agent.name + "-" + env.squashed + "-" + str(env.ctrl_dt) + "-" + env.name + "-episode-" + str(0)
            env.save_video(env.name, fname)
        except AttributeError:
            save_video = False
    data = []
    data.append(test_rew_best)
    print()
    print("Iterations: ", 0)
    print("Time steps: ", 0)
    print("Reward: ", test_rew_best)
    print()
    
    for ep in range(1, iterations+1):
        trajectory = rollout(env, agent, batch_size, running_state)
        rewards = torch.stack(trajectory["rewards"]).to(device)
        masks = torch.stack(trajectory["masks"]).to(device)
        R = torch.sum(rewards)/(masks.size(0) - torch.sum(masks))
        print("Batch {},  mean reward: {:.4f}".format(ep, R))        
        agent.update(pol_opt, q_opt, v_opt, trajectory)
        if ep % log_interval == 0:
            test_rew = np.mean([test(env, agent, running_state) for _ in range(t_runs)])
            if save_video: 
                fname = agent.name + "-" + env.squashed + "-" + str(env.ctrl_dt) + "-" + env.name + "-episode-" + str(ep)
                env.save_video(env.name, fname)
            data.append(test_rew)
            print("------------------------------")
            print("Iterations: ", ep)
            print("Time steps: ", batch_size*ep)
            print("Reward: ", test_rew)
            print("------------------------------")
    return data


def train_offline_mp(envs, tenv, agent, pol_opt, q_opt, v_opt, batch_size=1024, iterations=500, log_interval=10, t_runs=10):
    global save_video
    running_state = None#utils.ZFilter((agent.beta.input_dim,), clip=5)
    test_rew_best = np.mean([test(tenv, agent, running_state) for _ in range(t_runs)])
    if save_video: 
        try:
            fname = agent.name + "-" + tenv.squashed + "-" + str(tenv.ctrl_dt) + "-" + tenv.name + "-episode-" + str(0)
            tenv.save_video(tenv.name, fname)
        except AttributeError:
            save_video = False
    data = []
    data.append(test_rew_best)
    print()
    print("Iterations: ", 0)
    print("Time steps: ", 0)
    print("Reward: ", test_rew_best)
    print()
    states = torch.Tensor(envs.reset()).to(device)
    for ep in range(1, iterations+1):
        states, trajectory = rollout_mp(states, envs, agent, batch_size)
        agent.update(pol_opt, q_opt, v_opt, trajectory)
        if ep % log_interval == 0:
            test_rew = np.mean([test(tenv, agent, running_state) for _ in range(t_runs)])
            if save_video: 
                fname = agent.name + "-" + tenv.squashed + "-" + str(tenv.ctrl_dt) + "-" + tenv.name + "-episode-" + str(ep)
                tenv.save_video(tenv.name, fname)
            data.append(test_rew)
            print("------------------------------")
            print("Iterations: ", ep)
            print("Time steps: ", batch_size*ep)
            print("Reward: ", test_rew)
            print("------------------------------")
    return data


def train_online(env, agent, pol_opt, q_opt, v_opt, steps=1000, warmup=1000, batch_size=128, iterations=500, log_interval=10, t_runs=10):
    # warmup to add transitions to replay memory
    global save_video
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
    
    test_rew_best = np.mean([test(env, agent, None) for _ in range(t_runs)])
    if save_video: 
        try:
            fname = agent.name + "-" + env.squashed + "-" + str(env.ctrl_dt) + "-" + env.name + "-episode-" + str(0)
            env.save_video(env.name, fname)
        except AttributeError:
            save_video = False
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
                agent.update(pol_opt, q_opt, v_opt, batch_size)
                state = next_state
                t += 1
            T += t
            R = (R*(n-1)+r)/n
            n += 1
        print("Batch {},  mean reward: {:.4f}".format(ep, R))        
        if ep % log_interval == 0:
            test_rew = np.mean([test(env, agent, None) for _ in range(t_runs)])
            if save_video:
                fname = agent.name + "-" + env.squashed + "-" + str(env.ctrl_dt) + "-" + env.name + "-episode-" + str(ep)
                env.save_video(env.name, fname)
            print("------------------------------")
            print("Iterations: ", ep)
            print("Time steps: ", ep * steps)
            print("Test Reward: ", test_rew)
            print("------------------------------")
            data.append(test_rew)
    return data