import torch
import torch.nn as nn
import utilities as utils
import settings as cfg

device = cfg.device
iters = cfg.online_update_iters
gamma = cfg.online_gamma
alpha = cfg.sac_alpha
ddpg_sigma = cfg.ddpg_sigma
td3_sigma = cfg.td3_sigma
c = cfg.c

class DDPG(nn.Module):
    """
    Simplified implementation of DDPG, as outlined in Lillicrap et. al 2015. We use
    hyperparameters that have been shown to work in other implementations, though
    we don't implement a target actor (we do have a target critic). The reason for
    this is to cut down on code, and because, from experience, we have seen little
    (if any) change in performance as a result of this modification. More modern
    online methods such as SAC typically don't use a target actor.

    Unlike the original paper's implementation, we use 3 critic updates for every 
    policy update. The reason for this is that it seems to work better for us, and
    has become fairly standard since the release of TD3. Accordingly, we apply the
    same update schedule for *all* of the online methods below.
    """

    def __init__(self, beta, q_fn, q_fn_targ, replay_memory, noise):
        super(DDPG, self).__init__()
        self.beta = beta
        self.q_fn = q_fn
        self.q_fn_targ = q_fn_targ
        self.replay_memory = replay_memory
        self.noise = noise
        self.name = "DDPG"
        if self.noise is not None: self.noise.reset()
        if q_fn_targ is not None: utils.hard_update(self.q_fn_targ, self.q_fn)
    
    def select_action(self, x, noise=True):
        action = self.beta(x)
        if noise:
            eps = ddpg_sigma * torch.Tensor(self.noise.sample()).to(device)
            return action + eps, None, None
        else:
            return action, None, None
        
    def update(self, pol_opt, q_opt, v_opt, batch_size):
        for _ in range(iters):
            states, actions, rewards, next_states, masks = self.replay_memory.sample_and_split(batch_size)
            next_actions, _, _ = self.select_action(next_states)
            next_state_actions = torch.cat([next_states, next_actions], dim=-1)
            q_targ = rewards + masks * gamma * masks * self.q_fn_targ.get_min_value(next_state_actions)
            state_actions = torch.cat([states, actions], dim=-1)
            q_values = self.q_fn(state_actions)
            q_fn_loss = torch.mean(torch.cat([(q - q_targ.detach()) ** 2 for q in q_values], dim=-1))
            q_opt.zero_grad()
            q_fn_loss.backward()
            q_opt.step()
        states, _, _, _, _ = self.replay_memory.sample_and_split(batch_size)
        new_actions, _, _ = self.select_action(states, noise=False)
        state_actions = torch.cat([states, new_actions], dim=-1)
        policy_loss = -torch.mean(self.q_fn(state_actions)[0])
        pol_opt.zero_grad()
        policy_loss.backward()
        pol_opt.step()
        utils.soft_update(self.q_fn_targ, self.q_fn)
        return policy_loss.item(), q_fn_loss.item()
        
        
class TD3(DDPG):
    """
    Simple implementation of TD3, as outlined in []. As with the DDPG implementation,
    we don't use a target actor, only a target critic. This is common in more modern
    methods (e.g. SAC), which is why we felt justified making this change. We keep the
    eps ~ N(0, sigma) noise injection that TD3 uses in the original paper, rather than
    the OU noise that DDPG typically uses.
    """

    def __init__(self, beta, q_fn, q_fn_targ, replay_memory):
        super(TD3, self).__init__(beta, q_fn, None, replay_memory, None)
        self.name = "TD3"
        self.q_fn_targ = q_fn_targ
        utils.hard_update(self.q_fn_targ, self.q_fn)

    def select_action(self, x, noise=True):
        action = self.beta(x)
        if noise:
            eps = torch.normal(mean=0, std=td3_sigma*torch.ones(action.size()))
            eps = torch.clamp(eps, -c, c)
            return action + eps, None, None
        else:
            return action, None, None

    def update(self, pol_opt, q_opt, v_opt, batch_size):
        for _ in range(iters):
            states, actions, rewards, next_states, masks = self.replay_memory.sample_and_split(batch_size)
            next_actions, _, _ = self.select_action(next_states)
            next_state_actions = torch.cat([next_states, next_actions], dim=-1)
            q_targ = rewards + masks * gamma * masks * self.q_fn_targ.get_min_value(next_state_actions)
            state_actions = torch.cat([states, actions], dim=-1)
            q_values = self.q_fn(state_actions)
            q_fn_loss = torch.mean(torch.cat([(q - q_targ.detach()) ** 2 for q in q_values], dim=-1))
            q_opt.zero_grad()
            q_fn_loss.backward()
            q_opt.step()
        states, _, _, _, _ = self.replay_memory.sample_and_split(batch_size)
        new_actions, _, _ = self.select_action(states)
        state_actions = torch.cat([states, new_actions], dim=-1)
        policy_loss = -torch.mean(self.q_fn(state_actions)[0])
        pol_opt.zero_grad()
        policy_loss.backward()
        pol_opt.step()
        utils.soft_update(self.q_fn_targ, self.q_fn)
        return policy_loss.item(), q_fn_loss.item()

    
class SVG(DDPG):
    """
    Simple implementation of SVG(0), as outlined in Heess et. al, 2017. This algorithm 
    is very similar to DDPG, except that it parameterizes a stochastic policy, and uses the 
    reparametrization trick to push the derivative back through to the policy. Conceptually,
    this algorithm is very similar to SAC; whereas SAC minimizes the KL-divergence between
    the policy and the critic, SVG can be shown to minimize the cross entropy (i.e. they
    are separated by a constant term -- the entropy of the policy).
    """

    def __init__(self, beta, q_fn, q_fn_targ, replay_memory):
        super(SVG, self).__init__(beta, q_fn, q_fn_targ, replay_memory, None)
        self.name = "SVG(0)"
    
    def select_action(self, x):
        return self.beta.select_action(x)
    
    def update(self, pol_opt, q_opt, v_opt, batch_size):
        for _ in range(iters):
            states, actions, rewards, next_states, masks = self.replay_memory.sample_and_split(batch_size)
            next_actions, _, _ = self.select_action(next_states)
            next_state_actions = torch.cat([next_states, next_actions], dim=-1)
            q_targ = rewards + masks * gamma * masks * self.q_fn_targ.get_min_value(next_state_actions)
            state_actions = torch.cat([states, actions], dim=-1)
            q_values = self.q_fn(state_actions)
            q_fn_loss = torch.mean(torch.cat([(q - q_targ.detach()) ** 2 for q in q_values], dim=-1))
            q_opt.zero_grad()
            q_fn_loss.backward()
            q_opt.step()

        states, _, _, _, _ = self.replay_memory.sample_and_split(batch_size)
        actions, _, _ = self.select_action(next_states)
        state_actions = torch.cat([states, actions], dim=-1)
        policy_loss = -torch.mean(self.q_fn(state_actions)[0])
        pol_opt.zero_grad()
        policy_loss.backward()
        pol_opt.step()
        utils.soft_update(self.q_fn_targ, self.q_fn)
        return policy_loss.item(), q_fn_loss.item()
    
    
class SAC(SVG):
    """
    Simple version of soft actor critic. This is the original version that doesn't make use
    of automatic alpha tuning, and has a separate V network (most modern versions don't). The 
    reason we chose to use this version is that it was simpler to get running -- we found that 
    adding automatic alpha tuning had a tendency to explode. The second reason we opted for
    version was because it is conceptually much closer to the theory outlined in the paper.
    To isolate effects, this implementation *doesn't* use multiple critics (i.e. TD3), though
    we have a version further below that does.
    """

    def __init__(self, beta, q_fn, v_fn, v_fn_targ, replay_memory):
        super(SAC, self).__init__(beta, None, None, replay_memory)
        self.name = "SAC"
        self.q_fn = q_fn
        self.v_fn = v_fn
        self.v_fn_targ = v_fn_targ
        utils.hard_update(self.v_fn_targ, self.v_fn)
    
    def update(self, pol_opt, q_opt, v_opt, batch_size):
        for _ in range(iters):
            states, batch_actions, rewards, next_states, masks = self.replay_memory.sample_and_split(batch_size)
            actions, log_probs, _ = self.select_action(states)

            state_actions = torch.cat([states, actions], dim=-1)
            v_targ = self.q_fn.get_min_value(state_actions) - log_probs
            v_values = self.v_fn(states)
            v_fn_loss = torch.mean(torch.cat([(v - v_targ.detach()) ** 2 for v in v_values], dim=-1))
            v_opt.zero_grad()
            v_fn_loss.backward()
            v_opt.step()
            
            q_targ = rewards + masks * gamma * masks * self.v_fn_targ.get_min_value(next_states)
            batch_state_actions = torch.cat([states, batch_actions], dim=-1)
            q_values = self.q_fn(batch_state_actions)
            q_fn_loss = torch.mean(torch.cat([(q - q_targ.detach()) ** 2 for q in q_values], dim=-1))
            q_opt.zero_grad()
            q_fn_loss.backward()
            q_opt.step()

        states, _, _, _, _ = self.replay_memory.sample_and_split(batch_size)
        actions, log_probs, _ = self.select_action(states)
        state_actions = torch.cat([states, actions], dim=-1)
        policy_loss = torch.mean(alpha * log_probs - self.q_fn(state_actions)[0])
        pol_opt.zero_grad()
        policy_loss.backward()
        pol_opt.step()
        utils.soft_update(self.v_fn_targ, self.v_fn)
        return policy_loss.item(), q_fn_loss.item()