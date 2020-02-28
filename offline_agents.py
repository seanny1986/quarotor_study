import torch
import torch.nn as nn
import utilities as utils
import settings as cfg

device = cfg.device
gamma = cfg.offline_gamma
tau = cfg.tau
iters = cfg.offline_update_iters
bootstrap = cfg.bootstrap
eps = cfg.eps
max_kl = cfg.max_kl

class A2C(nn.Module):
    """
    Basic Monte Carlo Policy Gradient. This implementation is single-threaded only, and uses
    the score function estimator to find the policy gradient. We use importance sampling to
    take multiple update steps at the end of each trajectory. This is necessary to correct
    for the fact that the value function was estimated under a previous policy.
    """
    def __init__(self, beta, v_fn):
        super(A2C, self).__init__()
        self.beta = beta
        self.v_fn = v_fn
        
    def select_action(self, x):
        return self.beta.select_action(x)

    def get_phi(self, trajectory, critic):
        states = torch.stack(trajectory["states"]).to(device)
        rewards = torch.stack(trajectory["rewards"]).to(device)
        next_states = torch.stack(trajectory["next_states"]).to(device)
        masks = torch.stack(trajectory["masks"]).to(device)
        values = critic.get_max_value(states)
        if bootstrap: next_values = critic.get_max_value(next_states)
        else: next_values = torch.zeros(rewards.size(0), 1).to(device)
        returns = torch.Tensor(rewards.size(0),1).to(device)
        deltas = torch.Tensor(rewards.size(0),1).to(device)
        advantages = torch.Tensor(rewards.size(0),1).to(device)
        prev_return = 0
        prev_value = 0
        prev_advantage = 0
        for i in reversed(range(rewards.size(0))):
            returns[i] = rewards[i] + gamma * (prev_return * masks[i] + (1 - masks[i]) * next_values[i])
            deltas[i] = rewards[i] + gamma * prev_value * masks[i] - values.data[i]
            advantages[i] = deltas[i] + gamma * tau * prev_advantage * masks[i]
            prev_return = returns[i, 0]
            prev_value = values.data[i, 0]
            prev_advantage = advantages[i, 0]
        return advantages, returns

    def update(self, pol_opt, q_opt, v_opt, trajectory):
        log_probs = torch.stack(trajectory["log_probs"]).to(device)
        states = torch.stack(trajectory["states"]).to(device)
        actions = torch.stack(trajectory["actions"]).to(device)
        for i in range(iters):
            advantages, returns = self.get_phi(trajectory, self.v_fn)
            phi = advantages / returns.std()
            values = self.v_fn(states)
            v_fn_loss = torch.mean(torch.cat([(v - returns.detach()) ** 2 for v in values], dim=-1))

            lp_p = self.beta.get_log_prob(states, actions)
            ratio = torch.exp(lp_p - log_probs.detach())
            pol_loss = -torch.mean(ratio * phi.detach())
            loss = pol_loss + v_fn_loss

            pol_opt.zero_grad()
            v_opt.zero_grad()
            if i < iters-1:
                loss.backward(retain_graph=True)
            else:
                loss.backward()
            pol_opt.step()
            v_opt.step()
        return pol_loss.item(), v_fn_loss.item()

        
class PPO(A2C):
    """
    Simple implementation of the proximal policy optimization algorithm (Schulman, 2017). The
    only difference between this and a standard REINFORCE algorithm is the use of the clipped
    objective, which helps keep the policy updates bound to a trust region (though there is some
    controversy around this).

    This implementation uses the same importance sampling correction as in the REINFORCE implementation
    above, but uses a clipped surrogate objective to keep the policy update bounded to a trust 
    region.
    """
    def __init__(self, beta, v_fn):
        super(PPO, self).__init__(beta, v_fn)
    
    def update(self, pol_opt, q_opt, v_opt, trajectory):
        log_probs = torch.stack(trajectory["log_probs"]).to(device)
        states = torch.stack(trajectory["states"]).to(device)
        actions = torch.stack(trajectory["actions"]).to(device)
        for i in range(iters):
            deltas, returns = self.get_phi(trajectory, self.v_fn)
            phi = deltas / returns.std()
            values = self.v_fn(states)
            v_fn_loss = torch.mean(torch.cat([(v - returns.detach()) ** 2 for v in values], dim=-1))
            
            lp_p = self.beta.get_log_prob(states, actions)
            ratio = torch.exp(lp_p - log_probs.detach())
            clipped_objective = torch.min(ratio * phi, torch.clamp(ratio, 1 + eps, 1 - eps) * phi)
            pol_loss = -torch.mean(clipped_objective)
            loss = pol_loss + v_fn_loss
            
            pol_opt.zero_grad()
            v_opt.zero_grad()
            if i < iters-1:
                loss.backward(retain_graph=True)
            else:
                loss.backward()
            pol_opt.step()
            v_opt.step()
        return pol_loss.item(), v_fn_loss.item()

    
class TRPO(A2C):
    """
    Simple implementation of Trust Region Policy Optimization (Schulman, 2016). This is second order
    extension of the REINFORCE algorithm, that bounds updates to be within a given KL-divergence of
    the original policy. The advantage of this is that it stabilizes learning by preventing the policy
    from taking a large update and moving into a "bad" region of the search space from which recovery
    might be difficult or impossible.

    This implementation uses CG method outlined in the original paper to find the gradient direction.
    Unlike the original, we use a first order update of the value function, since it allows us to use
    the GPU. Empirically, this seems to work, though we haven't quantified the difference in performance
    (if any) compared to using a second-order critic update.
    """
    def __init__(self, beta, pi, v_fn, fvp):
        super(TRPO, self).__init__(beta, v_fn)
        self.pi = pi
        self.fvp = fvp
        utils.hard_update(self.pi, self.beta)

    def policy_loss(self, traj, pi, critic):
        states = torch.stack(traj["states"]).to(device)
        actions = torch.stack(traj["actions"]).to(device)
        log_probs = torch.stack(traj["log_probs"]).to(device)
        deltas, returns = self.get_phi(traj, critic)
        phi = deltas / returns.std()
        lp_p = pi.get_log_prob(states, actions)
        ratio = torch.exp(lp_p - log_probs.detach())
        loss = -ratio * phi
        return loss.mean()

    def update(self, pol_opt, q_opt, v_opt, trajectory):
        states = torch.stack(trajectory["states"]).to(device)
        for i in range(iters):
            _, returns = self.get_phi(trajectory, self.v_fn)
            returns = (returns - returns.mean()) / returns.std()
            values = self.v_fn(states)
            v_fn_loss = torch.mean(torch.cat([(v - returns.detach()) ** 2 for v in values], dim=-1))
            v_opt.zero_grad()
            if i < iters-1:
                v_fn_loss.backward(retain_graph=True)
            else:
                v_fn_loss.backward()
            v_opt.step()
        pol_loss = self.policy_loss(trajectory, self.pi, self.v_fn)
        grads = torch.autograd.grad(pol_loss, self.pi.parameters())
        loss_grad = torch.cat([grad.view(-1) for grad in grads]).detach()
        stepdir = utils.conjugate_gradient(self.fvp, -loss_grad, states, self.pi, self.beta)
        shs = 0.5 * (stepdir.dot(self.fvp(stepdir, states, self.pi, self.beta)))
        lm = torch.sqrt(max_kl / shs)
        fullstep = stepdir * lm
        expected_improve = -loss_grad.dot(fullstep)
        old_params = utils.get_flat_params_from(self.beta)
        _, params = utils.linesearch(trajectory, self.pi, self.v_fn, self.policy_loss, old_params, fullstep, expected_improve)
        utils.set_flat_params_to(self.pi, params)
        utils.hard_update(self.beta, self.pi)
        return pol_loss.item(), v_fn_loss.item()


class  CorrelatedA2C(A2C):
    def __init__(self, beta, v_fn):
        super(CorrelatedA2C, self).__init__(beta, v_fn)
    
    def get_phi(self, trajectory, critic):
        states = torch.stack(trajectory["states"]).to(device)
        rewards = torch.stack(trajectory["rewards"]).to(device)
        next_states = torch.stack(trajectory["next_states"]).to(device)
        masks = torch.stack(trajectory["masks"]).to(device)
        if bootstrap: next_values = critic.get_min_value(next_states)
        else: next_values = torch.zeros(rewards.size(0), 1).to(device)
        returns = torch.Tensor(rewards.size(0),1).to(device)
        deltas = torch.Tensor(rewards.size(0),1).to(device)
        advantages = torch.Tensor(rewards.size(0),1).to(device)
        prev_return = 0
        for i in reversed(range(rewards.size(0))):
            returns[i] = rewards[i] + gamma * (prev_return * masks[i] + (1 - masks[i]) * next_values[i])
            prev_return = returns[i, 0]
        values = critic.get_highest_correlation(states, returns)
        prev_value = 0
        prev_advantage = 0
        for i in reversed(range(rewards.size(0))):
            deltas[i] = rewards[i] + gamma * prev_value * masks[i] - values.data[i]
            advantages[i] = deltas[i] + gamma * tau * prev_advantage * masks[i]
            prev_value = values.data[i, 0]
            prev_advantage = advantages[i, 0]
        return advantages, returns
  
    def update(self, pol_opt, q_opt, v_opt, trajectory):
        log_probs = torch.stack(trajectory["log_probs"]).to(device)
        states = torch.stack(trajectory["states"]).to(device)
        actions = torch.stack(trajectory["actions"]).to(device)
        for i in range(iters):
            advantages, returns = self.get_phi(trajectory, self.v_fn)
            phi = advantages / returns.std()
            values = self.v_fn(states)
            v_fn_loss = torch.mean(torch.cat([(v - returns.detach()) ** 2 for v in values], dim=-1))

            lp_p = self.beta.get_log_prob(states, actions)
            ratio = torch.exp(lp_p - log_probs.detach())
            pol_loss = -torch.mean(ratio * phi.detach())
            loss = pol_loss + v_fn_loss

            pol_opt.zero_grad()
            v_opt.zero_grad()
            if i < iters-1:
                loss.backward(retain_graph=True)
            else:
                loss.backward()
            pol_opt.step()
            v_opt.step()
        return pol_loss.item(), v_fn_loss.item()

        