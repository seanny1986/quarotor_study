import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, MultivariateNormal
import utilities as utils
import settings as cfg

device = cfg.device
LOG_SIG_MAX = cfg.LOG_SIG_MAX
LOG_SIG_MIN = cfg.LOG_SIG_MIN

artanh = lambda z : 0.5 * torch.log((1+z)/(1-z))

class DeterministicPolicy(nn.Module):
    """
    Deterministic policy function for continuous control tasks. This policy relies
    on noise injection to explore the space (i.e. OU noise as is used in DDPG-based
    algorithms)
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DeterministicPolicy, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.mu = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                               nn.ReLU(),
                               nn.Linear(hidden_dim, hidden_dim),
                               nn.ReLU(),
                               nn.Linear(hidden_dim, output_dim))
        
    def forward(self, x):
        mu = torch.tanh(self.mu(x))
        return mu


class IndependentGaussianPolicy(DeterministicPolicy):
    """
    Gaussian policy function for continuous control tasks. Assumes all actions are
    independent (i.e. diagonal covariance matrix).
    """
    def __init__(self, input_dim, hidden_dim, output_dim, pwd=False):
        super(IndependentGaussianPolicy, self).__init__(input_dim, hidden_dim, output_dim)   
        self.logsigma = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                               nn.ReLU(),
                               nn.Linear(hidden_dim, hidden_dim),
                               nn.ReLU(),
                               nn.Linear(hidden_dim, output_dim))
        self.pwd = pwd
        
    def forward(self, x):
        mu = self.mu(x)
        logsigma = torch.clamp(self.logsigma(x), min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mu, logsigma

    def select_action(self, x):
        mu, logsigma = self.forward(x)
        sigma = torch.exp(logsigma)
        dist = Normal(mu, sigma)
        if self.pwd:
            action = dist.rsample()
        else:
            action = dist.sample()
        log_prob = torch.sum(dist.log_prob(action), dim=-1, keepdim=True)
        entropy = dist.entropy()
        return action, log_prob, entropy

    def get_log_prob(self, states, actions):
        mu, logsigma = self.forward(states)
        sigma = torch.exp(logsigma)
        dist = Normal(mu, sigma)
        return torch.sum(dist.log_prob(actions), dim=-1, keepdim=True)


class SquashedGaussianPolicy(IndependentGaussianPolicy):
    """
    Sqaushed Gaussian policy function for continuous control tasks. Assumes all actions are
    independent (i.e. diagonal covariance matrix), and uses a tanh squashing function to
    constrain all actions to [-1, 1]. In order to recover the log_prob, we need to subtract
    log(1 - tahn(action)^2) from the original log_prob (to understand why, see normalizing
    flows and the change of variables formula).
    """
    def __init__(self, input_dim, hidden_dim, output_dim, pwd=False):
        super(SquashedGaussianPolicy, self).__init__(input_dim, hidden_dim, output_dim, pwd=pwd)
        
    def select_action(self, x):
        action, log_prob, entropy = super(SquashedGaussianPolicy, self).select_action(x)
        action = torch.tanh(action)
        log_prob -= torch.sum(torch.log((1 - action.pow(2)) + 1e-10), dim=-1, keepdim=True)
        return action, log_prob, entropy
    
    def get_log_prob(self, states, actions):
        log_prob = super(SquashedGaussianPolicy, self).get_log_prob(states, artanh(actions))
        log_prob -= torch.sum(torch.log((1 - actions.pow(2)) + 1e-10), dim=-1, keepdim=True)
        return log_prob


class MVGaussianPolicy(nn.Module):
    """
    Multivariate Gaussian policy function for continuous control tasks. Assumes all actions 
    are correlated. To do this, we output a triangular scaling matrix, where the diagonal
    elements are constrained to being positive. The downside of this type of policy is that
    the number of outputs from the network scales with the square of the number of actions.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, pwd=False):
        super(MVGaussianPolicy, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.tril_dim = int(output_dim * (output_dim + 1) / 2)

        self.mu = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                               nn.ReLU(),
                               nn.Linear(hidden_dim, hidden_dim),
                               nn.ReLU(),
                               nn.Linear(hidden_dim, output_dim))
        self.cov = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                               nn.ReLU(),
                               nn.Linear(hidden_dim, hidden_dim),
                               nn.ReLU(),
                               nn.Linear(hidden_dim, self.tril_dim))
    
    def reshape_output(self, mu, cov):
        tril = torch.zeros((mu.size(0), mu.size(1), mu.size(1)))
        tril_indices = torch.tril_indices(row=mu.size(1), col=mu.size(1), offset=0)
        tril[:, tril_indices[0], tril_indices[1]] = cov
        diag_indices = np.diag_indices(tril.shape[1])
        tril[:, diag_indices[0], diag_indices[1]] = torch.exp(tril[diag_indices[0], diag_indices[1]])
        return tril

    def forward(self, x):
        mu = self.mu(x)
        cov = self.cov(x)
        return mu, cov

    def select_action(self, x):
        mu, cov = self.forward(x)
        tril = self.reshape_output(mu, cov)
        dist = MultivariateNormal(mu, scale_tril=tril)
        if self.pwd:
            action = dist.rsample()
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy
    
    def get_log_prob(self, states, actions):
        mu, cov = self.forward(states)
        tril = self.reshape_output(mu, cov)
        dist = MultivariateNormal(mu, scale_tril=tril)
        return dist.log_prob(actions)


class SquashedMVGaussianPolicy(MVGaussianPolicy):
    """
    Sqaushed Multivariate Gaussian policy function for continuous control tasks. Assumes all 
    actions are correlated, and uses a tanh squashing function to constrain all actions to 
    [-1, 1]. In order to recover the log_prob, we need to subtract log(1 - tahn(action)^2) 
    from the original log_prob (to understand why, see normalizing flows and the change of 
    variables formula).
    """

    def __init__(self, input_dim, hidden_dim, output_dim, pwd=False):
        super(SquashedMVGaussianPolicy, self).__init__(input_dim, hidden_dim, output_dim, pwd=pwd)
        
    def select_action(self, x):
        action, log_prob, entropy = super(SquashedMVGaussianPolicy, self).select_action(x)
        action = torch.tanh(action)
        log_prob -= torch.sum(torch.log((1 - action.pow(2)) + 1e-10))
        return action, log_prob, entropy

    def get_log_prob(self, states, actions):
        log_prob = super(SquashedMVGaussianPolicy, self).get_log_prob(states, artanh(actions))
        log_prob -= torch.sum(torch.log((1 - action.pow(2)) + 1e-10), dim=-1, keepdim=True)
        return log_prob

        
class ValueNet(nn.Module):
    """
    Simple parameterized value function. We use this for both state value functions,
    and state-action value functions. We can spawn either one or multiple heads and
    use the TD3 trick of selecting the most conservative (minimum) value function
    estimate. If the number of heads is 1, then our value function is the same as a
    standard implementation.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=1):
        super(ValueNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, output_dim)) for _ in range(num_heads)])
        
    def forward(self, x):
        values = [h(x) for h in self.heads]
        return values
    
    def get_min_value(self, x):
        values = self.forward(x)
        data = torch.min(torch.cat(values, dim=-1), dim=-1, keepdim=True)
        min_v = data[0]
        return min_v

    def get_max_value(self, x):
        values = self.forward(x)
        data = torch.max(torch.cat(values, dim=-1), dim=-1, keepdim=True)
        max_v = data[0]
        return max_v
    
    def get_nearest_value(self, x, targ_vals):
        values = self.forward(x)
        data = torch.cat(values, dim=-1)
        dist = torch.norm(targ_vals - data, dim=-1, p=None)
        nearest = dist.topk(1, largest=False)
        return nearest.values
    
    def get_highest_correlation(self, x, targ_vals):
        values = self.forward(x)
        data = torch.cat(values, dim=-1)
        vx = data - torch.mean(data, dim=0)
        vy = targ_vals - torch.mean(targ_vals)
        cov_xy = torch.sum(vx * vy.expand_as(vx), dim=0)
        var_x = torch.sum(vx ** 2, dim=0)
        var_y = torch.sum(vy ** 2, dim=0)
        denom = torch.sqrt(var_x) * torch.sqrt(var_y)
        pearsons =  cov_xy / denom
        best = torch.max(pearsons, dim=-1, keepdim=True)
        idx = best[1]
        return data[:, idx]
