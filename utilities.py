import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, MultivariateNormal
from collections import namedtuple, deque
import random
import settings as cfg

device = cfg.device


class RandomProcess(object):
    def reset_states(self):
        pass

class AnnealedGaussianProcess(RandomProcess):
    def __init__(self, mu, sigma, sigma_min, n_steps_annealing):
        self.mu = mu
        self.sigma = sigma
        self.n_steps = 0

        if sigma_min is not None:
            self.m = -float(sigma - sigma_min) / float(n_steps_annealing)
            self.c = sigma
            self.sigma_min = sigma_min
        else:
            self.m = 0.
            self.c = sigma
            self.sigma_min = sigma

    @property
    def current_sigma(self):
        sigma = max(self.sigma_min, self.m * float(self.n_steps) + self.c)
        return sigma


# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckProcess(AnnealedGaussianProcess):
    def __init__(self, theta=0.15, mu=0., sigma=0.22, dt=1e-2, x0=None, size=1, sigma_min=None, n_steps_annealing=1000):
        super(OrnsteinUhlenbeckProcess, self).__init__(mu=mu, sigma=sigma, sigma_min=sigma_min, n_steps_annealing=n_steps_annealing)
        self.theta = theta
        self.mu = mu
        self.dt = dt
        self.x0 = x0
        self.size = size
        self.reset_states()

    def sample(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.current_sigma * np.sqrt(self.dt) * np.random.normal(size=self.size)
        self.x_prev = x
        self.n_steps += 1
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)

    
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'masks'))
class ReplayMemory:
    """
    Implements a basic replay memory buffer as used in DDPG, TD3, SVG, SAC, etc.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """
        Saves a transition.
        """
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = int((self.position + 1) % self.capacity)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def sample_and_split(self, batch_size):
        transitions = self.sample(batch_size)
        batch = Transition(*zip(*transitions))
        states = torch.stack(batch.state).float()
        actions = torch.stack(batch.action).float()
        rewards = torch.stack(batch.reward).float()
        masks = torch.stack(batch.masks).float()
        next_states = torch.stack(batch.next_state).float()
        return states, actions, rewards, next_states, masks

    def __len__(self):
        return len(self.memory)


# from https://github.com/joschu/modular_rl
# http://www.johndcook.com/blog/standard_deviation/
class RunningStat(object):
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)

    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape


class ZFilter:
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """

    def __init__(self, shape, demean=True, destd=True, clip=10.0):
        self.demean = demean
        self.destd = destd
        self.clip = clip

        self.rs = RunningStat(shape)

    def __call__(self, x, update=True):
        if update: self.rs.push(x)
        if self.demean:
            x = x - self.rs.mean
        if self.destd:
            x = x / (self.rs.std + 1e-8)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x

    def output_shape(self, input_space):
        return input_space.shape


def soft_update(target, source, tau=5e-3):
    """
    Performs a soft update of a target network for online, pathwise derivative methods
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

            
def hard_update(target, source):
    """
    Performs a hard update of a target network
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

            
def get_flat_params_from(model):
    """
    Returns a flattened parameter vector for conjugate gradient based method (TRPO)
    """
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))

    flat_params = torch.cat(params)
    return flat_params


def set_flat_params_to(model, flat_params):
    """
    Takes a flattened parameter vector, and uses it to set the network weights
    """
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size

        
def get_flat_grad_from(net, grad_grad=False):
    """
    Returns a flattened gradient vector for the network parameters
    """
    grads = []
    for param in net.parameters():
        if grad_grad:
            grads.append(param.grad.grad.view(-1))
        else:
            grads.append(param.grad.view(-1))
    flat_grad = torch.cat(grads)
    return flat_grad


def mvgaussian_fvp(gradient_vector, states, pi, beta):
    """
    Calculates the Fisher Vector Product using the KL-divergence of two Multivariate Gaussian
    Policies.
    """
    mu_pi, cov_pi = pi(states)
    mu_beta, cov_beta = pi(states)
    tril_pi = pi.reshape_output(cov_pi)
    tril_beta = beta.reshape_output(cov_beta)
    dist_pi = MultivariateNormal(mu_pi, scale_tril=tril_pi)
    dist_beta = MultivariateNormal(mu_beta, scale_tril=tril_beta)
    kl = torch.distributions.kl_divergence(dist_beta, dist_pi)
    kl = torch.mean(kl)
    grads = torch.autograd.grad(kl, pi.parameters(), create_graph=True)
    flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])
    kl_v = flat_grad_kl.dot(gradient_vector)
    grads = torch.autograd.grad(kl_v, pi.parameters())
    flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).data
    return flat_grad_grad_kl


def gaussian_fvp(gradient_vector, states, pi, beta):
    """
    Calculates the Fisher Vector Product using the KL-divergence of two Independent Gaussian
    Policies.
    """
    mus_pi, logsigmas_pi = pi(states)
    mus_beta, logsigmas_beta = beta(states)
    sigmas_pi = torch.exp(logsigmas_pi)
    sigmas_beta = torch.exp(logsigmas_beta)
    dist_pi = Normal(mus_pi, sigmas_pi)
    dist_beta = Normal(mus_beta, sigmas_beta)
    kl = torch.distributions.kl_divergence(dist_beta, dist_pi)
    kl = torch.mean(torch.sum(kl, dim=-1))
    grads = torch.autograd.grad(kl, pi.parameters(), create_graph=True)
    flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])
    kl_v = flat_grad_kl.dot(gradient_vector)
    grads = torch.autograd.grad(kl_v, pi.parameters())
    flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).data
    return flat_grad_grad_kl


def conjugate_gradient(fvp, gradient_vector, states, pi, beta, n_steps=10, residual_tol=1e-10):
    """
    Estimate the function Fv = g, where F is the FIM, and g is the gradient.
    Since dx ~= F^{-1}g for a stochastic process, v is dx. The CG algorithm 
    assumes the function is locally quadratic. In order to ensure our step 
    actually improves the policy, we need to do a linesearch after this.
    """
    x = torch.zeros(gradient_vector.size()).to(device)
    r = gradient_vector.clone()
    p = gradient_vector.clone()
    rdotr = torch.dot(r, r)
    for i in range(n_steps):
        fisher_vector_product = fvp(p, states, pi, beta)
        alpha = rdotr / p.dot(fisher_vector_product)
        x += alpha * p
        r -= alpha * fisher_vector_product
        new_rdotr = r.dot(r)
        tau = new_rdotr/rdotr
        p = r + tau * p
        rdotr = new_rdotr
        if rdotr <= residual_tol:
            break
    return x


def linesearch(trajectory, pi, critic, policy_loss, old_params, fullstep, expected_improve, max_backtracks=10, accept_ratio=.1):
    """
    Conducts an exponentially decaying linesearch to guarantee that our update step improves the
    model. 
    """
    set_flat_params_to(pi, old_params)
    fval = policy_loss(trajectory, pi, critic).data
    steps = 0.5**torch.arange(max_backtracks).to(device).float()
    for n, stepfrac in enumerate(steps):
        xnew = old_params+stepfrac*fullstep
        set_flat_params_to(pi, xnew)
        newfval = policy_loss(trajectory, pi, critic).data
        actual_improve = fval-newfval
        expected_improve = expected_improve*stepfrac
        ratio = actual_improve/expected_improve
        if ratio.item() > accept_ratio and actual_improve.item() > 0:
            return True, xnew
    return False, old_params