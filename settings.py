import torch

# cuda device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# generic hyperparameter settings
hidden_dim = 256
training_iterations = 1000
log_freq = 10
render = True
save_video = True
reps = 1
test_runs = 10
num_envs = 32

# limits for stochastic policy networks
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

# OFFLINE ALGORITHM SETTINGS
offline_batch_size = 16384
pol_lr = 1e-4
v_lr = 1e-4
max_kl = 1e-2
bootstrap = False
offline_gamma = 0.995
tau = 0.97
eps = 0.2
offline_update_iters = 4

# ONLINE ALGORITHM SETTINGS
warmup = 1000
online_steps = 1000
online_batch_size = 256
pol_lr = 1e-4
v_lr = 1e-4
q_lr = 1e-4
online_gamma = 0.99
online_update_iters = 3
memory_size = 1e6

ddpg_sigma = 1
ou_theta = 0.15
ou_mu = 0
ou_sigma = 0.2

td3_sigma = 0.2
c = 0.5

sac_alpha = 0.2