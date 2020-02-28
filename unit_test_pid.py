import matplotlib.pyplot as plt
import numpy as np
import pid
import gym
import gym_aero
from math import pi
from scipy.optimize import minimize

def get_gains(theta):
  K_p = np.zeros((4,4))
  K_i = np.zeros((4,4))
  K_d = np.zeros((4,4))
  for i in range(4):
    j = i % 4
    K_p[j,j] = theta[i]
    K_i[j,j] = theta[i+4]
    K_d[j,j] = theta[i+8]
  #print(K_p)
  #print(K_i)
  #print(K_d)
  #input()
  return K_p, K_i, K_d

def rollout(env, controller):
  controller.reset()
  state = env.reset()
  cost = 0
  prev_X = np.zeros((4,1))
  for h in H:
    xyz, zeta, uvw, pqr = state
    X = np.array([[xyz[-1]],
                  [pqr[0]],
                  [pqr[1]],
                  [pqr[2]]])
    w = controller.w(targ, state)
    state = env.step(w)
    cost += np.sum((X - targ) ** 2)
    prev_X = X
  return cost

def cost_fn(theta, env, targ, controller, n):
  K_p, K_i, K_d = get_gains(theta)
  controller.K_p = K_p
  controller.K_i = K_i
  controller.K_d = K_d
  cost = 0
  for _ in range(n):
    cost += rollout(env, controller)
  return cost/n

# set target rates
T = 1
dt = 0.001
targ = np.zeros((4,1))
H = np.arange(0, T, dt)

# initialize quadrotor
env = gym.make("PIDTest-v0")

K_p = np.eye(4)
K_i = np.zeros((4,4))
K_d = np.eye(4)

K_p[0,0] = 2
K_d[0,0] = 0.01

K_p[1,1] = 2.7/10
K_i[1,1] = 0
K_d[1,1] = 0.01/5

K_p[2,2] = 2.7/10
K_i[2,2] = 0
K_d[2,2] = 0.01/5

K_p[3,3] = 2.7/10
K_i[3,3] = 0
K_d[3,3] = 0.01/5

print(K_p)
print(K_i)
print(K_d)
print()

control_matrix = env.Q
u_max = np.array([[0, env.max_u1],
                  [-env.max_u2, env.max_u2],
                  [-env.max_u3, env.max_u3],
                  [-env.max_u2, env.max_u4]])
w_max = env.max_omega
tr = np.array([[0.25],
              [10*(2*pi/360)],
              [10*(2*pi/360)],
              [10*(2*pi/360)]])

mg = env.ac_mass * 9.81
rate_controller = pid.AngularRatesPID(K_p, K_i, K_d, control_matrix, u_max, w_max, mg, tr)

# test pid controller
test_data = []
print("targ: ", targ)
state = env.reset()
print("STATE: ", state)
input()
for t in H:
  env.render()
  xyz, zeta, uvw, pqr = state
  xyz_dot = env.body_to_inertial(uvw)
  X = np.array([[xyz_dot[-1]],
                [pqr[0]],
                [pqr[1]],
                [pqr[2]]])
  print("X: ", X)
  test_data.append(X.reshape(1, -1))
  w = rate_controller.w(targ, (xyz, zeta, xyz_dot, pqr))
  print("w: ", w[:,0])
  state = env.step(w[:,0])
  print(state)

plt.figure(figsize=(10,10))
state = np.vstack(test_data)
titles = ["Z-Dot Error", "P-Error", "Q-Error", "R-Error"]
xlabels = ["Time (s)", "Time (s)", "Time (s)", "Time (s)"]
ylabels = ["m/s", "rad/s", "rad/s", "rad/s"]
for j in range(4):
    plt.subplot(2, 2, j + 1)
    plt.plot(H, state[:,j])
    plt.title(titles[j])
    plt.xlabel(xlabels[j])
    plt.ylabel(ylabels[j])
plt.savefig("output.png")
plt.show()