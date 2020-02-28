import numpy as np
from math import cos

class AngularRatesPID:
    """
    Implements a angular rates PID controller for a quadrotor. We want to solve the equation:
    
    [[u1],      [[kt,   kt,     kt,     kt],    [[w_1^2],
    [u2],   =   [0.,    lkt,    0.,   -lkt],     [w_2^2],
    [u3],       [-lkt,  0.,     lkt,    0.],     [w_3^2],
    [u4]]       [-kq,   kq,    -kq,     kq]]     [w_4^2]]
    
    Where u1, u2, u3, and u4, correspond to the commanded thrust, roll, pitch, and yaw rates respectively.
    """
    
    def __init__(self, K_p, K_i, K_d, control_matrix, u_max, w_max, mg, tr, dt=1e-3):
        """
        K_p, K_i, K_d are 4x4 matrices of the form:

        [[Kz,  0,   0,   0],
        [0,    Kp,  0,   0],
        [0,    0,   Kq,  0],
        [0,    0,   0,   Kr]]

        Where Kz, Kp, Kq, and Kr are gains for z_dot, and p, q and r angular rates, respectively.
        
        control_matrix is a 4x4 matrix as outlined above.
        """

        self.K_p = K_p
        self.K_i = K_i
        self.K_d = K_d
        self.control_matrix = control_matrix
        self.dt = dt
        self.inv_control_matrix = np.linalg.inv(control_matrix)
        self.integral = np.zeros((4,1))
        self.previous_error = np.zeros((4,1))
        self.u_max = u_max
        self.w_max_sq = w_max ** 2
        self.mg = mg
        self.tr = tr
    
    def reset(self):
        self.integral = np.zeros((4,1))
        self.previous_error = np.zeros((4,1))
    
    def u(self, target, state):
        error = target - state
        error[0,0] *= -1
        de = error - self.previous_error
        mask = np.abs(error) < self.tr
        self.integral += error*mask
        u = self.K_p.dot(error) + self.dt*self.K_i.dot(self.integral) + (1/self.dt)*self.K_d.dot(de)
        self.previous_error = error
        return u
    
    def w(self, target, state):
        xyz, zeta, xyz_dot, pqr = state
        phi, theta, _ = zeta
        X = np.array([[xyz_dot[-1]],
                        [pqr[0]],
                        [pqr[1]],
                        [pqr[2]]])
        U = self.u(target, X)
        U[0] += self.mg
        U[0] /= cos(phi) * cos(theta)
        V = np.vstack([np.clip(u, n[0], n[1]) for u, n in zip(U, self.u_max)])
        W = np.sqrt(np.clip(self.inv_control_matrix.dot(V), 0, self.w_max_sq))
        return W