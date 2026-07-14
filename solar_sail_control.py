import numpy as np
import systems
import matplotlib.pyplot as plt
import find_orbits.linearize as linearize
from scipy.linalg import solve_continuous_are
from scipy.integrate import solve_ivp

AU_TO_KM = 149_597_870.7

"""
2D AEP STABILIZATION

Assumptions:
    - alpha is not a state variable, but a control variable (irrealistic)
    - 2D
    - zx plane
"""


class ControlledHillProblem(systems.HillProblem):
    
    def __init__(self,Omega, mu_tilde):
        super().__init__(Omega, mu_tilde)


    def set_AEP(self, r_eq):

        self.r_eq = r_eq
        self.n_e  = self.find_n_e(r_eq)
        self.a0_e = self.find_a0_e(r_eq)
        self.X_eq = np.concatenate((r_eq, np.zeros(3)))
        self.alpha_eq = 0.0
        self.M = linearize.compute_M(lambda r: self.F(r, self.alpha_eq), r_eq)
        self.A = linearize.build_A(self.M)
        self.B = self.compute_B()

    def rotate_about_y(self, vec, alpha):
        """
        Positive alpha rotates x toward z
        """
        c = np.cos(alpha)
        s = np.sin(alpha)
        R = np.array([
            [ c, 0, -s],
            [ 0, 1,  0],
            [ s, 0,  c]
        ])
        return R @ vec
    
    def F(self, r, alpha):
        """
        F = grad(V) + a_srp 
        Assumptions:
            - bidimensional case (only alpha)
            - a0 constant, since distance to Sun doesn't change expressively 
        """
        n = self.rotate_about_y(self.n_e, alpha)
        grad_U = self.find_grad_U(r * self.r_H, self.mu_tilde) / self.a_H

        # n is saturated in closed_loop_dynamics, so self.S_vec @ n > 0
        a_srp = self.a0_e * (self.S_vec @ n)**2 * n
        return grad_U + a_srp
    
    def controlled_dynamics(self, t, X, alpha):
        """
        Dynamical model: X = [x, y, z, vx, vy, vz]
        alpha = angle deviation from n_e 
        return X_dot
        observations:
            - positions/velocities normalized by r_H/a_H
            - positive alpha rotates x toward z
        """
    
        r = X[:3]
        v = X[3:]
        acc = self.F(r,alpha)
        return np.concatenate((v, acc))
    
    def compute_B(self, h=1e-6):
        """
        X_dot = f(X,u)
        dX_dot = AdX + Bdu

        A = df/dX|_eq ; B = df/du|_eq
        """
        
        dF = (self.F(self.r_eq, self.alpha_eq + h) - self.F(self.r_eq, self.alpha_eq - h)) / (2 * h)
        B = np.concatenate((np.zeros(3), dF))
        return B.reshape(-1, 1)

    # State-feedback controller (LQR)
    def lqr(self, A, B, Q, R):
        P = solve_continuous_are(A, B, Q, R)
        K = np.linalg.inv(R) @ B.T @ P
        return K
    
    def set_weights(self,Q,R):
        """
        Q: State weighting matrix
        R: Control weighting matrix
        """

        self.Q = Q
        self.R = R
    
    def closed_loop_dynamics(self, t, X):

        delta_X = X - self.X_eq
        K = self.lqr(self.A,self.B,self.Q,self.R)
        delta_alpha = float((-K @ delta_X).item())
  

        # limiting alpha so that -pi/2 < theta < pi/2
        # where theta is the angle from x to n
        theta_e = np.arctan2(self.n_e[2], self.n_e[0])
        margin = np.deg2rad(10)
        alpha = np.clip(
            self.alpha_eq + delta_alpha,
            -np.pi/2 - theta_e + margin,
            np.pi/2 - theta_e - margin
        )
            
        return self.controlled_dynamics(t, X, alpha)

    def add_perturbation(self,t_span,perturbation,rtol=1e-10,atol=1e-12,dense_output=True):
        """
        solve EDO X_dot = f(t,X) 
        initial value = X_eq + perturbation
        """

        X0 = self.X_eq + perturbation

        # ivp = Initial Value Problem
        sol = solve_ivp(
            self.closed_loop_dynamics,
            t_span=t_span,
            y0=X0,
            rtol=rtol,
            atol=atol,
            dense_output=dense_output
        )

        plt.plot(sol.y[0], sol.y[2])
        plt.scatter([self.r_eq[0]], [self.r_eq[2]], marker="x")
        plt.xlabel("x")
        plt.ylabel("z")
        plt.axis("equal")
        plt.grid(True)
        plt.show()
