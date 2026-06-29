import numpy as np
import matplotlib.pyplot as plt


class HillProblem():
    """
    Definition:
        - Single Asteroid
    """
    
    def __init__(self,Omega,mu_tilde):
        self.Omega = Omega
        self.mu_tilde = mu_tilde

        # sun to asteroid
        self.S_vec = np.array([1,0,0])

        # normalization factors
        self.r_H = (mu_tilde/(3*Omega**2))**(1/3) #km
        self.a_H = mu_tilde/self.r_H**2 # km/s2

    def find_grad_U(self, r_vec, mu_tilde):

        # position vector
        x, y, z = r_vec
        r = np.linalg.norm(r_vec)

        # calculate grad(U)
        term1 = -mu_tilde * r_vec / r**3
        term2 = np.array([3 * self.Omega**2 * x,0,-self.Omega**2 * z])

        return term1 + term2
    
    def find_n_e(self,r_vec):
        """
        calculate normal vector n_e at the equilibrium point 
        """

        grad_U = self.find_grad_U(r_vec*self.r_H, self.mu_tilde)
        grad_norm = np.linalg.norm(grad_U)
        n_e = -grad_U/grad_norm
        return n_e
    
    def find_a0_e(self,r_vec):
        """
        r_vec: normalized position

        Assumptions:
            - AEP: r_dot = r_dotdot = 0 => grad (U) = -a_s

        Return: a0/a_H (normalized) 
        """

        # calculate normal vector n_vec
        n_e = self.find_n_e(r_vec)
        
        # n_vec must be outwards from the sun
        if self.S_vec.dot(n_e)<=0: return np.nan

        # magnitude of solar-sailacceleration
        grad_U = self.find_grad_U(r_vec*self.r_H, self.mu_tilde) 
        a0 = -grad_U.dot(n_e)/(self.S_vec.dot(n_e))**2 # km/s2

        return a0/self.a_H 
    
    def plot_a0_contours(self,x,y,plane,levels):
        """
        Example:
            x = np.linspace(-1.5, 1.5, 400)
            y = np.linspace(-1.5, 1.5, 400)
            levels = np.linspace(0, 100, 200)
            plane: "xz" or "xy"
        """
        plot_acceleration_contours(x,y,plane,self.find_a0_e,levels)


    
class Bicircular():

    def __init__(self,mu,mu3,Omega_s,a,kappa):
        
        self.mu = mu
        self.mu3 = mu3
        self.Omega_s = Omega_s
        self.a = a
        self.kappa = kappa

    def find_grad_U(self,r_vec,r3_vec):

        # position vector
        x, y, z = r_vec
        r = np.linalg.norm(r_vec)

        # r1: asteroid 1 (big) to sail
        r_ast1 = np.array([-self.mu,0,0])
        r1_vec = r_vec - r_ast1
        r1 = np.linalg.norm(r1_vec)

        # r2: asteroid 2 (small) to sail
        r_ast2 = np.array([1-self.mu,0,0])
        r2_vec = r_vec - r_ast2
        r2 = np.linalg.norm(r2_vec)

        # r3: origin to Sun
        r3 = np.linalg.norm(r3_vec)

        # r4: sail to Sun
        r4_vec = r3_vec - r_vec
        r4 = np.linalg.norm(r4_vec)

        # calculate grad(U)
        term1 = (1-self.mu)*-r1_vec/r1**3 + self.mu*-r2_vec/r2**3
        term2 = np.array([x,y,0])
        term3 = self.mu3*(r4_vec/r4**3 - r3_vec/r3**3) 

        return term1 + term2 + term3

    def find_a0(self,r_vec,t):
        """
        Assumptions:
            - bicircular
            - AEP (r_dot = r_dotdot = 0 => nabla U = - a_s)
        """

        # get Sun direction/position
        S_vec,r3_vec = self.get_sun_parameters(t)

        # calculate normal vector n_vec
        grad_U = self.find_grad_U(r_vec,r3_vec)
        grad_norm = np.linalg.norm(grad_U)
        n_vec = -grad_U/grad_norm
        
        # n_vec must be outwards from the sun
        if S_vec.dot(n_vec)<=0: return np.nan

        # magnitude of solar-sailacceleration
        a0 = -grad_U.dot(n_vec)/(S_vec.dot(n_vec))**2 
        
        return a0 
    
    def get_sun_parameters(self,t):
        """
        Return:
            r3 = origin to Sun
            S_vec = unit vector from Sun to asteroid
        """

        Omega_s = self.Omega_s
        
        r3_vec = self.a*np.array([-np.cos(t*Omega_s),np.sin(t*Omega_s),0])/self.kappa
        S_vec = np.array([np.cos(t*Omega_s),-np.sin(t*Omega_s),0]) # sun to asteroid

        return S_vec,r3_vec
    
    def plot_a0_contours(self,x,y,plane,levels,t):
        """
        Example:
            x = np.linspace(-1.5, 1.5, 400)
            y = np.linspace(-1.5, 1.5, 400)
            levels = np.linspace(0, 100, 200)
            plane: "xz" or "xy"
        """
        plot_acceleration_contours(x,y,plane,self.find_a0,levels,params=[t])


       
        
def plot_acceleration_contours(x,y,plane,func,levels,params=[]):
    """
    x,y : direct inputs to 'func'

    Example:
        x = np.linspace(-1.5, 1.5, 400)
        y = np.linspace(-1.5, 1.5, 400)
        levels = np.linspace(0, 100, 200)
        plane: "xz" or "xy"
    """
    
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)


    build_r = {
        'xz': lambda a,b: np.array([a,0,b]),
        'xy': lambda a,b: np.array([a,b,0]),
    }


    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            r_vec = build_r[plane](X[i,j],Y[i,j])
            Z[i,j] = func(r_vec,*params)

    # plot
    fig, ax = plt.subplots()
    cs = ax.contour(X, Y, Z, levels=levels)
    fig.colorbar(cs, ax=ax)
    ax.grid(True)
    ax.set_title("$a_0$ contours")
    ax.set_xlabel("x")
    ax.set_ylabel(plane[1])
    ax.plot()