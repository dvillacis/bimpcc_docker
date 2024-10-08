import numpy as np
from .AbstractMPCC import AbstractMPCC
from .utils import hes
import scipy.sparse as sp
import matplotlib.pyplot as plt


class DCTVMPCC2(AbstractMPCC):

    def __init__(self, pi, Kx, Ky, K, utrue, unoisy, epsilon=1e-4, gam=10, rho=1e-3, q=0.5):
        '''
        TVDenoising MPCC Penalized Problem
        
        Parameters:
        pi: float
            Penalty parameter
        Kx: scipy.sparse.csr_matrix
            Gradient Operator - x direction
        Ky: scipy.sparse.csr_matrix
            Gradient Operator - y direction
        Q: scipy.sparse.lil_matrix
            Parameter Patch Operator
        A: scipy.sparse.lil_matrix
            Forward Operator
        utrue: np.ndarray
            True image
        '''
        self.pi = pi
        self.Kx = Kx
        self.Ky = Ky
        self.K = K
        self.M, self.N = Kx.shape
        self.A = np.eye(self.N)
        self.Q = np.ones((self.M, 1))
        self.P = self.Q.shape[1]
        self.utrue = utrue.ravel()
        self.unoisy = unoisy.ravel()
        self.W = np.eye(self.N)
        self.rho = rho
        self.q = q
        self.epsilon = epsilon
        self.gam = gam

    def getvars(self, x):
        '''
        Extracts the variables from the concatenated vector x
        
        Parameters:
        x: np.ndarray
            Concatenated vector of variables
            
        Returns:
        u: np.ndarray
            Image variable
        qx: np.ndarray
            Gradient in x direction
        qy: np.ndarray
            Gradient in y direction
        alpha: np.ndarray
            Patch variable
        r: np.ndarray 
            Complementarity variable  
        '''
        u = x[:self.N]
        zx = x[self.N:self.N+self.M]
        zy = x[self.N+self.M:self.N+2*self.M]
        beta = x[self.N+2*self.M:self.N+2*self.M+self.P]
        r = x[self.N+2*self.M+self.P:self.N+3*self.M+self.P]
        delta = x[self.N+3*self.M+self.P:self.N+4*self.M+self.P]
        theta = x[self.N+4*self.M+self.P:self.N+5*self.M+self.P]
        return u, zx, zy, beta, r, delta, theta

    def complementarity(self, x):
        u, zx, zy, beta, r, delta, theta = self.getvars(x)

        return np.dot(r, (self.Q@beta)-delta)
        # return np.dot(r, delta)

    def min_complementarity(self, x):
        u, zx, zy, beta, r, delta, theta = self.getvars(x)
        m = np.minimum(r, (self.Q@beta)-delta)
        return np.linalg.norm(m)

    def objective(self, x):
        u, zx, zy, beta, r, delta, theta = self.getvars(x)

        # print(f'Approx pi_max: {(0.01*np.linalg.norm(x)**2)/(0.5*np.dot(r,delta))}')
        m = np.maximum(0, -delta)
        return 0.5*np.linalg.norm(u-self.utrue)**2 + self.pi*self.complementarity(x) + self.epsilon*np.linalg.norm(zx)**2 + self.epsilon*np.linalg.norm(zy)**2 + self.epsilon*np.linalg.norm(beta)**2 + self.epsilon*np.linalg.norm(r)**2 + self.epsilon*np.linalg.norm(theta)**2 + self.epsilon*np.linalg.norm(delta)**2

    def gradient(self, x):
        u, zx, zy, beta, r, delta, theta = self.getvars(x)
        grad_u = u-self.utrue
        grad_zx = 2*self.epsilon*zx
        grad_zy = 2*self.epsilon*zy
        grad_beta = self.pi*(self.Q.T@r) + 2*self.epsilon*beta
        grad_r = self.pi*((self.Q@beta)-delta) + 2*self.epsilon*r
        grad_delta = -self.pi*(r) + 2*self.epsilon*delta
        grad_theta = 2*self.epsilon*theta
        return np.concatenate((grad_u, grad_zx, grad_zy, grad_beta, grad_r, grad_delta, grad_theta))

    def constraints(self, x):
        u, zx, zy, beta, r, delta, theta = self.getvars(x)
        delta_gamma = self.q ** self.q * self.gam ** (1-self.q)
        _, Da,_ = hes(u, self.Kx, self.Ky, self.K, beta, delta_gamma, self.gam, self.rho, self.q, self.M)
        cons = np.concatenate((
            (-1/delta_gamma)*(self.K.T@Da@self.K@u - u + self.unoisy) + self.Kx.T@zx + self.Ky.T@zy,
            self.Kx@u - r*np.cos(theta),
            self.Ky@u - r*np.sin(theta),
            zx-delta*np.cos(theta),
            zy-delta*np.sin(theta),
            self.Q@beta - delta
        ))
        return cons

    def jacobianstructure(self):
        Zm = np.zeros((self.M, self.M))
        Zp = np.zeros((self.M, self.P))
        Znp = np.zeros((self.N, self.P))
        Zn = np.zeros((self.M, self.N))
        Znm = np.zeros((self.N, self.M))
        Im = np.eye(self.M)
        # In = np.eye(self.N)
        # Ip = np.eye(self.P)
        Inp = np.ones((self.N,self.P))
        Kx = self.Kx
        Ky = self.Ky
        W = self.K.T@self.K
        jac = np.block([
            [W, Kx.T, Ky.T, Inp, Znm, Znm, Znm],
            [Kx, Zm, Zm, Zp, Im, Zm, Im],
            [Ky, Zm, Zm, Zp, Im, Zm, Im],
            [Zn, Im, Zm, Zp, Zm, Im, Im],
            [Zn, Zm, Im, Zp, Zm, Im, Im],
            [Zn, Zm, Zm, self.Q, Zm, Im, Zm]
        ])
        return jac.nonzero()

    def jacobian(self, x):
        u, zx, zy, beta, r, delta, theta = self.getvars(x)
        delta_gamma = self.q ** self.q * self.gam ** (1-self.q)
        Kw_dev, _, Kw_beta = hes(u, self.Kx, self.Ky, self.K, beta, delta_gamma, self.gam, self.rho, self.q, self.M)
        Zm = np.zeros((self.M, self.M))
        Zp = np.zeros((self.M, self.P))
        Znp = np.zeros((self.N, self.P))
        Zn = np.zeros((self.M, self.N))
        Znm = np.zeros((self.N, self.M))
        Zm_epsilon = Zm + np.diag(1e-1*np.ones(self.M))
        Im = np.eye(self.M)
        In = np.eye(self.N)
        Ip = np.eye(self.P)
        Kx = self.Kx
        Ky = self.Ky
        W = (-1/delta_gamma)*(Kw_dev - In)
        # print(f'{W=}')
        jac = np.block([
            [W, Kx.T, Ky.T, (-1/delta_gamma)*Kw_beta, Znm, Znm, Znm],
            [Kx, Zm, Zm, Zp, np.diag(-np.cos(theta)),
             Zm, np.diag(r*np.sin(theta))],
            [Ky, Zm, Zm, Zp, np.diag(-np.sin(theta)),
             Zm, np.diag(-r*np.cos(theta))],
            [Zn, Im, Zm, Zp, Zm,
                np.diag(-np.cos(theta)), np.diag(delta*np.sin(theta))],
            [Zn, Zm, Im, Zp, Zm,
                np.diag(-np.sin(theta)), np.diag(-delta*np.cos(theta))],
            [Zn, Zm, Zm, self.Q, Zm, -Im, Zm]
        ])
        # row, col = jac.nonzero()
        # return jac[row, col]
        # print(f'N={self.N}, M={self.M},jac={jac.shape}')
        # print('tamaño_jac', jac.shape)
        # return jac.ravel()
        row,col = self.jacobianstructure()
        return jac[row, col]
    
    # def get_nonzero_jacobian(self):
    #     return len(self.jacobianstructure()[0])
    
    def get_number_of_constraints(self):
        return len(self.constraints(np.zeros_like(self.utrue)))
    
    def get_number_of_variables(self):
        return len(self.utrue) + 6*self.M + self.P

    # def hessianstructure(self):
    #     Zm = np.zeros((self.M, self.M))
    #     Zp = np.zeros((self.M, self.P))
    #     Zpn = np.zeros((self.P, self.N))
    #     Znp = np.zeros((self.N, self.P))
    #     Zmp = np.zeros((self.M, self.P))
    #     Zpm = np.zeros((self.P, self.M))
    #     Znm = np.zeros((self.N, self.M))
    #     Zmn = np.zeros((self.M, self.N))
    #     Im = np.eye(self.M)
    #     In = np.eye(self.N)
    #     Ip = np.eye(self.P)
    #     Kx = self.Kx.toarray()
    #     Ky = self.Ky.toarray()
    #     H = np.block([
    #         [In, Znm, Znm, Znp, Znm, Znm, Znm],
    #         [Zmn, Im, Zm, Zmp, Zm, Zm, Zm],
    #         [Zmn, Zm, Im, Zmp, Zm, Zm, Zm],
    #         [Zpn, Zpm, Zpm, Ip, self.Q.T, Zpm, Zpm],
    #         [Zmn, Zm, Zm, self.Q, Im, Im, Im],
    #         [Zmn, Zm, Zm, Zmp, Im, Im, Im],
    #         [Zmn, Zm, Zm, Zmp, Im, Im, Im]
    #     ])
    #     return np.tril(H).nonzero()

    # def hessian(self, x, lagrange, obj_factor):
    #     u, qx, qy, alpha, r, delta, theta = self.getvars(x)
    #     l = np.split(lagrange, [self.N, self.N+self.M, self.N+2*self.M, self.N+3*self.M, self.N+4*self.M, self.N+4*self.M+self.P])

    #     In = np.eye(self.N)
    #     Im = np.eye(self.M)
    #     Ip = np.eye(self.P)
    #     Znm = np.zeros((self.N, self.M))
    #     Zmn = np.zeros((self.M, self.N))
    #     Zm = np.zeros((self.M, self.M))
    #     Zpn = np.zeros((self.P, self.N))
    #     Zpm = np.zeros((self.P, self.M))
    #     Znp = np.zeros((self.N, self.P))
    #     Zmp = np.zeros((self.M, self.P))

    #     D1 = np.diag(l[1]*np.sin(theta))-np.diag(l[2]*np.cos(theta))
    #     D2 = np.diag(l[3]*np.sin(theta))-np.diag(l[4]*np.cos(theta))
    #     D3 = np.diag(l[1]*r*np.cos(theta))+np.diag(l[2]*r*np.sin(theta)) + np.diag(l[3]*delta*np.cos(theta))+np.diag(l[4]*delta*np.sin(theta))
    #     H = np.block([
    #         [In, Znm, Znm, Znp, Znm, Znm, Znm],
    #         [Zmn, self.epsilon*Im, Zm, Zmp, Zm, Zm, Zm],
    #         [Zmn, Zm, self.epsilon*Im, Zmp, Zm, Zm, Zm],
    #         [Zpn, Zpm, Zpm, self.epsilon*Ip, self.pi*self.Q.T, Zpm, Zpm],
    #         [Zmn, Zm, Zm, self.pi*self.Q, self.epsilon*Im, -self.pi*Im, D1],
    #         [Zmn, Zm, Zm, Zmp, -self.pi*Im, self.epsilon*Im, D2],
    #         [Zmn, Zm, Zm, Zmp, D1, D2, D3]
    #     ])
    #     # print(np.linalg.cond(H))
    #     row,col = self.hessianstructure()
    #     return obj_factor*H[row,col]
