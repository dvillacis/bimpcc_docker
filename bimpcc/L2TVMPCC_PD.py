import numpy as np
from scipy.sparse import bmat,identity,diags
from scipy.sparse.linalg import svds,spilu
from .AbstractMPCC import AbstractMPCC
from .img_utils import compute_condition_number

class L2TVMPCC_PD(AbstractMPCC):

    def __init__(self, pi, A, Kx, Ky, Q, utrue, unoisy, epsilon=1e-4, gamma=0):
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
        self.A = A
        self.At = A.transpose()
        self.AtA = self.At@A
        self.Kx = Kx
        self.Kxt = Kx.transpose()
        self.Ky = Ky
        self.Kyt = Ky.transpose()
        self.Q = Q
        self.M, self.N = Kx.shape
        self.I = identity(self.M, format='coo')
        self.P = Q.shape[1]
        self.utrue = utrue.ravel()
        self.unoisy = unoisy.ravel()

        self.epsilon = epsilon
        self.gamma = gamma

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
        qx = x[self.N:self.N+self.M]
        qy = x[self.N+self.M:self.N+2*self.M]
        alpha = x[self.N+2*self.M:self.N+2*self.M+self.P]
        r = x[self.N+2*self.M+self.P:self.N+3*self.M+self.P]
        delta = x[self.N+3*self.M+self.P:self.N+4*self.M+self.P]
        theta = x[self.N+4*self.M+self.P:self.N+5*self.M+self.P]
        return u, qx, qy, alpha, r, delta, theta

    def complementarity(self, x):
        u, qx, qy, alpha, r, delta, theta = self.getvars(x)
        # c = np.dot(r, (self.Q@alpha)-delta)
        return np.dot(r, (self.Q@alpha)-delta)
    
    def min_complementarity(self, x):
        u, qx, qy, alpha, r, delta, theta = self.getvars(x)
        min_comp = np.minimum(r, (self.Q@alpha)-delta)
        return np.linalg.norm(min_comp)

    def objective(self, x):
        u, qx, qy, alpha, r, delta, theta = self.getvars(x)
        return 0.5*np.linalg.norm(u-self.utrue)**2 + self.pi*self.complementarity(x) + self.epsilon*np.linalg.norm(qx)**2 + self.epsilon*np.linalg.norm(qy)**2 + self.epsilon*np.linalg.norm(alpha)**2 + self.epsilon*np.linalg.norm(r)**2 + self.epsilon*np.linalg.norm(delta)**2

    def gradient(self, x):
        u, qx, qy, alpha, r, delta, theta = self.getvars(x)
        grad_u = u-self.utrue
        grad_qx = 2*self.epsilon*qx
        grad_qy = 2*self.epsilon*qy
        grad_alpha = self.pi*(self.Q.T@r) + 2*self.epsilon*alpha
        grad_r = self.pi*((self.Q@alpha)-delta) + 2*self.epsilon*r
        grad_delta = -self.pi*(r) + 2*self.epsilon*delta
        grad_theta = np.zeros(self.M)
        return np.concatenate((grad_u, grad_qx, grad_qy, grad_alpha, grad_r, grad_delta, grad_theta))

    def constraints(self, x):
        u, qx, qy, alpha, r, delta, theta = self.getvars(x)
        cons = np.concatenate((
            self.AtA@u - self.At@self.unoisy + self.Kxt@qx + self.Kyt@qy,
            self.Kx@u - r*np.cos(theta),
            self.Ky@u - r*np.sin(theta),
            qx-delta*np.cos(theta),
            qy-delta*np.sin(theta),
            self.Q@alpha - delta
        ))
        return cons

    def jacobianstructure(self):
        jac = bmat([
            [self.AtA,self.Kxt,self.Kyt,None,None,None,None],
            [self.Kx,None,None,None,self.I,None,self.I],
            [self.Ky,None,None,None,self.I,None,self.I],
            [None,self.I,None,None,None,self.I,self.I],
            [None,None,self.I,None,None,self.I,self.I],
            [None,None,None,self.Q,None,self.I,None]
        ])
        return jac.row, jac.col

    def jacobian(self, x):
        u, qx, qy, alpha, r, delta, theta = self.getvars(x)

        ct = np.cos(theta)
        st = np.sin(theta)

        ct = np.where(ct == 0, 1e-10, ct)
        st = np.where(st == 0, 1e-10, st)
        delta = np.where(delta == 0, 1e-10, delta)
        r = np.where(r == 0, 1e-10, r)

        jac = bmat([
            [self.AtA,self.Kxt,self.Kyt,None,None,None,None],
            [self.Kx,None,None,None,diags(-ct),None,diags(r*st)],
            [self.Ky,None,None,None,diags(-st),None,diags(-r*ct)],
            [None,self.I,None,None,None,diags(-ct),diags(delta*st)],
            [None,None,self.I,None,None,diags(-st),diags(-delta*ct)],
            [None,None,None,self.Q,None,-self.I,None]
        ])

        # condition_number = compute_condition_number(jac.toarray())
        # print(f"Jac Condition number: {condition_number}")
        # print(jac.size)
        return jac.data
    
    def hessianstructure(self):
        hess = bmat([
            [self.I,None,None,None,None,None,None],
            [None,self.I,None,None,None,None,None],
            [None,None,self.I,None,None,None,None],
            [None,None,None,diags(np.ones(self.P)),None,None,None],
            [None,None,None,self.Q,self.I,None,None],
            [None,None,None,None,self.I,self.I,None],
            [None,None,None,None,self.I,self.I,self.I]
        ])
        c1 = compute_condition_number(hess.toarray())
        print(f"Structure Hess Condition number: {c1}")
        # print(f'structue: {hess.size}')
        # print(hess.toarray())
        return hess.row, hess.col
    
    def hessian(self, x, lagrange, obj_factor):
        u, qx, qy, alpha, r, delta, theta = self.getvars(x)
        l = np.where(lagrange <= 1e-9, 1e-9, lagrange)
        l = np.split(l, [self.N, self.N+self.M, self.N+2*self.M, self.N+3*self.M, self.N+4*self.M, self.N+4*self.M+self.P])

        st = np.sin(theta)
        ct = np.cos(theta)
        ct = np.where(ct <= 1e-9 , 1e-10, ct)
        st = np.where(st <= 1e-9, 1e-10, st)
        delta = np.where(delta <= 1e-9, 1e-9, delta)
        r = np.where(r <= 1e-9, 1e-9, r)
        if obj_factor == 0:
            obj_factor += 1e-9

        # print(f'st: {st.shape}, ct: {ct.shape}, delta: {delta.shape}, r: {r.shape}, l: {l[1].shape}, {l[2].shape}, {l[3].shape}, {l[4].shape}')
        # print(f'st: {st}, ct: {ct}, delta: {delta}, r: {r}, l: {l[1]}, {l[2]}, {l[3]}, {l[4]}') 

        D1 = obj_factor*self.I
        D2 = 2*obj_factor*self.epsilon*self.I
        D3 = obj_factor*diags(2*self.epsilon*np.ones(self.P))

        #safeguards
        d4 = l[1]*st-l[2]*ct
        d4 = np.where(d4 <=1e-9, 1e-9, d4)
        d5 = l[3]*st-l[4]*ct
        d5 = np.where(d5 <=1e-9, 1e-9, d5)
        d6 = l[1]*r*ct+l[2]*r*st+l[3]*delta*ct+l[4]*delta*st
        d6 = np.where(d6 <=1e-9, 1e-9, d6)

        D4 = diags(d4)
        D5 = diags(d5)
        D6 = diags(d6)

        hess = bmat([
            [D1,None,None,None,None,None,None],
            [None,D2,None,None,None,None,None],
            [None,None,D2,None,None,None,None],
            [None,None,None,D3,None,None,None],
            [None,None,None,obj_factor*self.pi*self.Q,D2,None,None],
            [None,None,None,None,-obj_factor*self.pi*self.I,D2,None],
            [None,None,None,None,D4,D5,D6]
        ])
        # print(f'hess: {hess.size}')
        # print(f'{(l[1]*st-l[2]*ct)=}')
        # if hess.size == 250:
        #     print(f'st: {st}, ct: {ct}, delta: {delta}, r: {r}, l: {l[1]}, {l[2]}, {l[3]}, {l[4]}') 
        #     print(f'{D3}')

        Atest = bmat([
            [0.5*D2,None,None,None,None,None,None],
            [None,D2,None,None,None,None,None],
            [None,None,D2,None,None,None,None],
            [None,None,None,D3,None,None,None],
            # [None,None,None,1e-10*obj_factor*self.pi*self.Q,D2,None,None],
            [None,None,None,None,-1e-10*obj_factor*self.pi*self.I,D2,None],
            [None,None,None,None,D4,D5,D2]])

        ctest = compute_condition_number(Atest.toarray())
        c1 = compute_condition_number(D1.toarray())
        c2 = compute_condition_number(D2.toarray())
        c3 = compute_condition_number(D3.toarray())
        c4 = compute_condition_number(D4.toarray())
        c5 = compute_condition_number(D5.toarray())
        c6 = compute_condition_number(D6.toarray())
        c7 = compute_condition_number(hess.toarray())
        # print(f'Atest Condition number: {ctest}, obj_factor: {obj_factor}, pi: {self.pi}')
        print(f"Hess Condition number: {c7}")
        # print(f"Hess Condition number: {c1}, {c2}, {c3}, {c4}, {c5}, {c6}, {c7}")
        # print(f'd4: {d4}, d5: {d5}, d6: {d6}')
        # print(f'st: {st}, ct: {ct}, delta: {delta}, r: {r}, l: {l[1]}, {l[2]}, {l[3]}, {l[4]}')

        return hess.data

    def get_number_of_constraints(self):
        return len(self.constraints(np.zeros_like(self.utrue)))
    
    def get_number_of_variables(self):
        return len(self.utrue) + 6*self.M + self.P


