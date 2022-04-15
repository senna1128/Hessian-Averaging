
import numpy as np
import random
from time import time

import os
os.chdir('/.../code')


### Define Weight Sequences
def WeightPower(t, power = 1):
    return (1-1/(t+1))**power

def WeightLogPower(t, power = 1):
    return ((1-1/(t+1))**(power*np.log(t+2)))*(1/(t+1)**(power*np.log(1+1/(t+2))))

### Define Oracle Noises
def Gaussian_Noise(matrix, d, scale = 1):
    return matrix + np.random.normal(0,scale,(d,d))

def Exp_Noise(matrix, d, scale = 1):
    return matrix + np.random.exponential(scale,(d,d)) - scale 

### Define Sketching/Subsampling functions
def Gaussian_Sketch(n,matrix,sketch_size,nnz=None):
    S = np.random.randn(sketch_size,n)
    SA = S @ matrix
    return SA.T@SA/sketch_size

def Sub_Sampling(n,matrix,sub_size,nnz=None):
    Id_Sub_Set = np.zeros((n,1))
    Id_Sub_Set[np.random.choice(n,sub_size,replace=False)] = 1.0
    SA = Id_Sub_Set*matrix
    return SA.T@SA/sub_size*n

def Sparse_Sketch(n,matrix,sketch_size,nnz=None):
    S = np.zeros((sketch_size,n))
    S[np.random.choice(sketch_size,n),np.arange(n)]=np.random.choice(np.array([-1,1], dtype=np.float64), size=n)
    SA = S @ matrix
    return SA.T@SA

def SparRad_Sketch(n,matrix,sketch_size,nnz=None):
    if nnz is None:
        nnz = 0.1
    d_tilde = int(nnz*matrix.shape[1])
    row_index = np.repeat(np.arange(sketch_size),d_tilde)
    column_index = np.random.choice(n,sketch_size*d_tilde)
    values = np.random.choice(np.array([-1,1],dtype=np.float64),sketch_size*d_tilde)
    S = np.zeros((sketch_size,n))
    S[row_index,column_index] = values
    SA = S @ matrix
    return SA.T@SA*n/(sketch_size*d_tilde)
    

Weight = {'power':WeightPower, 'log_power':WeightLogPower}

Sto_Hess = {'gaussian_noise':Gaussian_Noise, 'exp_noise':Exp_Noise}

Sketch_Func = {'Gaussian':Gaussian_Sketch, 'CountSketch':Sparse_Sketch,\
               'Subsampled':Sub_Sampling,'LESS-uniform':SparRad_Sketch}


### Data Generating
class DataGenerate_HighCond:
    def __init__(self, n, d, lambd, kap=1, Rep=10):
        self.lambd, self.Rep = lambd, Rep
        self.IdCond, self.IdReal = 'true', 'Unreal'
        self.kap = kap
        # generate data
        np.random.seed(2022)
        U, _, _ = np.linalg.svd(np.random.randn(n,d),full_matrices=False)
        Sigma = np.array([j for j in np.linspace(1,d**kap,d)])
        self.Dat = U@np.diag(Sigma)
        x_under = 1./np.sqrt(d)*np.random.randn(d,1)
        Prob = 1./(1+np.exp(-self.Dat@x_under))
        self.Resp = 2*np.random.binomial(1, p=Prob)-1        

class DataGenerate_HighCoher:
    def __init__(self, n, d, lambd, kap=1, Rep=10):
        self.lambd, self.Rep = lambd, Rep
        self.IdCond, self.IdReal = 'false', 'Unreal'
        self.kap = kap
        # generate data
        np.random.seed(2022)
        g = np.tile(np.random.gamma(1/2,2,n),(d,1)).T
        U, _, _ = np.linalg.svd(np.random.randn(n,d)/np.sqrt(g), full_matrices=False)
        Sigma = np.array([j for j in np.linspace(1,d**kap,d)])
        self.Dat = U@np.diag(Sigma)
        x_under = 1./np.sqrt(d)*np.random.randn(d,1)
        Prob = 1./(1+np.exp(-self.Dat@x_under))
        self.Resp = 2*np.random.binomial(1, p=Prob)-1
       

### Problem Solver
class LogisticRegression:    
    def __init__(self, A, b, lambd):
        self.A, self.b, self.lambd = A, b, lambd
        self.n, self.d = A.shape
        np.random.seed(2022)
        random.seed(2022)
        self.x_0 = 1./np.sqrt(self.d)*np.random.randn(self.d,1)
#        self.x_0 = 0*np.ones((self.d,1))
        
    def logistic_loss(self, x):
        return np.log(1+np.exp(-self.b*self.A@x)).mean()+self.lambd/2*(x**2).sum()
        
    def grad(self, x):
        return -1./self.n*self.A.T@(self.b*1./(1+np.exp(self.b*self.A@x)))+self.lambd*x
        
    def Hess(self, x):
        v = np.exp(self.b*self.A@x)
        D = (v/(1+v)**2)/self.n
        return self.A.T@(D*self.A)+self.lambd*np.identity(self.d)

    def sqrt_hess(self, x):
        v = np.exp(self.b*self.A@x)
        D = np.sqrt(v)/(1+v)/np.sqrt(self.n)
        return D*self.A

    def line_search(self, x, f_x, NewDir, Del, beta=0.3, rho=0.8):
        mu = 1
        x_1 = x + mu*NewDir
        while self.logistic_loss(x_1) > f_x + beta*mu*Del:
            mu = mu*rho
            x_1 = x + mu*NewDir
        return mu

    def solve_exactly(self, Max_Iter=10**3, EPS=1e-10):
        # use Newton method to solve exactly
        x_0, grad_x_0 = self.x_0, self.grad(self.x_0)
        eps, t = np.linalg.norm(grad_x_0), 0
        while eps >= EPS and t <= Max_Iter:
            Hess_x_0 = self.Hess(x_0)
            NewDir = -np.linalg.inv(Hess_x_0)@grad_x_0
            Inner = (grad_x_0*NewDir).sum()
            Alp = self.line_search(x_0,self.logistic_loss(x_0),NewDir,Inner)
            x_0 = x_0 + Alp*NewDir
            grad_x_0 = self.grad(x_0)
            eps, t = np.linalg.norm(grad_x_0), t+1
        self.x_true = x_0 
        self.Hess_x_true = self.Hess(x_0)
        return self.x_true, self.Hess_x_true

    def BFGS(self,Max_Iter=10**3,EPS = 1e-8):
        # implement BFGS
        Xarray, Losses = [], []
        x_0, grad_x_0 = self.x_0, self.grad(self.x_0)
        B_inv = np.identity(self.d)
        eps, t = np.linalg.norm(grad_x_0), 0
        Xarray.append(x_0)
        Losses.append(self.logistic_loss(x_0))
        
        start = time()
        while eps>=EPS and t<= Max_Iter:
            NewDir = -B_inv@grad_x_0
            Inner = (grad_x_0*NewDir).sum() 
            Alp = self.line_search(x_0,Losses[-1],NewDir,Inner)
            s = Alp*NewDir
            x_0 = x_0 + s
            grad_x_0_ = self.grad(x_0)
            y = grad_x_0_ - grad_x_0 
            grad_x_0 = grad_x_0_.copy()
            eps, t = np.linalg.norm(grad_x_0), t+1
            Xarray.append(x_0)
            Losses.append(self.logistic_loss(x_0))
            # update B
            sy_inner, sy_outer, ss_outer = (s*y).sum(), s@y.T, s@s.T
            B_1 = (sy_inner+(y*(B_inv@y)).sum())/sy_inner**2 * ss_outer
            b_2 = B_inv@sy_outer.T
            B_2 = (b_2+b_2.T)/sy_inner
            B_inv = B_inv + B_1 - B_2
        Time = time()-start
        Xarray = np.hstack(Xarray)-self.x_true
        Err = np.sqrt(((self.Hess_x_true@Xarray)*Xarray).sum(axis=0))
        return Err, Losses, Time, Xarray    

    def sto_oracle_Newton(self,ora_set='gaussian_noise',scale=0,Max_Iter=10**3,EPS=1e-8):
        # implement weighted stochastic Newton (oracle noise)
        Xarray, Losses = [], []
        x_0, grad_x_0 = self.x_0, self.grad(self.x_0)
        eps, t = np.linalg.norm(grad_x_0), 0
        Xarray.append(x_0)
        Losses.append(self.logistic_loss(x_0))
        
        start = time()
        while eps>=EPS and t<=Max_Iter:
            H_hat_x_0 = Sto_Hess[ora_set](self.Hess(x_0), self.d, scale)
            if scale == 0:
                NewDir = -np.linalg.inv(H_hat_x_0)@grad_x_0
                Inner = (grad_x_0*NewDir).sum()
            else:
                if np.linalg.det(H_hat_x_0)!=0:
                    NewDir = -np.linalg.inv(H_hat_x_0)@grad_x_0
                    Inner = (grad_x_0*NewDir).sum()
                    if Inner > 0:
                        NewDir = -grad_x_0.copy()
                        Inner = (grad_x_0*NewDir).sum()
                else:
                    NewDir = -grad_x_0.copy()
                    Inner = (grad_x_0*NewDir).sum()
            Alp = self.line_search(x_0,Losses[-1],NewDir,Inner)
            x_0 = x_0 + Alp*NewDir
            grad_x_0 = self.grad(x_0)
            eps, t = np.linalg.norm(grad_x_0), t+1
            Xarray.append(x_0)
            Losses.append(self.logistic_loss(x_0))
        Time = time()-start
        Xarray = np.hstack(Xarray)-self.x_true
        Err = np.sqrt(((self.Hess_x_true@Xarray)*Xarray).sum(axis=0))
        return Err, Losses, Time, Xarray

    def sto_weight_oracle_Newton(self,wei_set='power',power=1,ora_set='gaussian_noise',scale=0,Max_Iter=10**3,EPS=1e-8):
        # implement weighted stochastic Newton (oracle noise)
        Xarray, Losses = [], []
        x_0, grad_x_0, w_H_0 = self.x_0, self.grad(self.x_0), np.identity(self.d)
        eps, t = np.linalg.norm(grad_x_0), 0
        Xarray.append(x_0)
        Losses.append(self.logistic_loss(x_0))
        
        start = time()
        while eps>=EPS and t<=Max_Iter:
            H_hat_x_0 = Sto_Hess[ora_set](self.Hess(x_0), self.d, scale)
            ratio = Weight[wei_set](t,power)
            w_H_0 = ratio*w_H_0 + (1-ratio)*H_hat_x_0
            if scale == 0:
                NewDir = -np.linalg.inv(w_H_0)@grad_x_0
                Inner = (grad_x_0*NewDir).sum()
            else:
                if np.linalg.det(w_H_0)!=0:
                    NewDir = -np.linalg.inv(w_H_0)@grad_x_0
                    Inner = (grad_x_0*NewDir).sum()
                    if Inner > 0:
                        NewDir = -grad_x_0.copy()
                        Inner = (grad_x_0*NewDir).sum()
                else:
                    NewDir = -grad_x_0.copy()
                    Inner = (grad_x_0*NewDir).sum()
            Alp = self.line_search(x_0,Losses[-1],NewDir,Inner)
            x_0 = x_0 + Alp*NewDir
            grad_x_0 = self.grad(x_0)
            eps, t = np.linalg.norm(grad_x_0), t+1
            Xarray.append(x_0)
            Losses.append(self.logistic_loss(x_0))
        Time = time()-start
        Xarray = np.hstack(Xarray)-self.x_true
        Err = np.sqrt(((self.Hess_x_true@Xarray)*Xarray).sum(axis=0))
        return Err, Losses, Time, Xarray

    def sketch_Newton(self,sketch_size,sketch_method='Gaussian',nnz=None,Max_Iter=10**3,EPS=1e-8):
        # implement stochastic Newton (sketching/subsampling)
        Xarray, Losses = [], []
        x_0, grad_x_0 = self.x_0, self.grad(self.x_0)
        eps, t = np.linalg.norm(grad_x_0), 0
        Xarray.append(x_0)
        Losses.append(self.logistic_loss(x_0))
        
        start = time()                
        while eps>=EPS and t<=Max_Iter:
            H_hat_x_0 = Sketch_Func[sketch_method](self.n,self.sqrt_hess(x_0),sketch_size,nnz=nnz)+ self.lambd*np.identity(self.d)
            NewDir = -np.linalg.inv(H_hat_x_0)@grad_x_0
            Inner = (grad_x_0*NewDir).sum() 
            Alp = self.line_search(x_0,Losses[-1],NewDir,Inner)
            x_0 = x_0 + Alp*NewDir
            grad_x_0 = self.grad(x_0)
            eps, t = np.linalg.norm(grad_x_0), t+1
            Xarray.append(x_0)
            Losses.append(self.logistic_loss(x_0))
        Time = time()-start
        Xarray = np.hstack(Xarray)-self.x_true
        Err = np.sqrt(((self.Hess_x_true@Xarray)*Xarray).sum(axis=0))
        return Err, Losses, Time, Xarray
    
    def sto_weight_Sket_Newton(self,sketch_size,wei_set='power',power=1,sketch_method='Gaussian',nnz=None,Max_Iter=10**3,EPS=1e-8):
        # implement weighted stochastic Newton (sketching/subsampling)
        Xarray, Losses = [], []
        x_0, grad_x_0, w_H_0 = self.x_0, self.grad(self.x_0), np.identity(self.d)
        eps, t = np.linalg.norm(grad_x_0), 0
        Xarray.append(x_0)
        Losses.append(self.logistic_loss(x_0))
        
        start = time()
        while eps>=EPS and t<=Max_Iter:
            H_hat_x_0 = Sketch_Func[sketch_method](self.n,self.sqrt_hess(x_0),sketch_size,nnz=nnz)+ self.lambd*np.identity(self.d)
            ratio = Weight[wei_set](t,power)
            w_H_0 = ratio*w_H_0 + (1-ratio)*H_hat_x_0
            NewDir = -np.linalg.inv(w_H_0)@grad_x_0
            Inner = (grad_x_0*NewDir).sum()
            Alp = self.line_search(x_0,Losses[-1],NewDir,Inner)
            x_0 = x_0 + Alp*NewDir
            grad_x_0 = self.grad(x_0)
            eps, t = np.linalg.norm(grad_x_0), t+1
            Xarray.append(x_0)
            Losses.append(self.logistic_loss(x_0))
        Time = time()-start
        Xarray = np.hstack(Xarray)-self.x_true
        Err = np.sqrt(((self.Hess_x_true@Xarray)*Xarray).sum(axis=0))
        return Err, Losses, Time, Xarray
       





    
    

