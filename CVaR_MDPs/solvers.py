import numpy as np
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False


class riskneutral_PI:

    def __init__(self, gamma, tol=10**-6):

        self.gamma = gamma  
        self.tol = tol
    
    def solve(self, mdp, v_0):

        I = np.eye(mdp.n)
        idx = np.arange(0, mdp.n)
        #first iteration
        ##policy improvement
        pi = np.argmin(mdp.c + self.gamma * np.dot(mdp.P, v_0), axis=0)
        
        P_pi = mdp.P[pi, idx, :]
        c_pi = mdp.c[pi, idx]
        J_pi = I - self.gamma*P_pi

        ##policy evaluation
        v = np.linalg.solve(J_pi, c_pi)
 
        ##compute the Bellman residual
        res = v - np.min(mdp.c + self.gamma * np.dot(mdp.P, v), axis=0)

        #enter in the loop 
        while (np.linalg.norm(res, np.inf))>self.tol:
            
            ##policy improvement
            pi = np.argmin(mdp.c + self.gamma * np.dot(mdp.P, v), axis=0)
            
            P_pi = mdp.P[pi, idx, :]
            c_pi = mdp.c[pi, idx]
            J_pi = I - self.gamma*P_pi

            ##policy evaluation
            v = np.linalg.solve(J_pi, c_pi)
    
            ##compute the Bellman residual
            res = v - np.min(mdp.c + self.gamma * np.dot(mdp.P, v), axis=0)

        return v, pi

class SNMI:

    def __init__(self, gamma, alpha, tol1=10**-6, tol2=10**-6):

        self.gamma = gamma
        self.alpha = alpha
        self.tol1 = tol1
        self.tol2 = tol2
        
    def solve(self, mdp, v_0, verbose=True):

        res_ = []

        #iteration counter
        iter = 0

        #pre-compute the quantities for the LPs which remain unchanged
        A = matrix(np.ones((1, mdp.n)))
        b = matrix(np.ones(1, ))
        G = matrix(np.concatenate((-np.eye(mdp.n, ), np.eye(mdp.n, ))))
        #initialization of the 3D probability tensor
        M_pi_k = np.zeros((mdp.m, mdp.n, mdp.n))

        #extract M_pi_0
        for a in range(mdp.m):
            P = mdp.P[a, :, :]
            for s in range(mdp.n):
                c = matrix(-v_0)
                h = matrix(np.concatenate((np.zeros((mdp.n, )), (P[s, :]/self.alpha).transpose())))
                sol = solvers.lp(c, G, h, A, b)
                M_pi_k[a, s, :] = np.array(sol["x"]).transpose()

        #compute the residual
        res = v_0 - np.min(mdp.c + self.gamma * np.dot(M_pi_k, v_0), axis=0)
        res_.append(np.linalg.norm(res, np.inf))

        if verbose:
            print("iteration: {}, inf-norm residual:{}".format(iter, res_[-1]))
        
        v = np.copy(v_0)

        while (res_[-1] > self.tol1):

            v = self.inner_solver(M_pi_k, mdp.c, v, mdp.n)
            iter += 1
        
            #extract M_pi_k
            for a in range(mdp.m):
                P = mdp.P[a, :, :]
                for s in range(mdp.n):
                    c = matrix(-v)
                    h = matrix(np.concatenate((np.zeros((mdp.n, )), (P[s, :]/self.alpha).transpose())))
                    sol = solvers.lp(c, G, h, A, b)
                    M_pi_k[a, s, :] = np.array(sol["x"]).transpose()

            #compute the residual
            res = v - np.min(mdp.c + self.gamma * np.dot(M_pi_k, v), axis=0)
            res_.append(np.linalg.norm(res, np.inf))

            if verbose:
                print("iteration: {}, inf-norm residual:{}".format(iter, res_[-1]))
            
        return v, res_

    def inner_solver(self, P, c, v_0, n):

        I = np.eye(n)
        idx = np.arange(0, n)
        
        #first iteration
        ##policy improvement
        pi = np.argmin(c + self.gamma * np.dot(P, v_0), axis=0)
        
        P_pi = P[pi, idx, :]
        c_pi = c[pi, idx]
        J_pi = I - self.gamma*P_pi

        ##policy evaluation
        v = np.linalg.solve(J_pi, c_pi)
 
        ##compute the Bellman residual
        res = v - np.min(c + self.gamma * np.dot(P, v), axis=0)

        #enter in the loop 
        while (np.linalg.norm(res, np.inf))>self.tol2:
            
            ##policy improvement
            pi = np.argmin(c + self.gamma * np.dot(P, v), axis=0)
            
            P_pi = P[pi, idx, :]
            c_pi = c[pi, idx]
            J_pi = I - self.gamma*P_pi

            ##policy evaluation
            v = np.linalg.solve(J_pi, c_pi)
    
            ##compute the Bellman residual
            res = v - np.min(c + self.gamma * np.dot(P, v), axis=0)

        return v




class SNMIII:

    def __init__(self, gamma, alpha, tol1=10**-6):

        self.gamma = gamma
        self.alpha = alpha
        self.tol1 = tol1
        
    def solve(self, mdp, v_0, verbose=True):
        
        res_ = []
        #initialize the iteration counter
        iter = 0

        #pre-compute the matrices for LPs that remain unchanged
        A = matrix(np.ones((1, mdp.n)))
        b = matrix(np.ones(1, ))
        G = matrix(np.concatenate((-np.eye(mdp.n, ), np.eye(mdp.n, ))))

        #initialization of the 3D tensor for the transition probabilities
        M_pi_k = np.zeros((mdp.m, mdp.n, mdp.n))

        #extract M_pi_0
        for a in range(mdp.m):
            P = mdp.P[a, :, :]
            for s in range(mdp.n):
                c = matrix(-v_0)
                h = matrix(np.concatenate((np.zeros((mdp.n, )), (P[s, :]/self.alpha).transpose())))
                sol = solvers.lp(c, G, h, A, b)
                M_pi_k[a, s, :] = np.array(sol["x"]).transpose()

        #compute the residual
        res = v_0 - np.min(mdp.c + self.gamma * np.dot(M_pi_k, v_0), axis=0)
        res_.append(np.linalg.norm(res, np.inf))

        if verbose:
            print("iteration: {}, inf-norm residual: {}".format(iter, res_[-1]))

        v = np.copy(v_0)
        
        while (res_[-1]>self.tol1):

            v = self.inner_solver(M_pi_k, mdp.c, v, mdp.n)
            iter += 1

            #extract M_pi_k
            for a in range(mdp.m):
                P = mdp.P[a, :, :]
                for s in range(mdp.n):
                    c = matrix(-v)
                    h = matrix(np.concatenate((np.zeros((mdp.n, )), (P[s, :]/self.alpha).transpose())))
                    sol = solvers.lp(c, G, h, A, b)
                    M_pi_k[a, s, :] = np.array(sol["x"]).transpose()

            #compute the residual
            res = v - np.min(mdp.c + self.gamma * np.dot(M_pi_k, v), axis=0)
            res_.append(np.linalg.norm(res, np.inf))

            if verbose:
                print("iteration: {}, inf-norm residual: {}".format(iter, res_[-1]))

        return v, res_


    def inner_solver(self, P, c, v_0, n):

        I = np.eye(n)
        idx = np.arange(0, n)
        #first iteration
        ##policy improvement
        pi = np.argmin(c + self.gamma * np.dot(P, v_0), axis=0)
        
        P_pi = P[pi, idx, :]
        c_pi = c[pi, idx]
        J_pi = I - self.gamma*P_pi

        ##policy evaluation
        v = np.linalg.solve(J_pi, c_pi)
 
        return v



class SNMII:

    def __init__(self, gamma, alpha, tol1=10**-6, tol2=10**-6):

        self.gamma = gamma
        self.alpha = alpha
        self.tol1 = tol1
        self.tol2 = tol2

    def solve(self, mdp, v_0, verbose=True, warmstarting=True):

        res_ = []
        #iteration counter
        iter = 0

        #pre-computing the elements in LP which are not changing 
        A = matrix(np.ones((1, mdp.n)))
        b = matrix(np.ones(1, ))
        G = matrix(np.concatenate((-np.eye(mdp.n, ), np.eye(mdp.n, ))))
        
        #initialization
        M_pi_k = np.zeros((mdp.m, mdp.n, mdp.n))

        #compute the initial policy
        for a in range(mdp.m):
            P = mdp.P[a, :, :]
            for s in range(mdp.n):
                c = matrix(-v_0)
                h = matrix(np.concatenate((np.zeros((mdp.n, )), (P[s, :]/self.alpha).transpose())))
                sol = solvers.lp(c, G, h, A, b)
                M_pi_k[a, s, :] = np.array(sol["x"]).transpose()

        #compute the residual
        res = v_0 - np.min(mdp.c + self.gamma * np.dot(M_pi_k, v_0), axis=0)
        res_.append(np.linalg.norm(res, np.inf))

        if verbose:
            print("iteration: {}, inf-norm residual: {}".format(iter, res_[-1]))
        
        #extraction of greedy policy
        pi = np.argmin(mdp.c + self.gamma * np.dot(M_pi_k, v_0), axis=0) 

        M_pi_ell = np.zeros((mdp.n, mdp.n))
        
        idx = np.arange(0, mdp.n)
        I = np.eye(mdp.n)

        #warm-starting the inner Newton loop
        if warmstarting:
            v_inner = np.copy(v_0)
        else:
            v_inner = np.random.randn(mdp.n)
        
        while (res_[-1])>self.tol1:
            
            c_pi = mdp.c[pi, idx]
            P_pi = mdp.P[pi, idx, :]

            inner_iter = 0

            if warmstarting:
                M_pi_ell = M_pi_k[pi, idx, :]
            else:
                for s in range(mdp.n):
                    c = matrix(-v_inner)
                    h = matrix(np.concatenate((np.zeros((mdp.n, )), (P_pi[s, :]/self.alpha).transpose())))
                    sol = solvers.lp(c, G, h, A, b)
                    M_pi_ell[s, :] = np.array(sol["x"]).transpose()

            res_inner = 10**2*np.ones(mdp.n)#v_inner - c_pi - self.gamma * np.dot(M_pi_ell, v_inner)
            
            while (np.linalg.norm(res_inner, np.inf))>self.tol2:

                v_inner = np.linalg.solve(I - self.gamma * M_pi_ell, c_pi)
                inner_iter += 1

                for s in range(mdp.n):
                    c = matrix(-v_inner)
                    h = matrix(np.concatenate((np.zeros((mdp.n, )), (P_pi[s, :]/self.alpha).transpose())))
                    sol = solvers.lp(c, G, h, A, b)
                    M_pi_ell[s, :] = np.array(sol["x"]).transpose()

                res_inner = v_inner - c_pi - self.gamma * np.dot(M_pi_ell, v_inner)

                if verbose:
                    print("inner iteration: {}, linear residual: {}".format(inner_iter, np.linalg.norm(res_inner, np.inf)))

            v = np.copy(v_inner)    
            #compute the initial policy
            for a in range(mdp.m):
                P = mdp.P[a, :, :]
                for s in range(mdp.n):
                    
                    c = matrix(-v)
                    h = matrix(np.concatenate((np.zeros((mdp.n, )), (P[s, :]/self.alpha).transpose())))
                    sol = solvers.lp(c, G, h, A, b)
                    M_pi_k[a, s, :] = np.array(sol["x"]).transpose()

            #compute the residual
            res = v - np.min(mdp.c + self.gamma * np.dot(M_pi_k, v), axis=0)
            res_.append(np.linalg.norm(res, np.inf))

            iter += 1

            if verbose:
                print("iteration: {}, inf-norm residual: {}".format(iter, res_[-1]))
            
            pi = np.argmin(mdp.c + self.gamma * np.dot(M_pi_k, v), axis=0) 

            if warmstarting:
                v_inner = np.copy(v)
            else:
                v_inner = np.random.randn(mdp.n)

        return v, res_


class CVaR_OPI:

    def __init__(self, gamma, alpha, tol1=10**-6, max_inner_iter=3):

        self.gamma = gamma
        self.alpha = alpha
        self.tol1 = tol1
        self.max_inner_iter = max_inner_iter

    def solve(self, mdp, v_0, verbose=True):

        res_ = []
        #iteration counter
        iter = 0

        #pre-computing the elements in LP which are not changing 
        A = matrix(np.ones((1, mdp.n)))
        b = matrix(np.ones(1, ))
        G = matrix(np.concatenate((-np.eye(mdp.n, ), np.eye(mdp.n, ))))
        
        #initialization
        M_pi_k = np.zeros((mdp.m, mdp.n, mdp.n))

        #compute the initial policy
        for a in range(mdp.m):
            P = mdp.P[a, :, :]
            for s in range(mdp.n):
                c = matrix(-v_0)
                h = matrix(np.concatenate((np.zeros((mdp.n, )), (P[s, :]/self.alpha).transpose())))
                sol = solvers.lp(c, G, h, A, b)
                M_pi_k[a, s, :] = np.array(sol["x"]).transpose()

        #compute the residual
        res = v_0 - np.min(mdp.c + self.gamma * np.dot(M_pi_k, v_0), axis=0)
        res_.append(np.linalg.norm(res, np.inf))

        if verbose:
            print("iteration: {}, inf-norm residual: {}".format(iter, res_[-1]))
        
        #extraction of greedy policy
        pi = np.argmin(mdp.c + self.gamma * np.dot(M_pi_k, v_0), axis=0) 

        M_pi_ell = np.zeros((mdp.n, mdp.n))
        
        idx = np.arange(0, mdp.n)
        I = np.eye(mdp.n)

        #warm-starting the inner Newton loop
        v = np.copy(v_0)
        
        while (res_[-1])>self.tol1:
            
            c_pi = mdp.c[pi, idx]
            P_pi = mdp.P[pi, idx, :]

            inner_iter = 0

            M_pi_ell = M_pi_k[pi, idx, :]

            v =  c_pi + self.gamma * np.dot(M_pi_ell, v)

            inner_iter+=1
            
            while inner_iter<self.max_inner_iter:

                for s in range(mdp.n):
                
                    c = matrix(-v)
                    h = matrix(np.concatenate((np.zeros((mdp.n, )), (P_pi[s, :]/self.alpha).transpose())))
                    sol = solvers.lp(c, G, h, A, b)
                    M_pi_ell[s, :] = np.array(sol["x"]).transpose()

                v =  c_pi + self.gamma * np.dot(M_pi_ell, v)
                inner_iter += 1

            #compute the initial policy
            for a in range(mdp.m):
                P = mdp.P[a, :, :]
                for s in range(mdp.n):
                    
                    c = matrix(-v)
                    h = matrix(np.concatenate((np.zeros((mdp.n, )), (P[s, :]/self.alpha).transpose())))
                    sol = solvers.lp(c, G, h, A, b)
                    M_pi_k[a, s, :] = np.array(sol["x"]).transpose()

            #compute the residual
            res = v - np.min(mdp.c + self.gamma * np.dot(M_pi_k, v), axis=0)
            res_.append(np.linalg.norm(res, np.inf))
            
            iter += 1
            
            if verbose:
                print("iteration: {}, inf-norm residual: {}".format(iter, res_[-1]))
            
            pi = np.argmin(mdp.c + self.gamma * np.dot(M_pi_k, v), axis=0) 

        return v, res_