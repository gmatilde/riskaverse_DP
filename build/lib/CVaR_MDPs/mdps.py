#creating artificial mdps
import numpy as np
import random
import time
import json

class mdp:
    def __init__(self, n, m):

        self.n = n
        self.m = m

        low_ = 0 
        high_ = 100

        #create the transition map
        rng = np.random.default_rng()
        self.P = rng.uniform(low=low_, high=high_, size=(m, n, n))

        #inserting some high values in the prob matrix
        outlier_ = 10**12
        n_outliers = int((m*n*n)/10) # one tenth of the entries are outliers
        
        for ii in range(n_outliers):

            rnd_idx = (random.randint(0, m-1), random.randint(0, n-1), random.randint(0, n-1))
            self.P[rnd_idx] = outlier_


        #inserting some zeros in the prob matrix
        sparsity_ = int((m*n*n)/3) # one-third of the entries equal to zero
        
        #TODO
        for ii in range(sparsity_):

            rnd_idx = (random.randint(0, m-1), random.randint(0, n-1), random.randint(0, n-1))
            self.P[rnd_idx] = 0.0


        #normalization
        S = self.P.sum(axis=2)

        for i in range(m):
            for j in range(n):

                self.P[i, j, :] = self.P[i, j, :]/S[i, j]
        
        
        #create the stage cost
        self.c = rng.uniform(low=-high_, high=high_, size=(m, n))
