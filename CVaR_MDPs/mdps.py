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



if __name__ == "__main__":

    random.seed(0)
    np.random.seed(0)

    n = 50#100
    m = 30#50

    gamma = 0.9
    alpha = 0.3

    VERBOSE = False

    MDP = mdp(n, m)

    from solvers import SNMI, SNMII, SNMIII, CVaR_OPI #policy_iteration, CVaR_Newton, CVaR_Newton_old, CVaR_PI
    #PI = policy_iteration(gamma)
    #v_star, pi_star = PI.solve(MDP, np.ones(MDP.n, ))

    tol1_ = 10**-6
    tol2_ = 10**-6

    print("SNMI")
    start_time = time.time()
    riskaverse_newton = SNMI(gamma, alpha, tol1 = tol1_, tol2 = tol2_)
    v_1, res1 = riskaverse_newton.solve(MDP, np.ones(MDP.n, ), verbose=VERBOSE)
    tot_time = time.time() - start_time
    print("tot time: {} [s]".format(tot_time))

    SNMI_res = {"time": tot_time, "residuals": res1}
    
    print("SNMII warm-starting False")
    start_time = time.time()
    riskaverse_PI = SNMII(gamma, alpha, tol1 = tol1_, tol2 = tol2_)
    v_2_wF, res2_wF = riskaverse_PI.solve(MDP, np.ones(MDP.n, ), warmstarting=False, verbose=VERBOSE)
    tot_time = time.time() - start_time
    print("tot time: {} [s]".format(tot_time))

    SNMII_wF_res = {"time": tot_time, "residuals": res2_wF, "warmstarting":False}

    print("SNMII warm-starting True")
    start_time = time.time()
    riskaverse_PI = SNMII(gamma, alpha, tol1 = tol1_, tol2 = tol2_)
    v_2_wT, res2_wT = riskaverse_PI.solve(MDP, np.ones(MDP.n, ), warmstarting=True, verbose=VERBOSE)
    tot_time = time.time() - start_time
    print("tot time: {} [s]".format(tot_time))

    SNMII_wT_res = {"time": tot_time, "residuals": res2_wT, "warmstarting":True}

    print("SNMIII")
    start_time = time.time()
    riskaverse_newton_old = SNMIII(gamma, alpha, tol1 = tol1_)
    v_3, res3 = riskaverse_newton_old.solve(MDP, np.ones(MDP.n, ), verbose=VERBOSE)
    tot_time = time.time() - start_time
    print("tot time: {} [s]".format(tot_time))
    
    SNMIII_res = {"time": tot_time, "residuals": res3}

    print("CVaR-OPI")

    W_ = [1, 2, 5, 10, 20, 30, 50]

    results = {"SNMI": SNMI_res, "SNMII_wT": SNMII_wT_res, "SNMII_wF": SNMII_wF_res, "SNMIII": SNMIII_res}

    for W in W_:

        start_time = time.time()
        riskaverse_newton_old = CVaR_OPI(gamma, alpha, tol1 = tol1_, max_inner_iter=W)
        v_4, res4 = riskaverse_newton_old.solve(MDP, np.ones(MDP.n, ), verbose=VERBOSE)
        tot_time = time.time() - start_time
        print("W:{}, tot time: {} [s]".format(W, tot_time))

        name = "CVaROPI_{}".format(W)

        results[name] = {"time": tot_time, "residuals": res4, "inner-iterations": W}
    results["general-info"] = {"tol1": tol1_, "tol2": tol2_}

    #save results into json file
    with open("CVaRMDP_{}_{}_{}_{}.json".format(n, m, gamma, alpha), "w") as f:
        json.dump(results, f)

    f.close()
