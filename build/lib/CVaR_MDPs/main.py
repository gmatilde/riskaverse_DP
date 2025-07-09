import numpy as np
import random
import time
import json
from mdps import mdp
from solvers import SNMI, SNMII, SNMIII, CVaR_OPI #policy_iteration, CVaR_Newton, CVaR_Newton_old, CVaR_PI

random.seed(0)
np.random.seed(0)

n = 10#50#100
m = 2#30#50

gamma = 0.9
alpha = 0.3

VERBOSE = True

MDP = mdp(n, m)

tol1_ = 10**-6
tol2_ = 10**-9

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
