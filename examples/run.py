import numpy as np
import random
import time
import json
import argparse

from riskaverse_DP.mdps import mdp

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument("-n", type=int, default=10,
                    help="number of states")
parser.add_argument("-m", type=int, default=2,
                    help="number of actions")
parser.add_argument("-gamma", type=float, default=0.9,
                    help="discount factor")
parser.add_argument("-alpha", type=float, default=0.3,
                    help="confidence level")
parser.add_argument("-verbose", type=str2bool, default=False)
parser.add_argument("-generate_plot", type=str2bool, default=True)
parser.add_argument("-risk_measure", type=str, default="CVaR")

                    
args = parser.parse_args()


random.seed(0)
np.random.seed(0)

n = args.n
m = args.m

gamma = args.gamma
alpha = args.alpha

if args.risk_measure == "CVaR":
    from riskaverse_DP.CVaR_solvers import SNMI, SNMII, SNMIII, OPI 

elif args.risk_measure == "MUS1":
    from riskaverse_DP.MUS1_solvers import SNMI, SNMII, SNMIII, OPI 

else:
    raise ValueError("{} is invalid!".format(args.risk_measure))

VERBOSE = args.verbose

MDP = mdp(n, m)

tol1_ = 10**-6
tol2_ = 10**-9

print("SNMI")
start_time = time.time()
riskaverse_newton = SNMI(gamma, alpha, tol1 = tol1_, tol2 = tol2_)
v_1, res1 = riskaverse_newton.solve(MDP, np.ones(MDP.n, ), verbose=VERBOSE)
tot_time1 = time.time() - start_time
print("tot time: {} [s]".format(tot_time1))

SNMI_res = {"time": tot_time1, "residuals": res1}

print("SNMII warm-starting False")
start_time = time.time()
riskaverse_PI = SNMII(gamma, alpha, tol1 = tol1_, tol2 = tol2_)
v_2_wF, res2_wF = riskaverse_PI.solve(MDP, np.ones(MDP.n, ), warmstarting=False, verbose=VERBOSE)
tot_time2wF = time.time() - start_time
print("tot time: {} [s]".format(tot_time2wF))
SNMII_wF_res = {"time": tot_time2wF, "residuals": res2_wF, "warmstarting":False}

print("SNMII warm-starting True")
start_time = time.time()
riskaverse_PI = SNMII(gamma, alpha, tol1 = tol1_, tol2 = tol2_)
v_2_wT, res2_wT = riskaverse_PI.solve(MDP, np.ones(MDP.n, ), warmstarting=True, verbose=VERBOSE)
tot_time2wT = time.time() - start_time
print("tot time: {} [s]".format(tot_time2wT))

SNMII_wT_res = {"time": tot_time2wT, "residuals": res2_wT, "warmstarting":True}

print("SNMIII Warning: no guarantees for global convergence!")
start_time = time.time()
riskaverse_newton_old = SNMIII(gamma, alpha, tol1 = tol1_)
v_3, res3 = riskaverse_newton_old.solve(MDP, np.ones(MDP.n, ), verbose=VERBOSE)
tot_time3 = time.time() - start_time
print("tot time: {} [s]".format(tot_time3))

SNMIII_res = {"time": tot_time3, "residuals": res3}

print("{}-OPI".format(args.risk_measure))

W_ = [1, 2, 5, 10, 20, 30, 50]

results = {"SNMI": SNMI_res, "SNMII_wT": SNMII_wT_res, "SNMII_wF": SNMII_wF_res, "SNMIII": SNMIII_res}

for W in W_:

    start_time = time.time()
    riskaverse_newton_old = OPI(gamma, alpha, tol1 = tol1_, max_inner_iter=W)
    v_4, res4 = riskaverse_newton_old.solve(MDP, np.ones(MDP.n, ), verbose=VERBOSE)
    tot_time = time.time() - start_time
    print("W:{}, tot time: {} [s]".format(W, tot_time))

    name = "{}-OPI_{}".format(args.risk_measure, W)

    results[name] = {"time": tot_time, "residuals": res4, "inner-iterations": W}
results["general-info"] = {"tol1": tol1_, "tol2": tol2_}

#save results into json file
with open("{}-MDP_{}_{}_{}_{}.json".format(args.risk_measure, n, m, gamma, alpha), "w") as f:
    json.dump(results, f)

f.close()

if args.generate_plot:
    #plotting results
    import matplotlib.pyplot as plt
    linestyles_ = ["solid", "dotted", "dashed", (0, (1, 10)), (0, (5, 10)), (5, (10, 3)), (0, (1, 10))]

    plt.figure(1)
    plt.semilogy(res1, c = "m", label="SNMI, time={:.3f} [s]".format(tot_time1))
    plt.semilogy(res2_wF, c='r', label="SNMII, time={:.3f} [s]".format(tot_time2wF))
    plt.semilogy(res2_wT, c='k', label="SNMII (warm-start), time={:.3f} [s]".format(tot_time2wT))
    plt.semilogy(res3, c='b', label="SNMIII, time={:.3f} [s]".format(tot_time3))

    ymin = 10**-6

    plt.ylabel(r"$\Vert r(v_k) \Vert_{\infty}$")
    plt.xlabel(r"$k$")
    plt.grid()
    plt.legend()
    plt.title(r"$\alpha = {},\,\gamma = {}$".format(args.alpha, args.gamma))
    plt.ylim(bottom=ymin) 

    plt.savefig('{}-SNMs_{}_{}_{}_{}.png'.format(args.risk_measure, args.n, args.m, args.gamma, args.alpha), bbox_inches='tight', dpi=600)


    plt.figure(2)

    for ii, W in enumerate(W_):

        name = "{}-OPI_{}".format(args.risk_measure, W)
        
        plt.semilogy(results[name]["residuals"], c='green', linestyle = linestyles_[ii], label="{}-OPI, W={}, time={:.3f}".format(args.risk_measure, W, results[name]["time"]))


    plt.semilogy(res2_wT, c='k', label="SNMII (warm-start), time={:.3f} [s]".format(tot_time2wT))


    plt.ylabel(r"$\Vert r(v _k) \Vert_{\infty}$")
    plt.xlabel(r"$k$")
    plt.grid()
    plt.legend()
    plt.title(r"$\alpha = {},\,\gamma = {}$".format(args.alpha, args.gamma))
    plt.ylim(bottom=ymin) 

    plt.savefig('{}-OPI_{}_{}_{}_{}.png'.format(args.risk_measure, args.n, args.m, args.gamma, args.alpha), bbox_inches='tight', dpi=600)

