import json
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-n", type=int,
                    help="number of states")
parser.add_argument("-m", type=int,
                    help="number of actions")
parser.add_argument("-gamma", type=float,
                    help="discount factor")
parser.add_argument("-alpha", type=float,
                    help="confidence level")
                    
args = parser.parse_args()

f_name = "CVaRMDP_{}_{}_{}_{}.json".format(args.n, args.m, args.gamma, args.alpha)

#open the file and read the results
with open(f_name, 'r') as f:
    res = json.load(f)
f.close()

#parse the results
res_SNMI = res["SNMI"]
res_SNMII_wT = res["SNMII_wT"]
res_SNMII_wF = res["SNMII_wF"]
res_SNMIII = res["SNMIII"]      

#settings for plots
linestyles_ = ["solid", "dotted", "dashed", (0, (1, 10)), (0, (5, 10)), (5, (10, 3))]
W_ = [1, 2, 5, 10, 20]

plt.figure(1)
plt.semilogy(res_SNMI["residuals"], c = "m", label="SNMI")
plt.semilogy(res_SNMII_wF["residuals"], c='k', label="SNMII")
plt.semilogy(res_SNMIII["residuals"], c='b', label="SNMIII")

print("SNMI CPU time {} [s]".format(res_SNMI["time"]))
print("SNMII CPU time {} [s]".format(res_SNMII_wF["time"]))
print("SNMII (warm-start) CPU time {} [s]".format(res_SNMII_wT["time"]))
print("SNMIII CPU time {} [s]".format(res_SNMIII["time"]))

plt.figure(2)
for ii, W in enumerate(W_):

    name = "CVaROPI_{}".format(W)
    dict_values = res[name] 

    plt.semilogy(dict_values["residuals"], c='green', linestyle = linestyles_[ii], label="Risk-averse OPI, W={}".format(W))

    print("CVaR OPI W={}, CPU time {} [s]".format(W, dict_values["time"]))


ymin = 10**-6

plt.figure(1)

plt.ylabel(r"$\Vert r(v_k) \Vert_{\infty}$")
plt.xlabel(r"$k$")
plt.grid()
plt.legend()
plt.title(r"$\alpha = {},\,\gamma = {}$".format(args.alpha, args.gamma))
plt.ylim(bottom=ymin) 

plt.savefig('SNMs_{}_{}_{}_{}.png'.format(args.n, args.m, args.gamma, args.alpha), bbox_inches='tight', dpi=600)

plt.figure(2)
plt.semilogy(res_SNMII_wF["residuals"], c='k', label="SNMII")

plt.ylabel(r"$\Vert r(v _k) \Vert_{\infty}$")
plt.xlabel(r"$k$")
plt.grid()
plt.legend()
plt.title(r"$\alpha = {},\,\gamma = {}$".format(args.alpha, args.gamma))
plt.ylim(bottom=ymin) 

plt.savefig('RiskAverseOPI_{}_{}_{}_{}.png'.format(args.n, args.m, args.gamma, args.alpha), bbox_inches='tight', dpi=600)

plt.show()
