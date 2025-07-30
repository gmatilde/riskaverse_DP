# Semismooth Newton Methods for Risk-Averse Dynamic Programming

This repository contains Python code for reproducing the benchmark experiments presented in the paper:
**"Semismooth Newton Methods for Risk-Averse Dynamic Programming"**  [arXiv:2501.13612](https://arxiv.org/abs/2501.13612)

# 🧠 riskaverse\_DP

**Risk-Averse Dynamic Programming via Semismooth Newton Methods**
This repository provides a Python implementation of novel semismooth Newton algorithms to solve **risk-averse Markov Decision Processes (MDPs)** under Markovian risk measures—particularly the Conditional Value at Risk (CVaR) and Mean-Upper-Semideviation of order 1 (MUS1).

---

## ⭐ Features

* Implements **three semismooth Newton methods** for risk-averse MDPs:

  1. **SNM I**: Sequential risk-neutral MDP solves with perturbed transitions.
  2. **SNM II**: A risk-averse policy-iteration–style Newton solve.
  3. **SNM III**: A linearized, piecewise-smooth Newton update.
* Includes **risk-averse optimistic policy iteration** as a comparative baseline.
* Supports **CVaR** and **MUS1** as risk measures, with theoretical guarantees of quadratic convergence under certain conditions.
* Benchmarked on synthetic MDPs with varied sizes and discount factors, with empirical performance data included.

---

## 🚀 Installation

Follow these steps to set up your development environment and install the package:

1. **Create and activate a Conda environment** (recommended):

   ```bash
   conda create -n riskaverse python=3.10
   conda activate riskaverse
   ```

2. **Install project dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Install the package in editable mode**:

   ```bash
   pip install -e . --use-pep517
   ```

4. **Verify installation**:

   ```bash
   python -c "from riskaverse_DP.mdps import mdp; print('Success')"
   ```

This will ensure your `riskaverse_DP` package is installed and ready for development.

---

## 📦 Usage

Basic usage pattern:

```python
import numpy as np
from riskaverse_DP.mdps import mdp

n_states = 10
n_actions = 3

# Initialize with transition and stage-cost data
my_mdp = mdp(n_states, n_actions)

gamma_ = 0.9
alpha_ = 0.1
v_0 = np.zeros(n_states, )

from riskaverse_DP.CVaR_solvers import SNMI

my_SNMI = SNMI(gamma_, alpha_)

# Solve using one of the methods: "SNMI", "SNMII", "SNMIII", or "OPI"
v_opt, res = my_SNMI.solve(my_mdp, v_0)
```

Check the `examples/` folder for detailed scripts on synthetic MDPs.

---

## 📈 Benchmark Results

Experiments on MDPs with sizes up to 100 states and diversified action spaces show:

* **SNM methods** reach convergence in fewer than 10 iterations, whereas risk-averse VI needs over 150.
* **CPU times**: SNM I and SNM III are most efficient; SNM II may face numeric issues depending on solver accuracy.

| **MUS1 SNMs** | **MUS1 OPI** |
|:-------------:|:------------:|
| ![MUS1 SNMs](/examples/MUS1-SNMs_10_2_0.9_0.6.png) | ![MUS1 OPI](/examples/MUS1-OPI_10_2_0.9_0.6.png) |

---

## 📚 Background & Theory

For a full theoretical foundation—including proofs of convergence and algorithmic derivations—see the arXiv preprint *“Semismooth Newton Methods for Risk-Averse Markov Decision Processes”*.

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or pull requests for:

* New risk measures or benchmarks
* Extensions to large/continuous state spaces
* Performance optimizations or parallel implementations

---

## 📄 License

This project is released under the **MIT License**. See LICENSE for complete details.
