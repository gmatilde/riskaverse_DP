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

### Mean-Upper-Semideviation (Order 1)

This project implements the **Mean-Upper-Semideviation risk measure of order 1 (MUS1)**, a coherent risk measure used in risk-averse optimization and decision-making.

Given a random variable `X` and a parameter `kappa` in `[0, 1]`, the MUS1 risk measure is defined as:

    rho(X) = E[X] + kappa * E[(X - E[X])_+]

Here, `(x)_+` denotes the positive part of `x`, i.e., `max(x, 0)`. The parameter `kappa` controls the degree of risk sensitivity to values above the mean.

For a finite probability space with outcomes `X_i` and probabilities `p_i`, the MUS1 risk measure has the following dual representation as a **linear program (LP)**:

**Linear Program:**

    Maximize:     sum_i p_i * xi_i * X_i

    Subject to:   sum_i p_i * xi_i = 1
                  0 <= xi_i <= 1 + kappa   for all i

This LP computes the worst-case expected value under a distorted probability distribution defined by the weights `xi_i * p_i`, consistent with the MUS1 risk measure.

| **MUS1 SNMs** | **MUS1 OPI** |
|:-------------:|:------------:|
| ![MUS1 SNMs](/examples/MUS1-SNMs_100_10_0.9_0.5.png) | ![MUS1 OPI](/examples/MUS1-OPI_100_10_0.9_0.5.png) |

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
