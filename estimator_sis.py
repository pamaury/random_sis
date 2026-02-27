# -*- coding: utf-8 -*-
"""
Run like this: first run in this directory:
    git clone https://github.com/malb/lattice-estimator.git

    sage: import sys
    sage: sys.path.append('path/to/lattice-estimator')
    sage: attach("estimator_sis.py")
    sage: %time results = runall()

"""
# Copyright Amaury Pouly and Yixin Shen
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from sage.all import sqrt, log, exp, tanh, coth, e, pi, RR, ZZ, erf

from estimator.cost import Cost
from estimator.lwe_parameters import LWEParameters
from estimator.reduction import delta as deltaf
from estimator.reduction import RC, ReductionCost
from estimator.reduction import cost as cost_bkz
from estimator.conf import red_cost_model as red_cost_model_default
from estimator.io import Logging
from estimator import sis
from estimator.schemes import Dilithium2_MSIS_WkUnf, Dilithium3_MSIS_WkUnf, Dilithium5_MSIS_WkUnf
from estimator.schemes import Dilithium2_MSIS_StrUnf, Dilithium3_MSIS_StrUnf, Dilithium5_MSIS_StrUnf
import itertools
from enum import Enum
from dataclasses import dataclass
import tabulate
import heapq
import time

# Precision of computations.
myRR = RealField(200)

def rho_Z(s):
    """Compute rho_s(Z) using Lemma 1. The approximation quality is 7e-6."""
    s = myRR(s)
    if s <= 1:
        # Not sure why: if I put the '1' inside myRR then the result is always 0
        res = 1 + 2*exp(myRR(-pi/s**2))
    else:
        res = s*(1+2*exp(myRR(-pi*s**2)))
    assert res >= 1, f"wrong res {res} for s={s}"
    return res

# This is the "exact" formula for the complexity of T_MCMC(L,s) where:
# * n is the dimension of the lattice L,
# * we assume that we have a BKZ-beta reduced basis,
# * the volume of the lattice L is red_vol^n.
# Returns the natural logarithm of the complexity.
def mcmc_sampling_complexity_exact(
    n,
    beta,
    red_vol,
    s
):
    res = myRR(0)
    # The complexity is the product (for i=1,..n) for rho_{|b_i*|/s}(Z)
    # where b_i* is the i-th GS vector. With the GSA, we have
    # - |b_i*|=delta_beta^{-2(-1)}*|b_1|
    # - |b_1|=vol^{1/n}*delta_beta^n
    delta_beta = deltaf(beta)
    # print("sampler: n={}, beta={}, red_vol={}, s={}".format(n, beta, red_vol, s))
    # print("arctic={}".format(exp(mcmc_sampling_complexity_arctic(n, beta, red_vol, s))))
    for i in range(1, n+1):
        s_i = (red_vol*delta_beta**(n-1-2*(i-1)) / s)
        # approximate rho_{s_i}(Z) with a few of the biggest terms
        res += log(rho_Z(s_i))

    return res

# Same as mcmc_sampling_complexity_exact() but use an approximate formula
# from Arctic crypt.
def mcmc_sampling_complexity_arctic(
    n,
    beta,
    red_vol,
    s,
    p = 3,
):
    delta_beta = deltaf(beta)
    b1 = red_vol * delta_beta**(n-1)
    alpha = myRR(b1 / s)

    def my_exp_integral_e1(x):
        # exp_integral_e1() produces errors on very large inputs:
        # for those we know the result is so incredibly small that we just return zero
        if x >= myRR(200):
            return 0
        return exp_integral_e1(x)

    k0 = max(-1, min(n - 1, floor(log(alpha) / 2 / log(delta_beta))))

    A = myRR((k0+1)*log(alpha/delta_beta**(k0)))
    B = sum(
        2**(ell-1)*(-1)**(ell+1)/ell*(
            my_exp_integral_e1(pi*ell*alpha**2*myRR(delta_beta**(-4*(k0+1/2)))) - my_exp_integral_e1(pi*ell*alpha**2*myRR(delta_beta**(-4*(-1/2))))
        )
        for ell in range(1, p+1)
    )
    C = sum(
        2**(ell-1)*(-1)**(ell+1)/ell*(
            my_exp_integral_e1(pi*ell/alpha**2*myRR(delta_beta**(4*(k0+1/2)))) - my_exp_integral_e1(pi*ell/alpha**2*myRR(delta_beta**(4*(n-1/2))))
        )
        for ell in range(1, p+1)
    )

    return myRR(A+(B+C)/2/log(delta_beta))

INFINITY_NORM = 0

# Compute \rho_s(B_n^{p,Z}(ell))
def rho_ball(s, n, p, ell):
    if p != INFINITY_NORM:
        raise ValueError("Only the infinity norm is supported")
    # For large values of s this is very accurate. Furthermore, this is
    # an unconditional upper bound.
    return myRR(1 + s * erf(ell * sqrt(pi) / s))

# Estimate the length of the shortest vector of a BKZ-reduced basis
def bkz_shortest_vector(n, red_vol, beta):
    delta_beta = deltaf(beta)
    return myRR(delta_beta**(n-1) * red_vol)

class Result:
    def __init__(self, sis_cost, frac_failure, beta, s, bkz_cost, sampl_cost, nr_samples, bkz_shortest_vector):
        self.cost = bkz_cost["rop"] + sis_cost
        self.sis_cost = sis_cost
        self.frac_failure = frac_failure
        self.beta = beta
        self.s = s
        self.bkz_cost = bkz_cost["rop"]
        self.sampl_cost = sampl_cost
        self.nr_samples = nr_samples
        self.bkz_shortest_vector = bkz_shortest_vector

    def __lt__(a, b):
        return a.cost > b.cost

    def __repr__(self):
        def myprint(x):
            return "2^{}".format(myRR(log(x)/log(2)))
        return (f"       cost: {myprint(self.cost)}\n" +
                f"   SIS cost: {myprint(self.sis_cost)}\n" +
                f"  frac fail: {myprint(self.frac_failure)}\n" +
                f"       beta: {self.beta}\n" +
                f"          s: {self.s}\n" +
                f"   BKZ cost: {myprint(self.bkz_cost)}\n" +
                f"   DGS cost: {myprint(self.sampl_cost)}\n" +
                f"          N: {myprint(self.nr_samples)}\n" +
                f"  BKZ short: {self.bkz_shortest_vector:f}\n")

# Estimate the complexity of the SIS algorithm (Theorem 5).
#
# - n,k: dimension of matrix A defining the lattice
# - q: prime defining the field Z_q
# - p: norm (use INFINITY_NORM for infinity norm)
# - s: width of the Gaussian sampler
# - beta: block size for BKZ to apply to the basis for use by the Gaussian sampler
# - ell: maximum length of the vector to be found
# - use_exact_mcmc: if True, use the exact MCMC sampler complexity.
# - bkz_cost: cost of BKZ reducing the basis
# - quantum:
# Returns:
# - a Result or None in case of error/invalid parameters.
def estimate_sis_cost(n, k, q, p, ell, s, beta, use_exact_mcmc, bkz_cost, quantum):
    # Compute cost of producing one Gaussian sample with a BKZ reduction basis.
    compl_fn = mcmc_sampling_complexity_exact if use_exact_mcmc else mcmc_sampling_complexity_arctic
    sampling_complexity = exp(compl_fn(n, beta, q**(1-k/n), s))

    # NOTE: numbers quickly become huge so we need to be careful to manipulate
    # them in log form when possible

    # Compute number of samples required and success probability.
    # Note: here we overapproximate \rho_s(ZZ\qZZ) by \rho_s(\ZZ\{0}), this only
    # increases the complexity estimate.

    # Neglect first term which is always very small.
    log2_q = myRR(log(q)/log(2))
    log2_nu = myRR((k-n)*log2_q + n*log(rho_Z(s))/log(2))
    log2_mu = myRR((k-n)*log2_q + n*log(rho_ball(s, n, p, ell))/log(2))
    log2_theta = myRR((1+k-n)*log2_q + n*log(rho_Z(s*sqrt(2)/q))/log(2) + n*log(rho_Z(s/sqrt(2)))/log(2))
    log2_eps = myRR(log(4*q)/log(2) - log2_mu)
    log2_p = myRR(log2_mu - 2 - log2_nu)
    log2_N = -log2_p

    if quantum:
        # Assume quadratic speedup on sampling and finding
        sampling_complexity = sqrt(sampling_complexity)
        sis_cost = 2**(log2_N/2) * (1 + sampling_complexity)
    else:
        sis_cost = 2**log2_N * (1 + sampling_complexity)

    return Result(
        sis_cost,
        2**log2_eps + 2**(log2_theta-2*log2_nu),
        beta,
        s,
        bkz_cost,
        sampling_complexity,
        2**log2_N,
        bkz_shortest_vector(n, q**(1-k/n), beta)
    )

# Optimize the choice the parameters to solve SIS.
#
# - n,k: dimension of matrix A defining the lattice
# - q: prime defining the field Z_q
# - p: norm (use INFINITY_NORM for infinity norm)
# - ell: maximum length of the vector to be found
# - max_failure_frac: maximum fraction of lattices on which the algorithm can fail
# - red_cost_model: BKZ reduction cost model
# Returns:
# - best complexity
# - fraction of lattices on which it does not work
# - s: Gaussian width
# or None in case of error/invalid parameters.
def opt_sis(n, k, q, p, ell, max_failure_frac, red_cost_model, quantum):
    # Perform a coarse-grained search.
    best_cost_heap = []
    keep_best_count = 100
    beta_step = 10

    if quantum:
        s_step = int(ell * 0.02)
        assert s_step > 0
        values_s = [x for x in range(int(ell * 0.2), int(ell * 1.5), s_step)]
    else:
        # Look around s=ell which experimenting seems to be close to optimal.
        s_step = int(ell * 0.02)
        assert s_step > 0
        values_s = [x for x in range(int(ell * 0.9), int(ell * 1.1), s_step)]

    start_time = time.time()
    for beta in range(n//4, n, beta_step):
        # Compute cost of running BKZ.
        bkz_cost = cost_bkz(red_cost_model, beta, n)

        for s in values_s:
            this_cost = estimate_sis_cost(n, k, q, p, ell, s, beta, False, bkz_cost, quantum)
            if this_cost.frac_failure > max_failure_frac:
                continue
            if this_cost is not None:
                if len(best_cost_heap) >= keep_best_count:
                    heapq.heappushpop(best_cost_heap, this_cost)
                else:
                    heapq.heappush(best_cost_heap, this_cost)
    print("Initial search took {}s".format(int(time.time() - start_time)))
    print("Best solution so far: ", heapq.nlargest(1, best_cost_heap))

    best_cost = heapq.nlargest(1, best_cost_heap)[0]

    mini_s_step = max(1, int(s_step / 20))
    # Look around the best solutions
    for candidate in best_cost_heap:
        for beta in range(candidate.beta - beta_step//2, candidate.beta + beta_step//2):
            # Compute cost of running BKZ.
            bkz_cost = cost_bkz(red_cost_model, beta, n)
            for s in range(candidate.s - s_step//2, candidate.s + s_step//2, mini_s_step):
                this_cost = estimate_sis_cost(n, k, q, p, ell, s, beta, False, bkz_cost, quantum)
                if this_cost.frac_failure > max_failure_frac:
                    continue
                if this_cost > best_cost:
                    best_cost = this_cost

    print("Complete search took {}s".format(int(time.time() - start_time)))

    return best_cost

DILITHIUM_SIS_PARAM = [
    {
        "level": "NIST Level 2",
        "m": 256 * 9,
        "n": 256 * 4,
        "q": 8380417,
        "ell": 350209,
    },
    {
        "level": "NIST Level 3",
        "m": 256 * 12,
        "n": 256 * 6,
        "q": 8380417,
        "ell": 724481,
    },
    {
        "level": "NIST Level 5",
        "m": 256 * 16,
        "n": 256 * 8,
        "q": 8380417,
        "ell": 769537,
    },
]

def runall(use_new_opt = True, quantum = False):
    if quantum:
        red_cost_model = RC.ChaLoy21.__class__()
    else:
        red_cost_model = RC.MATZOV.__class__(nn='list_decoding-classical')
    results = []
    for params in DILITHIUM_SIS_PARAM:
        print("{}:".format(params["level"]))
        res = opt_sis(
            params["m"],
            # NOTE: due to our definition of k in the paper, this corresponds to m-n for the
            # "typical" definition of LWE.
            params["m"]-params["n"],
            params["q"],
            INFINITY_NORM,
            params["ell"],
            1/2,
            red_cost_model,
            quantum
        )
        print(res)
        results.append(res)
    return results

# Same order as in DILITHIUM_SIS_PARAM.
EUROCRYPTO_PARAMS = [
    # NIST Level 2
    { "beta": 609, "s": 346702, "quantum": False },
    # NIST Level 3
    { "beta": 919, "s": 723749, "quantum": False },
    # NIST Level 5
    { "beta": 1314, "s": 769533, "quantum": False },
]

# Reproduce the estimates from the paper, does not run the optimizer but only
# the cost function
def reproduce_paper(paper_params = EUROCRYPTO_PARAMS):
    red_cost_model = RC.MATZOV.__class__(nn='list_decoding-classical')
    results = []
    for i in range(len(EUROCRYPTO_PARAMS)):
        params = DILITHIUM_SIS_PARAM[i]
        opts = paper_params[i]
        print("{}:".format(params["level"]))
        # Compute cost of running BKZ.
        bkz_cost = cost_bkz(red_cost_model, opts["beta"], params["m"])
        res = estimate_sis_cost(
            params["m"],
            params["m"]-params["n"],
            params["q"],
            INFINITY_NORM,
            params["ell"],
            opts["s"],
            opts["beta"],
            True,
            bkz_cost,
            opts["quantum"]
        )
        results.append(res)
    return results

def lattice_estimator_sis():
    red_cost_model = RC.MATZOV.__class__(nn='list_decoding-classical')
    print("NIST Level 2")
    print(sis.estimate(Dilithium2_MSIS_WkUnf, red_cost_model = red_cost_model))
    print("NIST Level 3")
    print(sis.estimate(Dilithium3_MSIS_WkUnf, red_cost_model = red_cost_model))
    print("NIST Level 5")
    print(sis.estimate(Dilithium5_MSIS_WkUnf, red_cost_model = red_cost_model))

class ColumnType(Enum):
    INTEGER = 0
    FLOAT = 1
    LOG2_INTEGER = 2

COLUMNS = [
    ("cost", "cost", ColumnType.LOG2_INTEGER),
    ("$\\beta$", "beta", ColumnType.INTEGER),
    ("$s$", "s", ColumnType.INTEGER),
    ("$\\log_2 N$", "nr_samples", ColumnType.LOG2_INTEGER),
    ("sample", "sampl_cost", ColumnType.LOG2_INTEGER),
    ("BKZ", "bkz_cost", ColumnType.LOG2_INTEGER),
]

def results_table_latex(results, columns = COLUMNS):
    col_headers = [col[0] for col in columns]
    all_rows = []
    for (i,res) in enumerate(results):
        res = {
            key: getattr(res, key)
            for key in dir(res)
        }
        this_row = [DILITHIUM_SIS_PARAM[i]["level"]]
        for col in columns:
            typ = col[2]
            if typ == ColumnType.INTEGER:
                this_row.append(int(res[col[1]]))
            elif typ == ColumnType.FLOAT:
                this_row.append("{:.2f}".format(res[col[1]]))
            elif typ == ColumnType.LOG2_INTEGER:
                this_row.append(int(log(res[col[1]])/log(2)))
            else:
                assert False, "unsupported"
        all_rows.append(this_row)

    return tabulate.tabulate(
        all_rows,
        headers=["Scheme"] + col_headers,
        tablefmt="latex_raw",
        floatfmt=".3f",
    )
