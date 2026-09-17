# -*- coding: utf-8 -*-
"""
Run like this: first run in this directory:
    git clone https://github.com/malb/lattice-estimator.git

    sage: import sys
    sage: sys.path.append('./lattice-estimator')
    sage: attach("estimator_eurocrypt.py")
    sage: %time results = runall()

"""
# Copyright Amaury Pouly and Yixin Shen
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from sage.all import sqrt, log, exp, tanh, coth, e, pi, RR, ZZ

from estimator.cost import Cost
from estimator.lwe_parameters import LWEParameters
from estimator.reduction import delta as deltaf
from estimator.reduction import RC, ReductionCost, ABFKSW20
from estimator.reduction import cost as cost_bkz
from estimator.util import local_minimum, early_abort_range
from estimator.io import Logging
from estimator.schemes import (
    Kyber512,
    Kyber768,
    Kyber1024
)
from estimator.nd import NoiseDistribution
import itertools
from enum import Enum
from dataclasses import dataclass
import tabulate
import heapq
import time

# Precision of computations.
myRR = RealField(200)

class Cost:
    def __init__(self, rop, problem):
        self.problem = problem
        self.others = {
            "rop": rop,
        }

    def __getitem__(self, index):
        return self.others[index]

    def __setitem__(self, index, value):
        self.others[index] = value

    def to_str(self, filter_fn):
        s = ""
        name_len = max(len(x) for x in self.others.keys())
        first = True
        for (name, v) in self.others.items():
            if not filter_fn(name):
                continue
            if isinstance(v, Cost):
                v = v["rop"]
            kk = f"{name:>{name_len}}"
            try:
                round_bound = 2048
                if (1 / round_bound < abs(v) < round_bound) or (not v):
                    if abs(v % 1) < 1e-7:
                        vv = "%8d" % round(v)
                    else:
                        vv = "%8.3f" % v
                else:
                    vv = "%7s" % ("≈2^%.1f" % log(v, 2))
            except TypeError:  # strings and such
                vv = "%8s" % v
            if first:
                s += "\n"
                first = False
            s += f"{kk}: {vv}"
        return s

    def __str__(self):
        return self.to_str(lambda v: True)

# Gaussian sampler to use
class Sampler(Enum):
    GPV = 1 # GPV, improved by BLPRS13
    # The complexity of the MCMC algorith essentially depends on
    # alpha^2 = |b1| / s
    # see https://eprint.iacr.org/2019/660.pdf (note that s=sqrt(2pi)sigma
    # using their notations)
    # If we take alpha = 2 then the complexity is essentially polynomial (and
    # does not need to assume the GSA)
    # If we take alpha = 1 then complexity becomes quite high without assuming the GSA.
    # If we take alpha = 1/sqrt(2) then we really need to assume the GSA to use the fact
    # that the |bi*| decreases exponentially fast to get a good complexity.
    MCMC_TWO = 2 # alpha=2
    MCMC_ONE = 3 # alpha = 1
    MCMC_SQRT_HALF = 4 # alpha = 1/sqrt(2) which is what we suggest in the paper.
    MCMC_ARCTIC = 5 # alpha = ...

class ModulusSwitching(Enum):
    Off = 1
    EstimateOn = 2 # estimate complexity with ideal modulus switching

# quantum version of [ABFKSW20](https://eprint.iacr.org/2020/707) with quadratic speedup
class QuantumABFKSW20(ABFKSW20):
    def __call__(self, *args, **kwargs):
        return sqrt(super().__call__(*args, **kwargs))

# [BCSS23](https://eprint.iacr.org/2022/676.pdf)
class BCSS23(ReductionCost):
    __name__ = "ChaLoy21"
    short_vectors = ReductionCost._short_vectors_sieve

    def __call__(self, beta, d, B=None):
        """

        See [AC:ChaLoy21]_.

        :param beta: Block size ≥ 2.
        :param d: Lattice dimension.
        :param B: Bit-size of entries.
        """

        return ZZ(2) ** RR(0.2563 * beta)

class QuantumMode(Enum):
    CLASSICAL = 1 # Entirely classical.
    # Quantum algorithm from Eurocrypt paper + quadratic speedup on MCMC sampler
    FULL_QUANTUM = 2

@dataclass
class Parameters:
    sampler: Sampler
    mod_switching: ModulusSwitching
    quantum_mode: QuantumMode
    red_cost_model: ReductionCost

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
        #res *= sum([exp(-myRR(pi*x**2/s_i**2)) for x in range(-10,11)])
        res += log(rho_Z(s_i))

    return res

# Same as mcmc_sampling_complexity_exact() but use an approximate formula
# only valid when |b_1|<=2s.
def mcmc_sampling_complexity_eurocrypt(
    n,
    beta,
    red_vol,
    s
):
    # The complexity is the product (for i=1,..n) for rho_{|b_i*|/s}(Z)
    # where b_i* is the i-th GS vector. With the GSA, we have
    # - |b_i*|=delta_beta^{-2(i-1)}*|b_1|
    # - |b_1|=vol^{1/n}*delta_beta^n
    # Since computing the product is expensive, we use the fact that
    # |b_i|/s will be rather small so rho_{|b_i|/s}(Z) will be close to 1,
    # hence, the log of the complexity is very close to for i=1,..,n
    # of log(rho_{|b_i*|/s}(Z)) ~= log(1+2*exp(-pi*s^2/|b_i*|^2)) ~= 2*exp(-pi*s^2/|b_i*|^2)
    # Still, computing the sum of all such terms is expensive so we replace it
    # by the integral when sum for i=1,...,infinity. There is an exact expression:
    # int(exp(-alpha*x^n),n=0..infinity)=Ei(1, alpha)/ln(x)
    delta_beta = deltaf(beta)
    b1 = red_vol * delta_beta**(n-1)
    # Check constraint on s, allow for small error in comparison due to floating point approx.
    assert b1/(2*s) <= 1.01, "b1={} needs to be smaller than 2s={}".format(myRR(b1), myRR(2*s))
    alpha = pi * s**2 / b1**2
    res = myRR(ln(1+2*exp(-alpha)) + 2 * exp_integral_e(1, alpha)/ln(delta_beta**4))
    # The following formula get mores precise as we add terms but it is more expensive:
    # for k in range(1, 5):
    #     res += -(-2)**k / k * exp_integral_e(1, k * alpha)/ln(delta_beta**4)
    return myRR(res)

# Same as mcmc_sampling_complexity_exact() but use an approximate formula
# from Arctic crypt
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

    # k0 = max(-1, min(n - 1, floor(log(alpha) / 2 / log(delta_beta))))
    # A = myRR((k0 + 1) * log(alpha) - k0 * (k0 + 1) * log(delta_beta))
    # B = my_exp_integral_e1(myRR(pi*alpha**2*delta_beta**(-4*(k0+1)))) - my_exp_integral_e1(myRR(pi*alpha**2))
    # C = my_exp_integral_e1(myRR(pi/alpha**2*delta_beta**(4*(k0+1)))) - my_exp_integral_e1(myRR(pi/alpha**2*delta_beta**(4*n)))
    # return myRR(A+(B+C)/2/log(delta_beta))

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

def mcmc_sampling_complexity(
    n,
    beta,
    red_vol,
    s
):
    # return mcmc_sampling_complexity_eurocrypt(n, beta, red_vol, s)
    return mcmc_sampling_complexity_arctic(n, beta, red_vol, s)
    # return mcmc_sampling_complexity_exact(n, beta, red_vol, s)

# Estimate the cost of the attack given all the parameters.
def compute_cost(
    scheme,
    m,
    n_guess,
    beta,
    val_s,
    params,
    quiet = True,
    use_exact_mcmc = False
):
    assert m > scheme.n, "n={}, m={}, beta={}".format(scheme.n, m, beta)
    # n = n_dual + n_guess
    n_dual = scheme.n - n_guess
    # compute the two lower bounds on s
    use_mcmc = False
    klein_factor_max = None
    if params.sampler == Sampler.GPV:
        klein_factor = sqrt(ln(2*scheme.n+4)/pi)
    elif params.sampler == Sampler.MCMC_TWO:
        klein_factor = sqrt(2)
        use_mcmc = True
    elif params.sampler == Sampler.MCMC_ONE:
        klein_factor = 1
        use_mcmc = True
    elif params.sampler == Sampler.MCMC_SQRT_HALF:
        klein_factor = 1/2
        use_mcmc = True
    elif params.sampler == Sampler.MCMC_ARCTIC:
        # Ideally, we can go down to:
        klein_factor = 2*deltaf(beta)**(-m)
        klein_factor_max = 1/2
        use_mcmc = True
    else:
        assert False, "unsupported"

    sigma_e = RR(sqrt(2*pi) * scheme.Xe.stddev)

    if not quiet: print("klein factor: {}".format(klein_factor))
    mins1 = deltaf(beta)**m * scheme.q**(n_dual / m - 1) * klein_factor
    if klein_factor_max is None:
        maxs1 = mins1
    else:
        maxs1 = deltaf(beta)**m * scheme.q**(n_dual / m - 1) * klein_factor_max

    mins2 = 1 / (scheme.q**(1-scheme.n/m)/sqrt(e) - 2*sigma_e)
    if not quiet: print("min s: {} and {}".format(myRR(mins1), myRR(mins2)))

    # always choose the smallest possible value
    mins = RR(max(mins1, mins2))
    maxs = RR(max(maxs1, mins2))

    # compute the cost of computing a short vector of norm delta_0^m*vol(L)^(1/m) by BKZ
    bkz_cost = cost_bkz(params.red_cost_model, beta, m)

    if params.mod_switching == ModulusSwitching.Off:
        # real formula:
        guessing_comp = scheme.q**n_guess
    elif params.mod_switching == ModulusSwitching.EstimateOn:
        # fake modulus switching (estimated complexity)
        guessing_comp = 2**n_guess
    else:
        assert False, "unsupported"

    # Values of to try
    if val_s is not None:
        values_of_s = [val_s]
    else:
        search_nr_values = 10
        arctic_search_interval = maxs - mins
        values_of_s = [mins + arctic_search_interval * i / search_nr_values for i in range(search_nr_values)]

    best_cost = None
    for s in values_of_s:
        if use_mcmc:
            # sample from D_{L_q^\perp(A),qs}: the lattice has volume q^n_dual
            compl_fn = mcmc_sampling_complexity_exact if use_exact_mcmc else mcmc_sampling_complexity
            sampling_complexity = exp(compl_fn(m, beta, scheme.q**(n_dual / m), scheme.q * s))
        else:
            sampling_complexity = 1

        # now delta is fixed
        delta = exp(-m * s**2 * sigma_e**2  / 2) / 100
        if not quiet: print("delta=", myRR(delta))
        # avoid dealing with huge numbers, if delta is too small the complexity will be bad anyway
        if delta < myRR(1e-300) or sampling_complexity > myRR(1e250):
            continue
        # now N is fixed
        N = myRR(scheme.n * log(scheme.q) / delta**2)

        if params.quantum_mode == QuantumMode.CLASSICAL:
            rop = sampling_complexity * N + guessing_comp + bkz_cost["rop"]
        else:
            rop = sqrt(sampling_complexity * N * guessing_comp) + bkz_cost["rop"]

        if best_cost is not None and rop > best_cost["rop"]:
            continue

        if not quiet: print("##### N={}, guess={}, BKZ={}".format(int(log(N)/log(2)), int(log(scheme.q**n_guess)/log(2)), int(log(bkz_cost["rop"])/log(2))))
        if not quiet: print("rop:", myRR(rop))
        best_cost = Cost(rop = rop, problem = scheme)
        best_cost["delta"] = delta
        best_cost["N"] = N
        best_cost["s"] = s
        best_cost["x"] = scheme.q**(n_dual / m - 1) / s
        best_cost["mins1"] = myRR(mins1)
        best_cost["maxs1"] = myRR(maxs1)
        best_cost["mins2"] = myRR(mins2)
        best_cost["beta"] = beta
        best_cost["n_guess"] = n_guess
        best_cost["n_dual"] = n_dual
        best_cost["m"] = m
        best_cost["sampling"] = sampling_complexity
        best_cost["sampling_all"] = sampling_complexity * N
        best_cost["guessing"] = guessing_comp
        best_cost["bkz"] = bkz_cost
        best_cost["alpha"] = scheme.q * s /  (scheme.q**(n_dual / m) * deltaf(beta)**m)

    return best_cost

# Search for the best possible parameters for a scheme given a scheme.
def opt_cost(scheme, max_m, quantum_mode, params):
    best_cost = None

    # Without modulus switching, we can also consider very small values of n_guess
    if params.mod_switching == ModulusSwitching.Off:
        max_guess = 40
        guess_step = 2
    elif params.mod_switching == ModulusSwitching.EstimateOn:
        max_guess = scheme.n
        guess_step = 20

    for m in range(scheme.n+1, max_m, 50):
        for n_guess in range(1, max_guess, guess_step):
            for beta in range(m//2, m, 20):
                this_cost = compute_cost(scheme, m, n_guess, beta, None, quantum_mode, params)
                if this_cost is not None and (best_cost is None or this_cost < best_cost):
                    best_cost = this_cost

    best_cost['sampling'] = myRR(best_cost['sampling'])

    # Once we have the cost, recompute with exact MCMC complexity model.
    real_cost = compute_cost(scheme, best_cost["m"], best_cost['n_guess'], best_cost['beta'], best_cost['s'],
                             quantum_mode, params, use_exact_mcmc = True)
    print("sampling: approx=", best_cost['sampling'], "real=", myRR(real_cost['sampling']))

    return real_cost

class Result:
    def __init__(self, cost):
        self.cost = cost

    def __lt__(a, b):
        return a.cost["rop"] > b.cost["rop"]

# Search for the best possible parameters for a scheme given a scheme.
# This function performs a more sophisticated search to find better parameters.
def opt_cost_arctic(scheme, max_m, params, two_level_search = False):
    best_cost = None

    # Without modulus switching, we can also consider very small values of n_guess
    if params.mod_switching == ModulusSwitching.Off:
        max_guess = 40
        guess_step = 2
    elif params.mod_switching == ModulusSwitching.EstimateOn:
        max_guess = scheme.n // 3
        guess_step = 5
    m_step = 50
    beta_step = 10
    keep_best_count = 100
    best_cost_heap = []

    prefix = f"[{scheme.tag}, {params}]"

    # Perform a two-level search: first we do a coarse-grid search to identify the best candidates.
    start_time = time.time()
    best_so_far = None
    for m in range(scheme.n+1, max_m, m_step):
        for n_guess in range(1, max_guess, guess_step):
            for beta in range(m//2, m, beta_step):
                this_cost = compute_cost(scheme, m, n_guess, beta, None, params)
                if this_cost is not None:
                    this_cost = Result(this_cost)
                    if best_so_far is None or best_so_far < this_cost:
                        best_so_far = this_cost
                        print(prefix, "new best so far:", best_so_far.cost.to_str(lambda v: v == "rop"))
                    if len(best_cost_heap) >= keep_best_count:
                        heapq.heappushpop(best_cost_heap, this_cost)
                    else:
                        heapq.heappush(best_cost_heap, this_cost)

    print(prefix, "found best {} candidates".format(keep_best_count))
    print(prefix, "initial earch took {}s".format(int(time.time() - start_time)))

    if two_level_search:
        # Then for each candidate we do a fine-grained search.
        best_cost = None
        for candidate_cost in best_cost_heap:
            candidate = candidate_cost.cost
            for m in range(candidate["m"], min(candidate["m"] + m_step, max_m), 5):
                for n_guess in range(candidate["n_guess"], candidate["n_guess"] + guess_step):
                    for beta in range(candidate["beta"], candidate["beta"] + beta_step):
                        this_cost = compute_cost(scheme, m, n_guess, beta, None, params)
                        if this_cost is not None and (best_cost is None or this_cost < best_cost):
                            best_cost = this_cost
    else:
        print(prefix, "Two-level search disabled")
        best_cost = heapq.nlargest(1, best_cost_heap)[0].cost

    print(prefix, "Search took {}s".format(int(time.time() - start_time)))

    # Once we have the cost, recompute with exact MCMC complexity model.
    real_cost = compute_cost(scheme, best_cost["m"], best_cost['n_guess'], best_cost['beta'], best_cost['s'],
                             params, use_exact_mcmc = True)

    return real_cost

ALL_SCHEMES = [
    {"name": "Kyber512", "scheme": Kyber512},
    {"name": "Kyber768", "scheme": Kyber768},
    {"name": "Kyber1024", "scheme": Kyber1024},
]

# Run the optimizer on Kyber with the MCMC sampler, with and without modulus switching.
def runall(use_new_opt = True, quantum_mode = QuantumMode.FULL_QUANTUM, red_cost_model = RC.MATZOV.__class__(nn='list_decoding-classical')):
    results = {}
    for scheme in ALL_SCHEMES:
        scheme_results = {}
        for use_mod_switch in [False, True]:
            mod_switch_name = "{} modulus switching".format("with" if use_mod_switch else "no")
            params = Parameters(
                Sampler.MCMC_ARCTIC,
                ModulusSwitching.EstimateOn if use_mod_switch else ModulusSwitching.Off,
                quantum_mode,
                red_cost_model,
            )
            sc = scheme["scheme"]
            max_m = sc.m * 2
            fn = opt_cost_arctic if use_new_opt else opt_cost
            res = fn(sc, max_m, params)
            print("{} ({}):".format(scheme["name"], mod_switch_name))
            print(res)
            scheme_results[use_mod_switch] = res
        results[scheme["name"]] = scheme_results
    return results

def runall_quantum():
    return runall(quantum_mode = QuantumMode.FULL_QUANTUM, red_cost_model = RC.MATZOV.__class__(nn='list_decoding-classical'))

# Quantum attack, quadratic speed up on DGS, classical BKZ using list decoding.
EUROCRYPT_PARAMS_QUANTUM_CLASSICAL = [
    {"name": "Kyber512", "scheme": Kyber512, "mod_switch": False, "m": 1013, "n_guess": 1, "beta": 596, "s": 0.110, "quantum": QuantumMode.FULL_QUANTUM, "red_cost_model": RC.MATZOV.__class__(nn='list_decoding-classical')},
    {"name": "Kyber768", "scheme": Kyber768, "mod_switch": False, "m": 1519, "n_guess": 1, "beta": 919, "s": 0.120, "quantum": QuantumMode.FULL_QUANTUM, "red_cost_model": RC.MATZOV.__class__(nn='list_decoding-classical')},
    {"name": "Kyber1024", "scheme": Kyber1024, "mod_switch": False, "m": 2025, "n_guess": 1, "beta": 1302, "s": 0.130, "quantum": QuantumMode.FULL_QUANTUM, "red_cost_model": RC.MATZOV.__class__(nn='list_decoding-classical')},
    {"name": "Kyber512", "scheme": Kyber512, "mod_switch": True, "m": 913, "n_guess": 166, "beta": 476, "s": 0.080, "quantum": QuantumMode.FULL_QUANTUM, "red_cost_model": RC.MATZOV.__class__(nn='list_decoding-classical')},
    {"name": "Kyber768", "scheme": Kyber768, "mod_switch": True, "m": 1269, "n_guess": 251, "beta": 744, "s": 0.100, "quantum": QuantumMode.FULL_QUANTUM, "red_cost_model": RC.MATZOV.__class__(nn='list_decoding-classical')},
    {"name": "Kyber1024", "scheme": Kyber1024, "mod_switch": True, "m": 1725, "n_guess": 336, "beta": 1052, "s": 0.100, "quantum": QuantumMode.FULL_QUANTUM, "red_cost_model": RC.MATZOV.__class__(nn='list_decoding-classical')},
]

# Quantum attack, quadratic speed up on DGS, quantum BKZ using enumeration estimates
EUROCRYPT_PARAMS_QUANTUM_QABFKSW20 = [
    {"name": "Kyber512", "scheme": Kyber512, "mod_switch": False, "m": 1013, "n_guess": 1, "beta": 596, "s": 0.110, "quantum": QuantumMode.FULL_QUANTUM, "red_cost_model": QuantumABFKSW20()},
    {"name": "Kyber768", "scheme": Kyber768, "mod_switch": False, "m": 1519, "n_guess": 1, "beta": 869, "s": 0.130, "quantum": QuantumMode.FULL_QUANTUM, "red_cost_model": QuantumABFKSW20()},
    {"name": "Kyber1024", "scheme": Kyber1024, "mod_switch": False, "m": 2025, "n_guess": 1, "beta": 1192, "s": 0.130, "quantum": QuantumMode.FULL_QUANTUM, "red_cost_model": QuantumABFKSW20()},
    {"name": "Kyber512", "scheme": Kyber512, "mod_switch": True, "m": 913, "n_guess": 166, "beta": 496, "s": 0.070, "quantum": QuantumMode.FULL_QUANTUM, "red_cost_model": QuantumABFKSW20()},
    {"name": "Kyber768", "scheme": Kyber768, "mod_switch": True, "m": 1269, "n_guess": 251, "beta": 714, "s": 0.100, "quantum": QuantumMode.FULL_QUANTUM, "red_cost_model": QuantumABFKSW20()},
    {"name": "Kyber1024", "scheme": Kyber1024, "mod_switch": True, "m": 1675, "n_guess": 336, "beta": 967, "s": 0.100, "quantum": QuantumMode.FULL_QUANTUM, "red_cost_model": QuantumABFKSW20()},
]

# Quantum attack, quadratic speed up on DGS, quantum BKZ using quantum sieving
EUROCRYPT_PARAMS_QUANTUM_BCSS23 = [
    {"name": "Kyber512", "scheme": Kyber512, "mod_switch": False, "m": 1013, "n_guess": 1, "beta": 656, "s": 0.110, "quantum": QuantumMode.FULL_QUANTUM, "red_cost_model": BCSS23()},
    {"name": "Kyber768", "scheme": Kyber768, "mod_switch": False, "m": 1519, "n_guess": 1, "beta": 989, "s": 0.130, "quantum": QuantumMode.FULL_QUANTUM, "red_cost_model": BCSS23()},
    {"name": "Kyber1024", "scheme": Kyber1024, "mod_switch": False, "m": 2025, "n_guess": 1, "beta": 1392, "s": 0.130, "quantum": QuantumMode.FULL_QUANTUM, "red_cost_model": BCSS23()},
    {"name": "Kyber512", "scheme": Kyber512, "mod_switch": True, "m": 913, "n_guess": 166, "beta": 556, "s": 0.070, "quantum": QuantumMode.FULL_QUANTUM, "red_cost_model": BCSS23()},
    {"name": "Kyber768", "scheme": Kyber768, "mod_switch": True, "m": 1319, "n_guess": 246, "beta": 839, "s": 0.100, "quantum": QuantumMode.FULL_QUANTUM, "red_cost_model": BCSS23()},
    {"name": "Kyber1024", "scheme": Kyber1024, "mod_switch": True, "m": 1725, "n_guess": 336, "beta": 1152, "s": 0.100, "quantum": QuantumMode.FULL_QUANTUM, "red_cost_model": BCSS23()},
]

# Reproduce the estimates from Eurocrypt paper, does not run the optimizer but only
# the cost function
def reproduce_paper(params):
    results = {}
    for p in params:
        scheme_results = results.get(p["name"], {})
        use_mod_switch = p['mod_switch']
        mod_switch_name = "{} modulus switching".format("with" if use_mod_switch else "no")
        params = Parameters(
            Sampler.MCMC_ARCTIC,
            ModulusSwitching.EstimateOn if use_mod_switch else ModulusSwitching.Off,
            p['quantum'],
            p['red_cost_model'],
        )
        res = compute_cost(p["scheme"], p["m"], p["n_guess"], p["beta"], p["s"], params, True)
        print("{} ({}):".format(p["name"], mod_switch_name))
        print(res)
        scheme_results[use_mod_switch] = res
        results[p["name"]] = scheme_results
    return results

class ColumnType(Enum):
    INTEGER = 0
    FLOAT = 1
    LOG2_INTEGER = 2

COLUMNS = [
    ("attack", "rop", ColumnType.LOG2_INTEGER),
    ("m", "m", ColumnType.INTEGER),
    ("$n_\\guess$", "n_guess", ColumnType.INTEGER),
    ("$n_\\dual$", "n_dual", ColumnType.INTEGER),
    ("$\\beta$", "beta", ColumnType.INTEGER),
    ("s", "s", ColumnType.FLOAT),
]

CASES = [
    ("no modulus switching", False),
    ("with modulus switching", True)
]

ROWS = [
    ("Kyber512", "Kyber512"),
    ("Kyber768", "Kyber768"),
    ("Kyber1024", "Kyber1024"),
]

# Assume a array of the form: {scheme_name_1: result_1, ...}
# where each result_i is of the form: {case_1: subresult_1, ...}
# where subresult_i is a Cost structure
def results_table_latex(results, columns = COLUMNS, cases = CASES, rows = ROWS):
    col_headers = [col[0] for col in columns] * len(cases)
    all_rows = []
    for r in rows:
        if r[1] not in results:
            print("warning: row {} not found in results".format(r[1]))
            continue
        this_row = [r[0]]
        for case in cases:
            res = results[r[1]][case[1]]
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
