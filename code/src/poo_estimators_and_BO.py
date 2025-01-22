# Copyright (c) 2025 Nicolas Menet, Jonas Hübotter, Parnian Kassraie, Andreas Krause
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from functools import partial
from typing import Tuple, Callable
import jax
import jax.numpy as jnp
import src.gaussians as gaussians
import time

# -------------------------------------------------- Helpers --------------------------------------------------

@partial(jax.jit, static_argnames=['func', 'm']) # jitting generally does not help the performance due to func always changing
def _m_ary_root_search(func:Callable, l:float, r:float, accuracy:float, m:int):
    """
    Args:
        func (Callable): scalar-valued monotonously increasing function on the real line
        l (float): left border of m-ary search
        r (float): right border of m-ary search
        accuracy (float): the accuracy of binary search
        m (int): sets the number of parallel function evaluations (m-1) in m-ary search
    Returns:
        Tuple[float, float]: (a,b) such that |func((a+b)/2)| < accuracy
    """
    assert m >= 2, "m-ary search is only defined for an integer m > 1"
    min_float, max_float = jnp.finfo(jnp.float32).min*jnp.ones((1,)), jnp.finfo(jnp.float32).max*jnp.ones((1,))
    v_func = jax.vmap(func, in_axes = 0, out_axes = 0) # can now be applied to a whole vector
    @jax.jit
    def body_func(borders: Tuple[float, float]):
        subdivisions = jnp.linspace(start=borders[0], stop=borders[1], num=m+1, endpoint=True)
        evaluations = jnp.concatenate((min_float, v_func(subdivisions[1:-1]), max_float), dtype=jnp.float32)
        idx_l = jnp.argmax(jnp.where(evaluations <= 0, evaluations, -jnp.inf))
        idx_r = jnp.argmin(jnp.where(evaluations >= 0, evaluations, jnp.inf))
        return (subdivisions[idx_l], subdivisions[idx_r])
    @jax.jit
    def cond_func(borders: Tuple[float, float]):
        return abs(func((borders[0] + borders[1])/2)) > accuracy

    borders = jax.lax.while_loop(cond_func, body_func, (l, r))
    return borders

@jax.jit
def p_gaussian_geq_kappa(means:jax.Array, standard_deviations:jax.Array, kappa_:float) -> jax.Array:
    """Computes P[N[i] >= kappa] for N[i] ~ N(means[i], standard_deviations[i]^2)

    Args:
        means (jax.Array): (m,) 
        standard_deviations (jax.Array): (m,)
        kappa_ (float): (1,) threshold

    Returns:
        jax.Array: (m,) P[N[i] >= kappa] for i=1,...,m
    """
    c = (means - kappa_) / standard_deviations
    return jax.scipy.special.ndtr(c)

@jax.jit
def p_vapordist_geq_kappa(means:jax.Array, standard_deviations:jax.Array, kappa_:float) -> jax.Array:
    """Computes P[N[i] >= kappa] for (N[i]-means[i])/standard_deviations[i] with CDF exp(-(sqrt(x^2+4)-x)^2 / 8)

    Args:
        means (jax.Array): (m,) 
        standard_deviations (jax.Array): (m,)
        kappa_ (float): (1,) threshold

    Returns:
        jax.Array: (m,) P[N[i] >= kappa] for i=1,...,m
    """
    c = (means - kappa_) / standard_deviations
    return jnp.exp(-(jnp.sqrt(c**2 + 4)-c)**2 / 8)

@partial(jax.jit, static_argnames=['exploration_factor'])
def sum_of_p_gaussian_geq_kappa(means:jax.Array, standard_deviations:jax.Array, kappa_:float, exploration_factor:float = 0) -> jax.Array:
    """Computes sum_i P[N[i] >= kappa] for N[i] ~ N(means[i], standard_deviations[i]^2)

    Args:
        means (jax.Array): (m,) 
        standard_deviations (jax.Array): (m,)
        kappa_ (float): (1,) threshold
        exploration_factor (float): how much to bias the distribution towards more explorative behaviour when the optimal point is mostly assumed to be known

    Returns:
        jax.Array: (1,) sum_{i=1,...,m} P[N[i] >= kappa]
    """
    probs = p_gaussian_geq_kappa(means, standard_deviations, kappa_)
    return jnp.sum(probs/(1-exploration_factor*probs))

@jax.jit
def sum_of_p_vapordist_geq_kappa(means:jax.Array, standard_deviations:jax.Array, kappa_:float) -> jax.Array:
    """Computes sum_i P[N[i] >= kappa] for (N[i]-means[i])/standard_deviations[i] with CDF exp(-(sqrt(x^2+4)-x)^2 / 8)

    Args:
        means (jax.Array): (m,) 
        standard_deviations (jax.Array): (m,)
        kappa_ (float): (1,) threshold

    Returns:
        jax.Array: (1,) sum_{i=1,...,m} P[N[i] >= kappa]
    """
    probs = p_vapordist_geq_kappa(means, standard_deviations, kappa_)
    return jnp.sum(probs)

@partial(jax.jit, static_argnames=['vapor', 'exploration_factor'])
def find_normalising_threshold(means:jax.Array, standard_deviations:jax.Array, alpha:float, vapor:bool = False, exploration_factor:float = 0) -> Tuple[float, float]:
    """Computes kappa such that sum_i P[N[i] >= kappa] ~= 1 for N[i] ~ N(means[i], standard_deviations[i]^2) if vapor is False and 
       if vapor is True for (N[i]-means[i])/standard_deviations[i] with CDF exp(-(sqrt(x^2+4)-x)^2 / 8)

    Args:
        means (jax.Array): (m,)
        standard_deviations (jax.Array): (m,)
        alpha (float): sets precision of binary search to 1/(alpha |X|)
        vapor (bool): whether to use the VAPOR distribution instead of the Gaussian in CME
        exploration_factor (float): how much to bias the distribution towards more explorative behaviour when the optimal point is mostly assumed to be known,
        results in P[N[i] >= kappa]/(1-exploration_factor * P[N[i] >= kappa]) instead of P[N[i] >= kappa].

    Returns:
        Tuple[float, float]: [kappa_lower_bound, kappa_upper_bound]
    """
    assert exploration_factor == 0 or vapor == False, "VAPOR is not set up for additional exploration"
    min_mu = jnp.min(means)
    min_sigma = jnp.min(standard_deviations)
    max_mu = jnp.max(means)
    max_sigma = jnp.max(standard_deviations)
    domain_size = jnp.shape(means)[0]
    eta = 1/(alpha * domain_size)
    if vapor:
        beta = (1/jnp.sqrt(2*jnp.log(domain_size)) - jnp.sqrt(2*jnp.log(domain_size)))
    else:
        beta = jax.scipy.special.ndtri(1/(domain_size + exploration_factor)) # inverse CDF
    kappa_lower_bound = min_mu - beta * min_sigma
    kappa_upper_bound = max_mu - beta * max_sigma
    if exploration_factor > 0:
        kappa_lower_bound = jnp.maximum(kappa_lower_bound, jnp.max(means - standard_deviations * jax.scipy.special.ndtri(1/(1+exploration_factor)))) # second factor ensures that no single entry will have a probability exceeding 1

    func = lambda a: p_gaussian_geq_kappa(means, standard_deviations, a) if not vapor \
            else p_vapordist_geq_kappa(means, standard_deviations, a) # element-wise monotonously decreasing in kappa_


    @jax.jit
    def body_func(kappa_search_window: Tuple[float, float]):
        kappa_low, kappa_up = kappa_search_window
        kappa = (kappa_low + kappa_up) / 2
        probs = func(kappa) 
        probs /= (1-exploration_factor*probs)
        normalisation_delta = 1 - jnp.sum(probs) # monotonously increasing in kappa

        kappa_low = jnp.where(normalisation_delta < 0, kappa, kappa_low)
        kappa_up = jnp.where(normalisation_delta >= 0, kappa, kappa_up)

        return kappa_low, kappa_up
    @jax.jit
    def cond_func(kappa_search_window: Tuple[float, float]):
        kappa_low, kappa_up = kappa_search_window
        probs_low = func(kappa_up)
        probs_up = func(kappa_low)
        probs_low /= (1-exploration_factor*probs_low)
        probs_up  /= (1-exploration_factor*probs_up)
        return jnp.max(probs_up - probs_low) >= eta

    kappa_low, kappa_up = jax.lax.while_loop(cond_func, body_func, (kappa_lower_bound, kappa_upper_bound))

    return kappa_low, kappa_up

# -------------------------------------------------- Estimating F^* --------------------------------------------------

@partial(jax.jit, static_argnames=['number_of_samples', 'resolution'])
def etse_F_star(gaussian_means:jax.Array, gaussian_cov:jax.Array, number_of_samples:int, resolution:int, random_key:jax.Array) -> Tuple[jax.Array, jax.Array]:
    """Computes the exhaustive thompson sampling estimator (E-TSE) for p(F^*),
       i.e. an unbiased estimator

    Args:
        gaussian_means (jax.Array): (m,)
        gaussian_cov (jax.Array): (m, m)
        number_of_samples (int): corresponds to n
        resolution (int): the number of bins used in the histogram to compute p(F^*), i.e. its resolution
        random_key (jax.Array): a single random key that seeds the sampling

    Returns:
        Tuple[jax.Array, jax.Array]: the histogram bin centers and (normalised) values for F^*
    """
    gaussian_sqrt_covs = gaussians.spsd_matrix_square_root(gaussian_cov)
    maximums = gaussians.sample_n_max(gaussian_means, gaussian_sqrt_covs, number_of_samples, random_key)
    density, bin_edges = jnp.histogram(maximums, bins=resolution, density=True)
    bin_centres = (bin_edges[1:] + bin_edges[:-1])/2
    return density, bin_centres

@partial(jax.jit, static_argnames=['num'])
def numerical_F_star_expectation(gaussian_means:jax.Array, gaussian_stds:jax.Array, num:int) -> Tuple[float, float]:
    """Computes the expectation of F^*|D assuming independence across F_z|D for z in X
       based on E[F^* | D] = int_0^\infty P[F^* > f | D] - P[F^* < -f | D] df ~= int_0^\infty P[F^* > f | D] df
    Args:
        gaussian_means (jax.Array): (m,)
        gaussian_stds (jax.Array): (m,)
        num (int): the number of integration points for numerical integration or the number of samples in case of a sampling approach

    Returns:
        Tuple[float, float]: lower and upper bound on the integral neglecting the negative component (-P[F^* < -f | D])
    """
    # figure out start and end of integration (of only positive component in actual integrand) up to a factor 1.1
    @jax.jit
    def body_func(state):
        f_last, f = state[0], state[1]
        positive_term = 1 - jnp.prod(jax.scipy.special.ndtr((f-gaussian_means)/gaussian_stds))
        f_last = jax.lax.cond(positive_term == 0, lambda t: t[1], lambda t: t[0], (f_last, f))
        f *= 1.1
        return f_last, f
    @jax.jit
    def cond_func(state):
        f_last = state[0]
        return f_last == jnp.inf
    f_last, _ = jax.lax.while_loop(cond_func, body_func, (jnp.inf, 1.0))
    f_first = 0

    evaluation_points = jnp.linspace(start=f_first, stop=f_last, num=num, endpoint=True)
    positive_terms = 1 - jnp.prod(jax.scipy.special.ndtr((jnp.expand_dims(evaluation_points, axis=1)-gaussian_means)/gaussian_stds), axis=1)

    # lower bound on integral
    r_low = jnp.sum(positive_terms[1:]) * (f_last - f_first) / (num-1) + f_first
    # upper bound on integral
    r_up = jnp.sum(positive_terms[:-1]) * (f_last - f_first) / (num-1) + f_first

    return (r_low, r_up)

@partial(jax.jit, static_argnames=['num'])
def empirical_F_star_expectation(gaussian_means:jax.Array, gaussian_stds:jax.Array, random_key, num:int) -> Tuple[float, float]:
    """Computes the empirical expectation of F^*|D assuming independence across F_z|D for z in X, as well as an estimate of the standard deviation of the expectation using
       Var[(f_1 + f_2 + … + f_n)/n | D) = sum_i=1^n Var[f_i / n | D] = sum_{i=1}^n Var[f_i | D] / n^2 = Var[f_i | D] / n
    Args:
        gaussian_means (jax.Array): (m,)
        gaussian_stds (jax.Array): (m,)
        num (int): the number of integration points for numerical integration or the number of samples in case of a sampling approach

    Returns:
        Tuple[float, float]: mean, std
    """
    f_maxs = gaussians.sample_n_max_independent(gaussian_means, gaussian_stds, num, random_key, 10)
    mean = jnp.mean(f_maxs)
    #Var[(f_1 + f_2 + … + f_n)/n | D) = sum_i=1^n Var[f_i / n | D] = sum_{i=1}^n Var[f_i | D] / n^2 = Var[f_i | D] / n for i.i.d. f_i ~ f^* | D
    var = jnp.var(f_maxs, ddof=1) / num
    return mean, var**.5

# -------------------------------------------------- Acquisition functions based on estimating X^* --------------------------------------------------

#@partial(jax.jit, static_argnames=['number_of_samples'])
#def ose_poo(gaussian_means:jax.Array, gaussian_cov:jax.Array, number_of_samples:int, random_key:jax.Array, unroll:int=1) -> jax.Array:
#    assert number_of_samples >= 2, "at least two sample are required for the OSE estimator" 
#    cov_sqrt = gaussians.spsd_matrix_square_root(gaussian_cov)
#    f_samples = gaussians.sample_n_max(gaussian_means, cov_sqrt, number_of_samples, random_key, unroll)
#    sorted_f_samples = jnp.sort(f_samples)

#    partition_boundaries = (sorted_f_samples[1:] + sorted_f_samples[:-1]) / 2
#    a_i = jnp.concatenate((-jnp.inf, partition_boundaries), axis=None)
#    b_i = jnp.concatenate((partition_boundaries, jnp.inf), axis=None)
    # P[F^* \in [a_i, b_i] | D] = 1 / number_of_samples, since we shrunk the intervals such that we only have one sample per interval
#    p_gaussian_geq_kappa()

#def cmes_poo(gaussian_means:jax.Array, gaussian_cov:jax.Array, number_of_samples:int, random_key:jax.Array, unroll:int=1):

@partial(jax.jit, static_argnames=['number_of_samples', 'unroll'])
def etse_poo(gaussian_means:jax.Array, gaussian_cov:jax.Array, number_of_samples:int, random_key:jax.Array, unroll:int=1) -> jax.Array:
    """Computes the exhaustive thompson sampling estimator (E-TSE) for probability of optimality (poo), 
       i.e. an unbiased estimator

    Args:
        gaussian_means (jax.Array): (m,)
        gaussian_cov (jax.Array): (m, m)
        number_of_samples (int): corresponds to n
        random_key (jax.Array): a single random key that seeds the sampling
        unroll (int): the number of times the argmax sampling loop is unrolled (more parallel computation with more memory consumption)
    Returns:
        jax.Array: (m, ) the probabilities of optimality (summing to 1)
    """
    gaussian_sqrt_covs = gaussians.spsd_matrix_square_root(gaussian_cov)
    #assert sigma_min != 0, "singular covariance matrix not allowed (may lead to non-unique maximum). Try decreasing the domain_size or length_scale."
    maximiser_counts = gaussians.count_sample_n_arg_max(gaussian_means, gaussian_sqrt_covs, number_of_samples, random_key, unroll)
    r = maximiser_counts / number_of_samples # ensures integrability, evens out rounding errors
    return r

# cannot be jitted because the number of evaluation points depends on the entries of means and stds
def ie_poo_guaranteed(gaussian_means:jax.Array, gaussian_stds:jax.Array, alpha:float, unroll=1) -> jax.Array:
    """Computes the independence estimator (IE) for probability of optimality (poo) with guaranteed runtime for the desired accuracy. Compared to
    ie_poo_parallel_guaranteed this implementation is more sequential but only requires O(|X|) memory instead of O(|X|^2)

    Args:
        gaussian_means (jax.Array): (m,)
        gaussian_stds (jax.Array): (m,)
        alpha (float): reciprocal of desired relative accuracy (abs accuracy <= 1/(|X| alpha))
        unroll (int): the number of times the loop is unrolled (more parallel computation & more memory consumption)
    Returns:
        jax.Array: (m, ) the probabilities of optimality (summing to 1)
    """
    domain_size = jnp.shape(gaussian_means)[0]
    eta = 1/(alpha * domain_size)
    assert eta <= 1/4, "least eta for the approximation guarantees (abs accuracy <= 1/(|X| alpha)) to hold according to the theorem in the article"

    means = gaussian_means
    stds = gaussian_stds
    min_mean = jnp.min(means)
    max_mean = jnp.max(means)
    min_std = jnp.min(stds)
    max_std = jnp.max(stds)

    beta = -jax.scipy.special.ndtri(2/(alpha*domain_size))

    f_1 = min_mean - beta * max_std
    f_l_min_1 = max_mean + beta * max_std
    f_i = jnp.linspace(start=f_1, stop=f_l_min_1, num=int(jnp.ceil(domain_size * alpha * (f_l_min_1 - f_1) / (2 * (2*jnp.pi)**.5 * min_std))+1), endpoint=True) # i = 1,...,l-1

    # run time O(|X|^2), memory consumption O(|X|)
    sum_log_prob_F_z_leq_f_i = jnp.zeros_like(f_i) # i = 1,...,l-1
    @jax.jit
    def body_func(i:int, l:jax.Array):
        return l.at[i].set(jnp.sum(jax.scipy.special.log_ndtr((f_i[i] - means)/stds)))
    sum_log_prob_F_z_leq_f_i = jax.lax.fori_loop(0, len(f_i), body_func, sum_log_prob_F_z_leq_f_i, unroll=min(len(f_i), unroll))
    
    # run time O(|X|^2), memory consumption O(|X|)
    probs_of_optimality = jnp.zeros_like(means) # x = 1,...,|X|
    @jax.jit
    def body_func2(i:int, l:jax.Array):
        log_prob_F_x_leq_f_i = jax.scipy.special.log_ndtr((f_i - means[i])/stds[i])
        log_g_x_at_f_i = sum_log_prob_F_z_leq_f_i - log_prob_F_x_leq_f_i 
        g_x_at_f_i = jnp.exp(log_g_x_at_f_i)
        trapezoidal_means = (g_x_at_f_i[1:] + g_x_at_f_i[:-1])/2 # l-2 segments for i
        trapezoidal_means = jnp.concatenate(((g_x_at_f_i[:1]+0)/2, trapezoidal_means, (1+g_x_at_f_i[-1:])/2)) # l segments for i
        prob_F_x_leq_f_i = jnp.exp(log_prob_F_x_leq_f_i)
        trapezoidal_measures = jnp.concatenate((prob_F_x_leq_f_i[:1], prob_F_x_leq_f_i[1:] - prob_F_x_leq_f_i[:-1], 1-prob_F_x_leq_f_i[-1:])) # l segments for i
        return l.at[i].set(jnp.sum(trapezoidal_means * trapezoidal_measures))
    probs_of_optimality = jax.lax.fori_loop(0, means.size, body_func2, probs_of_optimality, unroll=min(means.size, unroll))
    probs_of_optimality /= jnp.sum(probs_of_optimality) # ensures integrability, evens out rounding errors
    return probs_of_optimality

# cannot be jitted because the number of evaluation points depends on the entries of means and stds
def ie_poo_parallel_guaranteed(gaussian_means:jax.Array, gaussian_stds:jax.Array, alpha:float) -> jax.Array:
    """Computes the independence estimator (IE) for probability of optimality (poo) with guaranteed runtime for the desired accuracy, based on vector operations.
    More suitable for hardware accelerators, but memory scales O(|X|^2) instead of O(|X|)

    Args:
        gaussian_means (jax.Array): (m,)
        gaussian_stds (jax.Array): (m,)
        alpha (float): reciprocal of desired relative accuracy (abs accuracy <= 1/(|X| alpha))

    Returns:
        jax.Array: (m, ) the probabilities of optimality (summing to 1)
    """
    domain_size = jnp.shape(gaussian_means)[0]
    eta = 1/(alpha * domain_size)
    assert eta <= 1/4, "least eta for the approximation guarantees (abs accuracy <= 1/(|X| alpha)) to hold according to the theorem in the article"

    means = gaussian_means
    stds = gaussian_stds
    min_mean = jnp.min(means)
    max_mean = jnp.max(means)
    min_std = jnp.min(stds)
    max_std = jnp.max(stds)

    beta = -jax.scipy.special.ndtri(2/(alpha*domain_size))

    f_1 = min_mean - beta * max_std
    f_l_min_1 = max_mean + beta * max_std
    f_i = jnp.linspace(start=f_1, stop=f_l_min_1, num=int(jnp.ceil(domain_size * alpha * (f_l_min_1 - f_1) / (2 * (2*jnp.pi)**.5 * min_std))+1), endpoint=True) # i = 1,...,l-1

    # for all z \in X (axis=0) and f_i with i = 1,...,l-1  (axis=1)
    log_prob_F_z_leq_f_i = jax.scipy.special.log_ndtr((jnp.expand_dims(f_i, axis=0) - jnp.expand_dims(means, axis=1))/jnp.expand_dims(stds, axis=1))

     # for all f_i with i = 1,...,l-1  (axis=1)
    sum_log_prob_F_z_leq_f_i = jnp.sum(log_prob_F_z_leq_f_i, axis=0)

    # for all z \in X (axis=0) and f_i with i = 1,...,l-1  (axis=1)
    log_g_x_at_f_i = sum_log_prob_F_z_leq_f_i - log_prob_F_z_leq_f_i
    g_x_at_f_i = jnp.exp(log_g_x_at_f_i)

    # for all z \in X (axis=0) and f_i
    trapezoidal_means = (g_x_at_f_i[:, 1:] + g_x_at_f_i[:, :-1])/2 # l-2 segments for i
    trapezoidal_means = jnp.concatenate(((g_x_at_f_i[:, :1]+0)/2, trapezoidal_means, (1+g_x_at_f_i[:, -1:])/2), axis=1) # l segments for i

    # for all z \in X (axis=0) and i = 1,...,l-1  (axis=1)
    prob_F_z_leq_f_i = jnp.exp(log_prob_F_z_leq_f_i)

    # for all z \in X (axis=0) and i
    trapezoidal_measures = jnp.concatenate((prob_F_z_leq_f_i[:, :1], prob_F_z_leq_f_i[:, 1:] - prob_F_z_leq_f_i[:, :-1], 1-prob_F_z_leq_f_i[:, -1:]), axis=1) # l segments for i

    # integrate
    r = jnp.sum(trapezoidal_means * trapezoidal_measures, axis=1)
    r /= jnp.sum(r) # ensures integrability, evens out rounding errors
    return r

@partial(jax.jit, static_argnames=['num', 'unroll'])
def _ie_poo_integral_helper(means:jax.Array, stds:jax.Array, f_1:float, f_l_min_1:float, num:int, unroll:int):
    f_i = jnp.linspace(start=f_1, stop=f_l_min_1, num=num, endpoint=True) # i = 1,...,l-1

    sum_log_prob_F_z_leq_f_i = jnp.zeros_like(f_i) # i = 1,...,l-1
    @jax.jit
    def body_func(i:int, l:jax.Array):
        return l.at[i].set(jnp.sum(jax.scipy.special.log_ndtr((f_i[i] - means)/stds)))
    sum_log_prob_F_z_leq_f_i = jax.lax.fori_loop(0, len(f_i), body_func, sum_log_prob_F_z_leq_f_i, unroll=min(f_i.size, unroll))
    
    probs_of_optimality_up = jnp.zeros_like(means) # x = 1,...,|X|
    @jax.jit
    def body_func2(i:int, l:jax.Array):
        log_prob_F_x_leq_f_i = jax.scipy.special.log_ndtr((f_i - means[i])/stds[i])
        log_g_x_at_f_i = sum_log_prob_F_z_leq_f_i - log_prob_F_x_leq_f_i 
        g_x_at_f_i = jnp.exp(log_g_x_at_f_i)
        trapezoidal_maxs = jnp.concatenate((g_x_at_f_i, jnp.ones((1,)))) # l segments for i
        prob_F_x_leq_f_i = jnp.exp(log_prob_F_x_leq_f_i)
        trapezoidal_measures = jnp.concatenate((prob_F_x_leq_f_i[:1], prob_F_x_leq_f_i[1:] - prob_F_x_leq_f_i[:-1], 1-prob_F_x_leq_f_i[-1:])) # l segments for i
        return l.at[i].set(jnp.sum(trapezoidal_maxs * trapezoidal_measures))
    probs_of_optimality_up = jax.lax.fori_loop(0, means.size, body_func2, probs_of_optimality_up, unroll=min(means.size, unroll))

    probs_of_optimality_low = jnp.zeros_like(means) # x = 1,...,|X|
    @jax.jit
    def body_func3(i:int, l:jax.Array):
        log_prob_F_x_leq_f_i = jax.scipy.special.log_ndtr((f_i - means[i])/stds[i])
        log_g_x_at_f_i = sum_log_prob_F_z_leq_f_i - log_prob_F_x_leq_f_i 
        g_x_at_f_i = jnp.exp(log_g_x_at_f_i)
        trapezoidal_mins = jnp.concatenate((jnp.zeros((1,)), g_x_at_f_i)) # l segments for i
        prob_F_x_leq_f_i = jnp.exp(log_prob_F_x_leq_f_i)
        trapezoidal_measures = jnp.concatenate((prob_F_x_leq_f_i[:1], prob_F_x_leq_f_i[1:] - prob_F_x_leq_f_i[:-1], 1-prob_F_x_leq_f_i[-1:])) # l segments for i
        return l.at[i].set(jnp.sum(trapezoidal_mins * trapezoidal_measures))
    probs_of_optimality_low = jax.lax.fori_loop(0, means.size, body_func3, probs_of_optimality_low, unroll=min(means.size, unroll))

    return probs_of_optimality_low, probs_of_optimality_up # lower bound and upper bound

# cannot be jitted because the number of evaluation points depends on the entries of means and stds
def ie_poo(gaussian_means:jax.Array, gaussian_stds:jax.Array, alpha:float, unroll=1, start_num=10_000) -> jax.Array:
    """Computes the independence estimator (IE) for probability of optimality (poo). Compared to
    ie_poo_guaranteed this implementation does not guarantee a runtime complexity, but is in practice much faster due to
    the possibility of jitting subroutines and adaptive integration (selects number of integration points as needed)

    Args:
        gaussian_means (jax.Array): (m,)
        gaussian_stds (jax.Array): (m,)
        alpha (float): reciprocal of desired relative accuracy (abs accuracy <= 1/(|X| alpha))
        unroll (int): the number of times the loop is unrolled (more parallel computation & more memory consumption)
        start_num (int): with how many integration points to start (will always double if not accurate enough)
    Returns:
        jax.Array: (m, ) the probabilities of optimality (summing to 1)
    """
    min_mean = jnp.min(gaussian_means)
    max_mean = jnp.max(gaussian_means)
    max_std = jnp.max(gaussian_stds)
    domain_size = gaussian_means.size

    beta = -jax.scipy.special.ndtri(2/(alpha*domain_size))
    f_1 = min_mean - beta * max_std
    f_l_min_1 = max_mean + beta * max_std

    num = start_num
    while True:
        probs_of_optimality_low, probs_of_optimality_up = _ie_poo_integral_helper(gaussian_means, gaussian_stds, f_1, f_l_min_1, num, unroll)
        if jnp.max(jnp.abs(probs_of_optimality_up - probs_of_optimality_low))/2 < 1/(alpha * domain_size):
            break
        num *= 2
        print(f"integration error in IE too large, repeat with {num} integration points")

    probs_of_optimality = (probs_of_optimality_up + probs_of_optimality_low)/2
    probs_of_optimality /= jnp.sum(probs_of_optimality) # ensures integrability, evens out rounding errors
    return probs_of_optimality

@partial(jax.jit, static_argnames=['samples', 'unroll'])
def nies_poo(gaussian_means:jax.Array, gaussian_stds:jax.Array, samples:int, random_key:jax.Array, unroll=1, ) -> jax.Array:
    """Computes the normal independence estimator (NIE) for probability of optimality (poo) using Monte Carlo sampling to estimate mu_x and sigma_x.
    Args:
        gaussian_means (jax.Array): (m,)
        gaussian_stds (jax.Array): (m,)
        samples (int): the number of samples from (X^*, F^*) that are used to estimate first and second moments of g_x
        random_key (jax.Array): random key to seed sampling
        unroll (int): the number of times the innermost loops are unrolled (more parallel computation & more memory consumption)
    Returns:
        jax.Array: (m, ) the probabilities of optimality (summing to 1)
    """
    sampled_x_stars_and_f_stars = gaussians.sample_n_arg_max_and_max_independent(gaussian_means, gaussian_stds, random_key, samples, True, unroll) # (samples, 2, 2), consumes O(unroll * |X|) memory and O(samples * |X|) compute
    x_stars = sampled_x_stars_and_f_stars[:, 0, 0]
    f_stars = sampled_x_stars_and_f_stars[:, 0, 1]
    f_starsstars = sampled_x_stars_and_f_stars[:, 1, 1]

    @jax.jit
    def body_func(x:int, poos:jax.Array):
        f_stars_w_o_x = jnp.where(x == x_stars, f_starsstars, f_stars) #corresponds to f_i^*(x), where (x, F_x) is always excluded from (X^*, F^*)
        mu_x = jnp.mean(f_stars_w_o_x)
        sigma_x_squared = jnp.var(f_stars_w_o_x, ddof=1)
        return poos.at[x].set(jax.scipy.special.ndtr((gaussian_means[x] - mu_x)/jnp.sqrt(gaussian_stds[x]**2 + sigma_x_squared)))

    poos = jax.lax.fori_loop(0, gaussian_means.shape[0], body_func, jnp.zeros_like(gaussian_means), unroll=min((int(gaussian_means.shape[0]/samples)+1)*unroll, gaussian_means.shape[0]))
    # the unrolling ensures that we have the same degree of parallelism/memory consumption for the sampling process and the computation of f_stars_w_o_x,
    # hence a single "unroll" parameter can reasonably control both

    return poos / jnp.sum(poos) # ensures integrability, evens out rounding errors

@partial(jax.jit, static_argnames=['samples', 'unroll'])
def output_space(gaussian_means:jax.Array, gaussian_cov:jax.Array, samples:int, random_key:jax.Array, unroll=1, ) -> jax.Array:
    """Computes the output space estimator for probability of optimality (poo).
    Args:
        gaussian_means (jax.Array): (m,)
        gaussian_cov (jax.Array): (m,m)
        samples (int): the number of samples from (X^*, F^*) that are used to estimate first and second moments of g_x
        random_key (jax.Array): random key to seed sampling
        unroll (int): the number of times the innermost loops are unrolled (more parallel computation & more memory consumption)
    Returns:
        jax.Array: (m, ) the probabilities of optimality (summing to 1)
    """
    
    gaussian_sqrt_covs = gaussians.spsd_matrix_square_root(gaussian_cov)
    gaussian_stds = jnp.diag(gaussian_cov)**.5
    sampled_f_stars_and_products = gaussians.sample_n_max_w_products(gaussian_means, gaussian_sqrt_covs, random_key, samples, unroll)
    x_stars = sampled_f_stars_and_products[:, 0]
    f_stars = sampled_f_stars_and_products[:, 1]
    f_stars_products = sampled_f_stars_and_products[:, 2:2+int(gaussian_means.size)]
    f_starstars = sampled_f_stars_and_products[:, 2+int(gaussian_means.size)]
    f_starstars_products = sampled_f_stars_and_products[:, 3+int(gaussian_means.size):]

    @jax.jit
    def body_func(x:int, poos:jax.Array):
        mu_F_x = gaussian_means[x]
        f_stars_w_o_x = jnp.where(x == x_stars, f_starstars, f_stars) #corresponds to f_i^*(x), where (x, F_x) is always excluded from (X^*, F^*)
        mean_f_star_w_o_x = jnp.mean(f_stars_w_o_x)
        std_f_star_w_o_x_squared = jnp.var(f_stars_w_o_x, ddof=1)
        f_stars_products_w_o_x = jnp.where(x == x_stars, f_starstars_products[:, x], f_stars_products[:, x])
        empirical_correlation = jnp.mean(f_stars_products_w_o_x)
        empirical_covariance = empirical_correlation - mean_f_star_w_o_x * mu_F_x
        
        return poos.at[x].set(jax.scipy.special.ndtr((mu_F_x - mean_f_star_w_o_x)/jnp.sqrt(jnp.maximum(gaussian_stds[x]**2 - 2 * empirical_covariance + std_f_star_w_o_x_squared, 0))))

    poos = jax.lax.fori_loop(0, gaussian_means.shape[0], body_func, jnp.zeros_like(gaussian_means), unroll=min((int(gaussian_means.shape[0]/samples)+1)*unroll, gaussian_means.shape[0]))
    # the unrolling ensures that we have the same degree of parallelism/memory consumption for the sampling process and the computation of f_stars_w_o_x,
    # hence a single "unroll" parameter can reasonably control both

    return poos / jnp.sum(poos) # ensures integrability, evens out rounding errors

@partial(jax.jit, static_argnames=['simplified'])
def nie_poo(gaussian_means:jax.Array, gaussian_stds:jax.Array, alpha:float, simplified=False) -> jax.Array:
    """Computes the normal independence estimator (NIE) for probability of optimality (poo) using quartile fitting with binary search to estimate mu_x and sigma_x. 
    If simplified == True, only NIE-I is computed

    Args:
        gaussian_means (jax.Array): (m,)
        gaussian_stds (jax.Array): (m,)
        alpha (float): (reciprocal of desired relative accuracy (i.e. absolute accuracy eta <= 1/(|X| alpha)))
        simplified (bool): if set to true, only uses NIE-I. Otherwise, both NIE-I and NIE-II are computed and combined 

    Returns:
        jax.Array: (m,) the probabilities of optimality (summing to 1)
    """
    
    domain_size = gaussian_means.size
    assert domain_size > 1, "|X| must be at least 2 to run NIE, needed for validity of initial search window in logarithmic search"
    eta = 1 / (alpha * domain_size)
    log_quarter = jnp.log(0.25)
    log_three_quarters = jnp.log(0.75)

    log_g = lambda f: jnp.sum(jax.scipy.special.log_ndtr((f-gaussian_means)/gaussian_stds)) # by moving the next 11 lines outside of NIE_I_S we save a bit of compute, because neither the search window initialisation nor log_g changes

    mu_F_min = jnp.min(gaussian_means)
    mu_F_max = jnp.max(gaussian_means)
    sigma_F_min = jnp.min(gaussian_stds)
    sigma_F_max = jnp.max(gaussian_stds)

    q_1_low_init = mu_F_min + sigma_F_min * jax.scipy.special.ndtri(0.25**(1/domain_size))
    q_1_up_init  = mu_F_max + sigma_F_max * jax.scipy.special.ndtri(0.25**(1/domain_size))
    q_3_low_init = mu_F_min + sigma_F_min * jax.scipy.special.ndtri(0.75**(1/domain_size))
    q_3_up_init  = mu_F_max + sigma_F_max * jax.scipy.special.ndtri(0.75**(1/domain_size))

    def nie_I_S(m:int, mu_F:jax.Array, sigma_F:jax.Array):
        def body_func(i:int, bounds_on_qs:Tuple[float, float, float, float]):
            q_1_low, q_1_up, q_3_low, q_3_up = bounds_on_qs
            q_1 = (q_1_up + q_1_low) / 2
            q_3 = (q_3_up + q_3_low) / 2
            log_g_1 = log_g(q_1)
            log_g_3 = log_g(q_3)
            q_1_up  = jnp.where(log_g_1 >  log_quarter, q_1, q_1_up)
            q_1_low = jnp.where(log_g_1 <= log_quarter, q_1, q_1_low)
            q_3_up  = jnp.where(log_g_3 >  log_three_quarters, q_3, q_3_up)
            q_3_low = jnp.where(log_g_3 <= log_three_quarters, q_3, q_3_low)
            return q_1_low, q_1_up, q_3_low, q_3_up

        q_1_low, q_1_up, q_3_low, q_3_up = jax.lax.fori_loop(0, m, body_func, (q_1_low_init, q_1_up_init, q_3_low_init, q_3_up_init))

        mu_up     = (q_3_up  + q_1_up)  / 2
        mu_low    = (q_3_low + q_1_low) / 2
        sigma_up  = (q_3_up - q_1_low) / (2 * jax.scipy.special.ndtri(0.75))
        sigma_low = (q_3_low - q_1_up) / (2 * jax.scipy.special.ndtri(0.75))

        return mu_up, mu_low, sigma_up, sigma_low

    def nie_II_S(m:int, mu_up, mu_low, sigma_up, sigma_low, mu_F:jax.Array, sigma_F:jax.Array,):
        mu_tilde_up  = jnp.maximum(mu_up,  mu_F_max) # should always hold mathematically, so shouldn't have any effect.
        mu_tilde_low = jnp.maximum(mu_low, mu_F_max) # If it doesn't hold automatically, there must be numerical issues
        sigma_tilde_xs_up  = jnp.minimum(sigma_up,  sigma_F)
        sigma_tilde_xs_low = jnp.minimum(sigma_low, sigma_F)

        q_1_xs_low = jnp.minimum(mu_F - jnp.sqrt(2) * sigma_F, jnp.maximum((mu_tilde_low + mu_F)/2 - sigma_F**2 * jnp.log(2/0.25) / (mu_tilde_low - mu_F), mu_tilde_low - jnp.sqrt(2 * jnp.log(2/0.25) / (1-(sigma_tilde_xs_up / sigma_F)**2)) * sigma_tilde_xs_up)) # using that jax interprets 1/0 = inf
        q_1_xs_up  = mu_tilde_up + sigma_tilde_xs_low * jax.scipy.special.ndtri(0.25)
        q_3_xs_low = jnp.minimum(mu_F - jnp.sqrt(2) * sigma_F, jnp.maximum((mu_tilde_low + mu_F)/2 - sigma_F**2 * jnp.log(2/0.75) / (mu_tilde_low - mu_F), mu_tilde_low - jnp.sqrt(2 * jnp.log(2/0.75) / (1-(sigma_tilde_xs_up / sigma_F)**2)) * sigma_tilde_xs_up)) # using that jax interprets 1/0 = inf
        q_3_xs_up  = mu_tilde_up + sigma_tilde_xs_up  * jax.scipy.special.ndtri(0.75)

        def body_func(i:int, bounds_on_qs_and_memory_placeholders:Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]):
            # the implementation may look a bit cumbersome, but it ensures that no additional memory is allocated in each run. This is important to ensure fast execution.
            q_1_xs_low, q_1_xs_up, q_3_xs_low, q_3_xs_up, mem1, mem2, mem3, mem4, mem5, mem6, mem7, mem8 = bounds_on_qs_and_memory_placeholders # last 6 elements are just memory placeholders
            q_1_xs, q_3_xs, log_g_1_xs_up, log_g_1_xs_low, log_g_3_xs_up, log_g_3_xs_low, temp_buffer1, temp_buffer2 = mem1, mem2, mem3, mem4, mem5, mem6, mem7, mem8

            q_1_xs = q_1_xs_up
            q_1_xs += q_1_xs_low
            q_1_xs /= 2

            q_3_xs = q_3_xs_up
            q_3_xs += q_3_xs_low
            q_3_xs /= 2
            
            log_g_1_xs_up = q_1_xs
            log_g_1_xs_up -= mu_tilde_low
            log_g_1_xs_up /= sigma_tilde_xs_up
            log_g_1_xs_up = jax.scipy.special.log_ndtr(log_g_1_xs_up)
            temp_buffer1 = q_1_xs
            temp_buffer1 -= mu_F
            temp_buffer1 /= sigma_F
            temp_buffer1 = jax.scipy.special.log_ndtr(temp_buffer1)
            log_g_1_xs_up -= temp_buffer1
            temp_buffer2 = q_1_xs
            temp_buffer2 -= mu_tilde_low
            temp_buffer2 /= sigma_tilde_xs_low
            temp_buffer2 = jax.scipy.special.log_ndtr(temp_buffer2)
            temp_buffer2 -= temp_buffer1
            log_g_1_xs_up = jnp.maximum(log_g_1_xs_up, temp_buffer2)
            #log_g_1_xs_up  = log_g_1_xs_up.at[:].set(jnp.maximum(jax.scipy.special.log_ndtr((q_1_xs - mu_tilde_low)/sigma_tilde_xs_up)  - jax.scipy.special.log_ndtr((q_1_xs - mu_F)/sigma_F),
            #                             jax.scipy.special.log_ndtr((q_1_xs - mu_tilde_low)/sigma_tilde_xs_low) - jax.scipy.special.log_ndtr((q_1_xs - mu_F)/sigma_F)))
            log_g_1_xs_low = q_1_xs
            log_g_1_xs_low -= mu_tilde_up
            log_g_1_xs_low /= sigma_tilde_xs_up
            log_g_1_xs_low = jax.scipy.special.log_ndtr(log_g_1_xs_low)
            log_g_1_xs_low -= temp_buffer1
            temp_buffer2 = q_1_xs
            temp_buffer2 -= mu_tilde_up
            temp_buffer2 /= sigma_tilde_xs_low
            temp_buffer2 = jax.scipy.special.log_ndtr(temp_buffer2)
            temp_buffer2 -= temp_buffer1
            log_g_1_xs_low = jnp.minimum(log_g_1_xs_low, temp_buffer2)
            #log_g_1_xs_low = log_g_1_xs_low.at[:].set(jnp.minimum(jax.scipy.special.log_ndtr((q_1_xs - mu_tilde_up) /sigma_tilde_xs_up)  - jax.scipy.special.log_ndtr((q_1_xs - mu_F)/sigma_F),
            #                             jax.scipy.special.log_ndtr((q_1_xs - mu_tilde_up) /sigma_tilde_xs_low) - jax.scipy.special.log_ndtr((q_1_xs - mu_F)/sigma_F)))

            log_g_3_xs_up = q_3_xs
            log_g_3_xs_up -= mu_tilde_low
            log_g_3_xs_up /= sigma_tilde_xs_up
            log_g_3_xs_up = jax.scipy.special.log_ndtr(log_g_3_xs_up)
            temp_buffer1 = q_3_xs
            temp_buffer1 -= mu_F
            temp_buffer1 /= sigma_F
            temp_buffer1 = jax.scipy.special.log_ndtr(temp_buffer1)
            log_g_3_xs_up -= temp_buffer1
            temp_buffer2 = q_3_xs
            temp_buffer2 -= mu_tilde_low
            temp_buffer2 /= sigma_tilde_xs_low
            temp_buffer2 = jax.scipy.special.log_ndtr(temp_buffer2)
            temp_buffer2 -= temp_buffer1
            log_g_3_xs_up = jnp.maximum(log_g_3_xs_up, temp_buffer2)
            #log_g_3_xs_up  = log_g_3_xs_up.at[:].set(jnp.maximum(jax.scipy.special.log_ndtr((q_3_xs - mu_tilde_low)/sigma_tilde_xs_up)  - jax.scipy.special.log_ndtr((q_3_xs - mu_F)/sigma_F),
            #                             jax.scipy.special.log_ndtr((q_3_xs - mu_tilde_low)/sigma_tilde_xs_low) - jax.scipy.special.log_ndtr((q_3_xs - mu_F)/sigma_F)))
            log_g_3_xs_low = q_3_xs
            log_g_3_xs_low -= mu_tilde_up
            log_g_3_xs_low /= sigma_tilde_xs_up
            log_g_3_xs_low = jax.scipy.special.log_ndtr(log_g_3_xs_low)
            log_g_3_xs_low -= temp_buffer1
            temp_buffer2 = q_3_xs
            temp_buffer2 -= mu_tilde_up
            temp_buffer2 /= sigma_tilde_xs_low
            temp_buffer2 = jax.scipy.special.log_ndtr(temp_buffer2)
            temp_buffer2 -= temp_buffer1
            log_g_3_xs_low = jnp.minimum(log_g_3_xs_low, temp_buffer2)
            #log_g_3_xs_low = log_g_3_xs_low.at[:].set(jnp.minimum(jax.scipy.special.log_ndtr((q_3_xs - mu_tilde_up) /sigma_tilde_xs_up)  - jax.scipy.special.log_ndtr((q_3_xs - mu_F)/sigma_F),
            #                             jax.scipy.special.log_ndtr((q_3_xs - mu_tilde_up) /sigma_tilde_xs_low) - jax.scipy.special.log_ndtr((q_3_xs - mu_F)/sigma_F)))

            q_1_xs_up  = q_1_xs_up. at[:].set(jnp.where(log_g_1_xs_low >= log_quarter,         q_1_xs, q_1_xs_up))
            q_1_xs_low = q_1_xs_low.at[:].set(jnp.where(log_g_1_xs_up  <= log_quarter,         q_1_xs, q_1_xs_low))
            q_3_xs_up  = q_3_xs_up. at[:].set(jnp.where(log_g_3_xs_low >= log_three_quarters,  q_3_xs, q_3_xs_up))
            q_3_xs_low = q_3_xs_low.at[:].set(jnp.where(log_g_3_xs_up  <= log_three_quarters,  q_3_xs, q_3_xs_low))
            return q_1_xs_low, q_1_xs_up, q_3_xs_low, q_3_xs_up, mem1, mem2, mem3, mem4, mem5, mem6, mem7, mem8 # last 6 elements are just memory placeholders

        q_1_xs_low, q_1_xs_up, q_3_xs_low, q_3_xs_up, _, _, _, _, _, _, _, _ = jax.lax.fori_loop(0, m, body_func, (q_1_xs_low, q_1_xs_up, q_3_xs_low, q_3_xs_up, jnp.zeros_like(q_1_xs_low), jnp.zeros_like(q_1_xs_low), jnp.zeros_like(q_1_xs_low), jnp.zeros_like(q_1_xs_low), jnp.zeros_like(q_1_xs_low), jnp.zeros_like(q_1_xs_low), jnp.zeros_like(q_1_xs_low), jnp.zeros_like(q_1_xs_low)))

        mu_xs_up     = (q_3_xs_up  + q_1_xs_up)  / 2
        mu_xs_low    = (q_3_xs_low + q_1_xs_low) / 2
        sigma_xs_up  = (q_3_xs_up  - q_1_xs_low) / (2 * jax.scipy.special.ndtri(0.75))
        sigma_xs_low = (q_3_xs_low - q_1_xs_up) / (2 * jax.scipy.special.ndtri(0.75))

        # handle case where simultaneously mu_tilde_low == mu_F and sigma_tilde_xs_up == sigma_F, there Phi((f-mu)/sigma) / Phi((f-mu_x)/sigma_x) = 1 forall f
        mu_xs_up     = jnp.where((mu_tilde_low == mu_F) & (sigma_tilde_xs_up == sigma_F), -jnp.inf, mu_xs_up)
        mu_xs_low    = jnp.where((mu_tilde_low == mu_F) & (sigma_tilde_xs_up == sigma_F), -jnp.inf, mu_xs_low)
        sigma_xs_up  = jnp.where((mu_tilde_low == mu_F) & (sigma_tilde_xs_up == sigma_F),        0, sigma_xs_up)
        sigma_xs_low = jnp.where((mu_tilde_low == mu_F) & (sigma_tilde_xs_up == sigma_F),        0, sigma_xs_low)

        return mu_xs_up, mu_xs_low, sigma_xs_up, sigma_xs_low

    def body_func(bounds_on_poos_and_m:Tuple[jax.Array, jax.Array, float]):
        m = bounds_on_poos_and_m[2] * 2
        mu_up, mu_low, sigma_up, sigma_low = nie_I_S(m, gaussian_means, gaussian_stds)
        if not simplified:
            mu_xs_up, mu_xs_low, sigma_xs_up, sigma_xs_low = nie_II_S(m, mu_up, mu_low, sigma_up, sigma_low, gaussian_means, gaussian_stds)
            invalid_sigmas = (sigma_low < 0) | jnp.any(sigma_xs_low < 0)
        else:
            invalid_sigmas = sigma_low < 0
        p_xs_I_up = jax.scipy.special.ndtr(      jnp.maximum((gaussian_means - mu_low)    / jnp.sqrt(gaussian_stds**2 + sigma_low**2),    (gaussian_means - mu_low) / jnp.sqrt(gaussian_stds**2 + sigma_up**2)))
        p_xs_I_low = jax.scipy.special.ndtr(     jnp.minimum((gaussian_means - mu_up)     / jnp.sqrt(gaussian_stds**2 + sigma_low**2),    (gaussian_means - mu_up)  / jnp.sqrt(gaussian_stds**2 + sigma_up**2)))
        if not simplified:
            p_xs_II_up = jax.scipy.special.ndtr( jnp.maximum((gaussian_means - mu_xs_low) / jnp.sqrt(gaussian_stds**2 + sigma_xs_low**2), (gaussian_means - mu_xs_low) / jnp.sqrt(gaussian_stds**2 + sigma_xs_up**2)))
            p_xs_II_low = jax.scipy.special.ndtr(jnp.minimum((gaussian_means - mu_xs_up)  / jnp.sqrt(gaussian_stds**2 + sigma_xs_low**2), (gaussian_means - mu_xs_up)  / jnp.sqrt(gaussian_stds**2 + sigma_xs_up**2)))
            p_xs_up = jnp.maximum(p_xs_I_up, p_xs_II_up)
            p_xs_low = jnp.maximum(p_xs_I_low, p_xs_II_low)
        else:
            p_xs_up = p_xs_I_up
            p_xs_low = p_xs_I_low
        p_xs_up = jnp.where(invalid_sigmas, jnp.inf, p_xs_up) # if invalid sigmas exist, ensure the loop does not terminate by setting the upper bound of the probabilities to infinity
        return (p_xs_up, p_xs_low, m)
        
    def cond_func(bounds_on_poos_and_m:Tuple[jax.Array, jax.Array, float]):
        p_xs_up, p_xs_low, m = bounds_on_poos_and_m
        return (m == 1) | (jnp.max(p_xs_up - p_xs_low) >= eta)

    m = 1
    p_xs_up, p_xs_low, m = jax.lax.while_loop(cond_func, body_func, (jnp.zeros((domain_size,)), jnp.zeros((domain_size,)), m))
    p_xs = (p_xs_up + p_xs_low)/2
    return p_xs / jnp.sum(p_xs)

@jax.jit
def cme_poo_with_kappa(gaussian_means:jax.Array, gaussian_stds:jax.Array, alpha:float, scaling:float=1.0) -> Tuple[jax.Array, float]:
    """Computes the concentrated maximum estimator (CME) for probability of optimality (poo) but also returns the threshold kappa
    Args:
        gaussian_means (jax.Array): (m,)
        gaussian_stds (jax.Array): (m,)
        alpha (float): sets precision of binary search to 1 / (alpha |X|)
        scaling (float): the scaling of the standard deviations of each F_x
    Returns:
        Tuple[jax.Array, float]: ((m, ) the probabilities of optimality (summing to 1), threshold kappa)
    """
    kappa_low, kappa_up = find_normalising_threshold(gaussian_means, gaussian_stds*scaling, alpha, vapor=False)
    r_up =  p_gaussian_geq_kappa(gaussian_means, gaussian_stds*scaling, kappa_low)
    r_low = p_gaussian_geq_kappa(gaussian_means, gaussian_stds*scaling, kappa_up)
    r = (r_up + r_low) / 2
    r /= jnp.sum(r) # ensures integrability, evens out rounding errors
    return (r, (kappa_low + kappa_up) / 2)

def cme_poo(gaussian_means:jax.Array, gaussian_stds:jax.Array, alpha:float, scaling:float=1.0) -> jax.Array:
    """Computes the concentrated maximum estimator (CME) for probability of optimality (poo)
    Args:
        gaussian_means (jax.Array): (m,)
        gaussian_stds (jax.Array): (m,)
        alpha (float): sets precision of binary search to 1 / (alpha |X|)
        scaling (float): the scaling of the standard deviations of each F_x

    Returns:
        jax.Array: (m, ) the probabilities of optimality (summing to 1)
    """
    return cme_poo_with_kappa(gaussian_means, gaussian_stds, alpha, scaling)[0]

@partial(jax.jit, static_argnames=['exploration_factor'])
def ocme_poo_with_kappa(gaussian_means:jax.Array, gaussian_stds:jax.Array, alpha:float, exploration_factor:float=1.0) -> Tuple[jax.Array, float]:
    """Computes the optimistic concentrated maximum estimator (ocme) for probability of optimality (poo) but also returns the threshold kappa
    Args:
        gaussian_means (jax.Array): (m,)
        gaussian_stds (jax.Array): (m,)
        alpha (float): sets precision of binary search to 1 / (alpha |X|)
        exploration_factor (float): how much to bias the distribution towards more explorative behaviour when the optimal point is mostly assumed to be known
    Returns:
        Tuple[jax.Array, float]: ((m, ) the probabilities of optimality (summing to 1), threshold kappa)
    """
    kappa_low, kappa_up = find_normalising_threshold(gaussian_means, gaussian_stds, alpha, vapor=False, exploration_factor=exploration_factor)
    probs_low = p_gaussian_geq_kappa(gaussian_means, gaussian_stds, kappa_up)
    probs_up = p_gaussian_geq_kappa(gaussian_means, gaussian_stds, kappa_low)
    probs = (probs_low + probs_up) / 2
    r = probs / (1 - exploration_factor*probs)
    r /= jnp.sum(r) # ensures integrability, evens out rounding errors
    return (r, (kappa_low + kappa_up) / 2)

def ocme_poo(gaussian_means:jax.Array, gaussian_stds:jax.Array, alpha:float, exploration_factor:float=1.0) -> jax.Array:
    """Computes the optimistic concentrated maximum estimator (ocme) for probability of optimality (poo)
    Args:
        gaussian_means (jax.Array): (m,)
        gaussian_stds (jax.Array): (m,)
        alpha (float): sets precision of binary search to 1 / (alpha |X|)
        exploration_factor (float): how much to bias the distribution towards more explorative behaviour when the optimal point is mostly assumed to be known

    Returns:
        jax.Array: (m, ) the probabilities of optimality (summing to 1)
    """
    return ocme_poo_with_kappa(gaussian_means, gaussian_stds, alpha, exploration_factor)[0]

@jax.jit
def vapor_poo_with_kappa(gaussian_means:jax.Array, gaussian_stds:jax.Array, alpha:float) -> Tuple[jax.Array, float]:
    """Computes the vapor estimator (VAPOR-E) for probability of optimality (poo), but also returns the threshold kappa
    Args:
        gaussian_means (jax.Array): (m,)
        gaussian_stds (jax.Array): (m,)
        alpha (float): sets precision of binary search to 1 / (alpha |X|)

    Returns:
        Tuple[jax.Array, float]: ((m, ) the probabilities of optimality (summing to 1), threshold kappa)
    """
    kappa_low, kappa_up = find_normalising_threshold(gaussian_means, gaussian_stds, alpha, vapor=True)
    r_up =  p_vapordist_geq_kappa(gaussian_means, gaussian_stds, kappa_low)
    r_low = p_vapordist_geq_kappa(gaussian_means, gaussian_stds, kappa_up)
    r = (r_low + r_up) / 2
    r /= jnp.sum(r) # ensures integrability, evens out rounding errors
    return (r, (kappa_low + kappa_up) / 2)

def vapor_poo(gaussian_means:jax.Array, gaussian_stds:jax.Array, alpha:float) -> jax.Array:
    """Computes the vapor estimator (VAPOR-E) for probability of optimality (poo)
    Args:
        gaussian_means (jax.Array): (m,)
        gaussian_stds (jax.Array): (m,)
        alpha (float): sets precision of binary search to 1 / (alpha |X|)

    Returns:
        jax.Array: (m, ) the probabilities of optimality (summing to 1)
    """
    return vapor_poo_with_kappa(gaussian_means, gaussian_stds, alpha)[0]

#import cvxpy as cp

#def vapor_poo_old(gaussian_means:jax.Array, gaussian_stds:jax.Array, alpha:float) -> jax.Array:
#    mu, sigma = gaussian_means, gaussian_stds
#
#    r = cp.Variable(mu.size)
#    objective = cp.Maximize(cp.sum( cp.multiply(r, gaussian_means) + cp.multiply(cp.sqrt(cp.entr(r**2)), sigma) ))
#    constraints = [r >= 0, cp.sum(r) == 1]
#    prob = cp.Problem(objective, constraints)
#    result = prob.solve()#solver=cp.SCS)#, eps=1/(alpha*mu.size))
#    return r.value()

@partial(jax.jit, static_argnames=['normalised']) # entirely jitted, but still usually slower than the numerical integration
def _est_poo_with_kappa_tilde_sampling(gaussian_means:jax.Array, gaussian_stds:jax.Array, random_key:jax.Array, alpha:float, normalised:bool = False) -> Tuple[jax.Array, float]:
    """Computes the EST estimator (EST) for probability of optimality (poo) and also returns kappa_tilde, i.e. an estimate of E[max_x F_x | D]
       based on i.i.d. samples of F^* | D
    Args:
        gaussian_means (jax.Array): (m,)
        gaussian_stds (jax.Array): (m,)
        random_key (jax.Array): random key for estimating E[max_x F_x | D]
        alpha (float): reciprocal of desired relative accuracy (abs accuracy <= 1/(|X| alpha) of estimated probabilities
        normalised (float): whether to normalise the estimates of probability of optimality

    Returns:
        Tuple[jax.Array, float]: ((m, ) the probabilities of optimality, kappa_tilde)
    """    
    batch_size = 100000
    @jax.jit
    def body_func(state: Tuple[float, float, float, jax.Array, float, float, float, float]):
        summed = state[0]
        squared_summed = state[1]
        n_terms = state[2]
        random_key = state[3]

        random_key, sample_key = jax.random.split(random_key, num=2)
        f_maxs = gaussians.sample_n_max_independent(gaussian_means, gaussian_stds, batch_size, sample_key, 1) # always sample 1000 at a time

        summed += jnp.sum(f_maxs)
        squared_summed += jnp.sum(f_maxs**2)
        n_terms += batch_size

        mean = summed / n_terms # unbiased estimator
        var = (squared_summed - mean * n_terms) / (n_terms-1) # unbiased estimator

        #Var[(f_1 + f_2 + … + f_n)/n | D) = sum_i=1^n Var[f_i / n | D] = sum_{i=1}^n Var[f_i | D] / n^2 = Var[f_i | D] / n for i.i.d. f_i ~ f^* | D
        mean_var = var / n_terms
        mean_std = mean_var**.5 # upper bound on E[std] by Jensen's inequality, we shall assume that it is also an upper bound on the groundtruth for the comment in the next line (which also uses a possibly loose chebyshev inequality)

        kappa_low, kappa_up = mean - 10*mean_std, mean + 10*mean_std # with high likelihood (99%) kappa_low and kappa_up should be lower and upper bounds on the expectation of mean (check chebyshev's inequality)
        
        # upper bound
        p_geq_kappa_tilde_up = p_gaussian_geq_kappa(gaussian_means, gaussian_stds, kappa_low) # monotonously decreasing in last argument (kappa)
        p_f_star_leq_kappa_tilde_up = jnp.prod(jax.scipy.special.ndtr((kappa_up-gaussian_means)/gaussian_stds)) # monotonously increasing in kappa
        r_up = p_f_star_leq_kappa_tilde_up * p_geq_kappa_tilde_up/(1-p_geq_kappa_tilde_up) 

        # lower bound
        p_geq_kappa_tilde_low = p_gaussian_geq_kappa(gaussian_means, gaussian_stds, kappa_up) # monotonously decreasing in last argument (kappa)
        p_f_star_leq_kappa_tilde_low = jnp.prod(jax.scipy.special.ndtr((kappa_low-gaussian_means)/gaussian_stds)) # monotonously increasing in kappa
        r_low = p_f_star_leq_kappa_tilde_low * p_geq_kappa_tilde_low/(1-p_geq_kappa_tilde_low) 

        jax.debug.print("EST used {t} samples (if it keeps increasing try using numerical integration instead)", t=n_terms) 

        return summed, squared_summed, n_terms, random_key, kappa_low, kappa_up, r_low, r_up
    @jax.jit
    def cond_func(state: Tuple[float, float, float, jax.Array, float, float, float, float]):
        r_low, r_up = state[6], state[7]
        max_error = jnp.max(jnp.abs(r_up - r_low))/2
        return max_error > 1/(gaussian_means.size * alpha)

    _, _, _, _, kappa_low, kappa_up, r_low, r_up = jax.lax.while_loop(cond_func, body_func, (0, 0, 0, random_key, 0, 0, jnp.zeros((gaussian_means.size,)), jnp.ones((gaussian_means.size,)) * jnp.inf))

    r = (r_up + r_low)/2
    kappa = (kappa_up + kappa_low)/2

    if normalised:
        norm = jnp.sum(r)
        #print(f"EST probability of optimality sums up to {norm}")

    return (r/norm, kappa) if normalised else (r, kappa)

# unjitted, yet faster than the sampling approach which is jitted
def _est_poo_with_kappa_tilde_numerical(gaussian_means:jax.Array, gaussian_stds:jax.Array, alpha:float, normalised:bool = False, start_nums:int=10000) -> Tuple[jax.Array, float]:
    """Computes the EST estimator (EST) for probability of optimality (poo) and also returns kappa_tilde, i.e. an estimate of E[max_x F_x | D]
       based on numerical integration
    Args:
        gaussian_means (jax.Array): (m,)
        gaussian_stds (jax.Array): (m,)
        alpha (float): reciprocal of desired relative accuracy (abs accuracy <= 1/(|X| alpha) of estimated probabilities
        normalised (float): whether to normalise the estimates of probability of optimality
        start_nums: number of integration points to start with
    Returns:
        Tuple[jax.Array, float]: ((m, ) the probabilities of optimality, kappa_tilde)
    """    
    nums = start_nums
    while True:
        kappa_low, kappa_up = numerical_F_star_expectation(gaussian_means, gaussian_stds, nums)

        # upper bound
        p_geq_kappa_tilde_up = p_gaussian_geq_kappa(gaussian_means, gaussian_stds, kappa_low) # monotonously decreasing in last argument (kappa)
        p_f_star_leq_kappa_tilde_up = jnp.prod(jax.scipy.special.ndtr((kappa_up-gaussian_means)/gaussian_stds)) # monotonously increasing in kappa
        r_up = p_f_star_leq_kappa_tilde_up * p_geq_kappa_tilde_up/(1-p_geq_kappa_tilde_up) 
        # lower bound
        p_geq_kappa_tilde_low = p_gaussian_geq_kappa(gaussian_means, gaussian_stds, kappa_up) # monotonously decreasing in last argument (kappa)
        p_f_star_leq_kappa_tilde_low = jnp.prod(jax.scipy.special.ndtr((kappa_low-gaussian_means)/gaussian_stds)) # monotonously increasing in kappa
        r_low = p_f_star_leq_kappa_tilde_low * p_geq_kappa_tilde_low/(1-p_geq_kappa_tilde_low) 

        max_error = jnp.max(jnp.abs(r_up - r_low))/2
        if max_error <= 1/(gaussian_means.size * alpha):
            break
        nums *= 2
        print(f"EST integration error too large, repeat with {nums} integration points")

    r = (r_up + r_low)/2
    kappa = (kappa_up + kappa_low)/2

    if normalised:
        norm = jnp.sum(r)
        print(f"EST probability of optimality sums up to {norm}")
    return (r/norm, kappa) if normalised else (r, kappa)

def est_poo_with_kappa_tilde(gaussian_means:jax.Array, gaussian_stds:jax.Array, random_key:jax.Array, alpha:float, normalised:bool = False, use_integration = True, start_nums:int=10_000) -> Tuple[jax.Array, float]:
    """Computes the EST estimator (EST) for probability of optimality (poo) and also returns kappa_tilde, i.e. an estimate of E[max_x F_x | D]
    Args:
        gaussian_means (jax.Array): (m,)
        gaussian_stds (jax.Array): (m,)
        random_key (jax.Array): random key for estimating E[max_x F_x | D], only used if use_integration = False
        alpha (float): reciprocal of desired relative accuracy (abs accuracy <= 1/(|X| alpha) of estimated probabilities
        normalised (float): whether to normalise the estimates of probability of optimality
        use_integration (jax.Array): whether to use the sampling based implementation (fully jitted) or instead the numerical integration (python while loop with jitted subroutine)
            numerical integration is usually significantly faster and never gets stuck unlike sampling-based estimation
        start_nums: number of integration points to start with
    Returns:
        Tuple[jax.Array, float]: ((m, ) the probabilities of optimality, kappa_tilde)
    """    
    if use_integration:
        return _est_poo_with_kappa_tilde_numerical(gaussian_means, gaussian_stds, alpha, normalised, start_nums)
    else: # not integration, i.e. sampling based
        return _est_poo_with_kappa_tilde_sampling(gaussian_means, gaussian_stds, random_key, alpha, normalised)
        

def est_poo(gaussian_means:jax.Array, gaussian_stds:jax.Array, random_key:jax.Array, alpha:float, normalised:bool = False, use_integration = True, start_nums:int=10_000) -> jax.Array:
    """Computes the EST estimator (EST) for probability of optimality (poo)
    Args:
        gaussian_means (jax.Array): (m,)
        gaussian_stds (jax.Array): (m,)
        random_key (jax.Array): random key for estimating E[max_x F_x | D], only used if use_integration = False
        alpha (float): reciprocal of desired relative accuracy (abs accuracy <= 1/(|X| alpha) due to numerical integration
        compared to the analytic expression of EST
        normalised (float) : whether to normalise the estimates of probability of optimality
        use_integration (jax.Array): whether to use the sampling based implementation (fully jitted) or instead the numerical integration (python while loop with jitted subroutine)
            numerical integration is usually significantly faster and never gets stuck unlike sampling-based estimation
        start_nums: number of integration points to start with
    Returns:
        jax.Array: (m, ) the probabilities of optimality
    """    
    return est_poo_with_kappa_tilde(gaussian_means, gaussian_stds, random_key, alpha, normalised, use_integration, start_nums=start_nums)[0]

@partial(jax.jit, static_argnames=['use_cme'])
def vest_with_kappa(gaussian_means:jax.Array, gaussian_stds:jax.Array, alpha:float, use_cme:bool=False) -> Tuple[jax.Array, float]:
    """Computes the VEST acquisition function (VEST) and also returns kappa, i.e. an upper bound on E[max_x F_x | D]
    Args:
        gaussian_means (jax.Array): (m,)
        gaussian_stds (jax.Array): (m,)
        alpha (float): reciprocal of desired relative accuracy (abs accuracy <= 1/(|X| alpha) in binary search

    Returns:
        Tuple[jax.Array, float]: ((m, ) the probabilities of optimality, kappa)
    """    
    if use_cme:
        probs = cme_poo(gaussian_means, gaussian_stds, alpha) # more accurate, but does not guarantee an upper bound
    else:
        probs = vapor_poo(gaussian_means, gaussian_stds, alpha)
    f_star_expect_upper_bound = jnp.sum(gaussian_means * probs + gaussian_stds * jnp.sqrt(-2 * probs * jax.scipy.special.xlogy(probs, probs)))
    probs = p_gaussian_geq_kappa(gaussian_means, gaussian_stds, f_star_expect_upper_bound)
    probs /= jnp.sum(probs) # ensures integrability, evens out rounding errors
    return (probs, f_star_expect_upper_bound)

def vest(gaussian_means:jax.Array, gaussian_stds:jax.Array, alpha:float, use_cme:bool=False) -> Tuple[jax.Array, float]:
    """Computes the VEST acquisition function (VEST)
    Args:
        gaussian_means (jax.Array): (m,)
        gaussian_stds (jax.Array): (m,)
        alpha (float): reciprocal of desired relative accuracy (abs accuracy <= 1/(|X| alpha) in binary search

    Returns:
        Tuple[jax.Array, float]: ((m, ) the probabilities of optimality, kappa)
    """    
    return vest_with_kappa(gaussian_means, gaussian_stds, alpha, use_cme)[0]

# -------------------------------------------------- Classical Acquisition Functions --------------------------------------------------

@jax.jit
def probability_of_improvement(gaussian_means:jax.Array, gaussian_stds:jax.Array, reference:float, marginal_threshold:float = 0) -> jax.Array:
    """Computes the probability of improvement acquisition function for a reference value and a marginal threshold

    Args:
        gaussian_means (jax.Array): (m,)
        gaussian_stds (jax.Array): (m,)
        reference (jax.Array): scalar
        marginal_threshold (float, optional): (1,) added to the reference value to form kappa. Defaults to 0.

    Returns:
        jax.Array: (m,) P[F_x >= reference + marginal_threshold] for x=1,...,m
    """
    r = p_gaussian_geq_kappa(gaussian_means, gaussian_stds, 
                              reference+marginal_threshold)
    return r

@jax.jit
def expected_improvement(gaussian_means:jax.Array, gaussian_stds:jax.Array, max_observation:float) -> jax.Array:
    """Computes the expected improvement acquisition function

    Args:
        gaussian_means (jax.Array): (m,)
        gaussian_stds (jax.Array): (m,)
        max_observation (float): (1,)

    Returns:
        jax.Array: (m,) E[max(0, F_x - max(observations))] for x=1,...,m
    """
    deltas = (gaussian_means - max_observation)
    r = deltas * p_gaussian_geq_kappa(gaussian_means, gaussian_stds, max_observation)\
    + gaussian_stds * jax.scipy.stats.norm.pdf(deltas / gaussian_stds, 0, 1)
    return r

@jax.jit
def ucb(gaussian_means:jax.Array, gaussian_stds:jax.Array, t:int, delta:float) -> jax.Array:
    """Computes the UCB acquisition function at time step t

    Args:
        gaussian_means (jax.Array): (m,)
        gaussian_stds (jax.Array): (m,)
        t (int): time step (starts at 1)
        delta (float): the accepted probability under which the GP-UCB regret bounds do not need to hold

    Returns:
        jax.Array: (m,) UCB(x) for x=1,...,m
    """
    beta = 2 * jnp.log(gaussian_means.size * t**2 * jnp.pi**2 / (6*delta))
    sqrt_beta = jnp.sqrt(beta)
    return gaussian_means + sqrt_beta * gaussian_stds

# -------------------------------------------------- Information Theoretic Acquisition Functions --------------------------------------------------

@partial(jax.jit, static_argnames=['gumbel', 'samples', 'account_for_sampling_noise'])
def max_value_entropy_search(gaussian_means:jax.Array, gaussian_stds:jax.Array, observation_noise_stds: jax.Array, random_key:jax.Array, gumbel:float = True, delta:float = 0.01, samples:int = 100, account_for_sampling_noise:bool = False) -> jax.Array:
    """Returns an estimate of the acquisition function I(F^* ; Y_x | D) using MES with Gumbel Sampling.

    gaussian_means (jax.Array): (m,)
    gaussian_stds (jax.Array): (m,)
    observation_noise_stds (jax.Array): (m,)
    random_key (jax.Array): seeds sampling
    delta (float): accuracy of binary search for interquartiles of Gumbel mean field approxiamtion to F^* | D, defaults to 0.01 (0.25+-0.01 and 0.75+-0.01)
    samples (int): the number of Gumbel samples used (approximate samples to F^*|D)
    account_for_sampling_noise (bool): whether to account for sampling noise (deviates from MES paper)

    Returns:
        jax.Array: (m,) the acquisition function I(F^* ; Y_x | D)

    References
    ----------
    [1] Wang, Z. & Jegelka, S.. (2017). Max-value Entropy Search for Efficient
        Bayesian Optimization. Proceedings of the 34th International Conference
        on Machine Learning, in PMLR 70:3627-3635
    """
    random_key, one_time_key = jax.random.split(random_key)
    # mean-field approximation to P[Y^* <= y], see [1]
    if gumbel:
        # borders of binary search for y1 and y2 that ensure the 0.25 - delta and 0.75 + delta quartiles to be inside
        left_border = jnp.min(gaussian_means - 20 * gaussian_stds) # instead of -10 could use jax.scipy.special.ndtri(0.25 - delta) and jnp.max, however, empirically it breaks without the theoretically too loose borders 
        right_border = jnp.max(gaussian_means + 20 * gaussian_stds) # instead of 10 could use jax.scipy.special.ndtri((0.75 + delta)**(1/len(gaussian_means))), but may be numerically unstable
        func = lambda y: jnp.prod(jax.scipy.special.ndtr(y - gaussian_means) / gaussian_stds) - 0.25
        y1_low, y1_up = _m_ary_root_search(func, left_border, right_border, delta, m=5)
        y1 = (y1_up + y1_low) / 2
        func = lambda y: jnp.prod(jax.scipy.special.ndtr(y - gaussian_means) / gaussian_stds) - 0.75
        y2_low, y2_up = _m_ary_root_search(func, left_border, right_border, delta, m=5)
        y2 = (y2_up + y2_low) / 2
        # solve for Gumbel parameters a and b
        b = (y2-y1) / (jnp.log(jnp.log(4) / jnp.log(4/3))) # (y2-y1) / (jnp.log(-jnp.log(0.25)) - jnp.log(-jnp.log(0.75)))
        a = y1 + b * jnp.log(jnp.log(4)) # y1 + b * jnp.log(-jnp.log(0.25))
        # sample f^*'s
        f_stars = a - b * jnp.log(-jnp.log(jax.random.uniform(one_time_key, (samples, 1))))
    else: # Gumbel = False
        f_stars = jnp.expand_dims(gaussians.sample_n_max_independent(gaussian_means, gaussian_stds, samples, one_time_key, 5), axis=1)
    
    if account_for_sampling_noise: # estimates entropy with a single sample => unbiased
        total_variances = observation_noise_stds**2 + gaussian_stds**2 # in (|X|)
        total_stds = jnp.sqrt(total_variances) # in (|X|)
        y_s = gaussians.sample_n_independent(gaussian_means, total_stds, samples, random_key) # in (samples, |X|)
        importance_weights = jax.scipy.special.ndtr((f_stars * (total_variances) - y_s * gaussian_stds**2 - gaussian_means * observation_noise_stds**2) / (gaussian_stds * observation_noise_stds * total_stds))\
                             / jax.scipy.special.ndtr((f_stars - gaussian_means)/ gaussian_stds) # in (samples, |X|)
        surprisals = (y_s - gaussian_means)**2 / (2*total_variances) + 0.5 * jnp.log(2 * jnp.pi * total_variances) - jnp.log(importance_weights) # in (samples, |X|)

        mutual_information = jnp.mean(importance_weights * surprisals, axis=0) # in (|X|)
    else: 
        gamma = (f_stars - gaussian_means)/(gaussian_stds) # (samples, |X|)
        #jax.debug.print("negative_fraction {f}", f=jnp.sum(gamma < 0) / gamma.size) 
        mutual_information = jnp.mean(gamma * gaussians.standard_normal_pdf_divided_by_cdf(gamma) / 2 - jax.scipy.special.log_ndtr(gamma), axis=0)

    return mutual_information



@partial(jax.jit, static_argnames=['alpha', 'samples', 'using_quantiles', 'unroll', 'using_CME'])
def entropy_search(points:jax.Array, gaussian_means:jax.Array, gaussian_cov:jax.Array, observation_noise_stds:jax.Array, random_key:jax.Array, alpha:float = 10.0, samples:int = 100, using_quantiles:bool=True, unroll:int=1, using_CME:bool=True) -> jax.Array:
    """Returns an estimate of the acquisition function I(X^* ; Y_x | D) at specified points using a variant of Entropy Search (ES) employing CME or E-TSE for PoM entropy estimation

    points (jax.Array): (n,) the indices (0...m-1) of the points for which the acquisition function is computed
    gaussian_means (jax.Array): (m,)
    gaussian_cov (jax.Array): (m, m)
    observation_noise_stds (jax.Array): (m,)
    random_key (jax.Array): seeds sampling
    alpha (float): accuracy of binary search in CME subroutines and E-TSE Monte Carlo sampling
    samples (int): the number of samples or quantiles of Y_x | D used to estimate conditional entropy
    using_quantiles (bool): whether to use quantiles of Y_x | D instead of independent samples from it to estimate conditional entropy
    unroll (int): how far to unroll the outer loop over x in X, each element of inner loop demands |X|*samples memory (and compute) for CME and |X|^2*samples memory (and compute) for E-TSE
    using_CME (bool): whether to use CME for PoM entropy estimation rather than E-TSE

    Returns:
        jax.Array: (m,) the acquisition function I(X^* ; Y_x | D) evaluated at the provided points
    """
    # as help to understand the code, here are the function declarations for cme and etse
    # cme_poo(gaussian_means:jax.Array, gaussian_stds:jax.Array, alpha:float, scaling:float=1.0)
    # etse_poo(gaussian_means:jax.Array, gaussian_cov:jax.Array, number_of_samples:int, random_key:jax.Array, unroll:int=1)

    if using_CME:
        p_x_star = cme_poo(gaussian_means, gaussians.standard_deviations(gaussian_cov), alpha=alpha)
        condition_on_several_observations = jax.vmap(gaussians.condition_on_observations_diagonal, in_axes=(None, None, None, 0, None), out_axes=0)
        poms_for_several_conditionings = jax.vmap(cme_poo, in_axes=(0, 0, None), out_axes=0)
    else:
        random_key, one_time_key = jax.random.split(random_key)
        p_x_star = etse_poo(gaussian_means, gaussian_cov, int(alpha**2 * gaussian_means.size**2), one_time_key, 1)
        condition_on_several_observations = jax.vmap(gaussians.condition_on_observations, in_axes=(None, None, None, 0, None), out_axes=0)
        poms_for_several_conditionings = jax.vmap(etse_poo, in_axes=(0, 0, None, 0, None), out_axes=0)
    entropy = jnp.sum(jax.scipy.special.entr(p_x_star))

    
    @jax.jit
    def body_func(i:int, state:Tuple[jax.Array, jax.Array]):
        x = points[i]
        cond_entropies = state[0]
        key, one_time_key = jax.random.split(state[1])
        if using_quantiles:
            y_samples = jnp.expand_dims(gaussian_means[x] + (gaussian_cov[x,x] + observation_noise_stds[x]**2)**.5 * jax.scipy.special.ndtri(jnp.linspace(start=1/(2*samples), stop=1-1/(2*samples), num=samples, endpoint=True)), axis=1)
        else:
            y_samples = gaussians.sample_n_independent(mean_vector=jnp.expand_dims(gaussian_means[x], 0), standard_deviations=jnp.expand_dims(gaussian_cov[x, x] + observation_noise_stds[x]**2, 0)**.5, n=samples, random_key=one_time_key)
        means, cov = condition_on_several_observations(gaussian_means, gaussian_cov, x*jnp.ones((1,), dtype=int), y_samples, jnp.expand_dims(observation_noise_stds[x], axis=0))
        # means and diag_cov are of shape (samples, |X|)
        if using_CME:
            p_x_stars = poms_for_several_conditionings(means, cov**.5, alpha)
        else:
            p_x_stars = poms_for_several_conditionings(means, cov, int(alpha**2 * means.size**2), jax.random.split(one_time_key, samples), 1)
        real_cond_entropies = jnp.sum(jax.scipy.special.entr(p_x_stars), axis=1)
        cond_entropy = jnp.mean(real_cond_entropies)
        return (cond_entropies.at[i].set(cond_entropy), key)
    
    conditional_entropies = jnp.zeros_like(points, dtype=float) # stores H[X^* | Y_x]
    conditional_entropies, _ = jax.lax.fori_loop(0, conditional_entropies.size, body_func, (conditional_entropies, random_key), unroll=min(conditional_entropies.size, unroll))
    #print(entropy, conditional_entropies)
    mutual_information = entropy - conditional_entropies
    return mutual_information

@partial(jax.jit, static_argnames=['alpha', 'samples', 'using_quantiles', 'unroll', 'using_CME'])
def entropy_search_global(gaussian_means:jax.Array, gaussian_cov:jax.Array, observation_noise_stds:jax.Array, random_key:jax.Array, alpha:float = 10.0, samples:int = 100, using_quantiles:bool=True, unroll:int=1, using_CME:bool=True) -> jax.Array:
    """Returns an estimate of the acquisition function I(X^* ; Y_x | D) over the whole domain using a variant of Entropy Search (ES) employing CME or E-TSE for PoM entropy estimation

    gaussian_means (jax.Array): (m,)
    gaussian_cov (jax.Array): (m, m)
    observation_noise_stds (jax.Array): (m,)
    random_key (jax.Array): seeds sampling
    alpha (float): accuracy of binary search in CME subroutines and E-TSE Monte Carlo sampling
    samples (int): the number of samples or quantiles of Y_x | D used to estimate conditional entropy
    using_quantiles (bool): whether to use quantiles of Y_x | D instead of independent samples from it to estimate conditional entropy
    unroll (int): how far to unroll the outer loop over x in X, each element of inner loop demands |X|*samples memory (and compute) and |X|^2*samples memory (and compute) for E-TSE
    using_CME (bool): whether to use CME for PoM entropy estimation rather than E-TSE

    Returns:
        jax.Array: (m,) the acquisition function I(X^* ; Y_x | D)
    """
    points = jnp.arange(gaussian_means.size)
    return entropy_search(points, gaussian_means, gaussian_cov, observation_noise_stds, random_key, alpha, samples, using_quantiles, unroll, using_CME)

# @partial(jax.jit, static_argnames=['kernel', 'alpha', 'samples', 'using_quantiles', 'unroll', 'using_CME'])
# def entropy_search_kernelized(points:jax.Array, gaussian_means:jax.Array, gaussian_stds:jax.Array, mean:float, kernel: Callable, kernel_params: dict, positions: jax.Array, observation_indices: jax.Array, observations: jax.Array, observation_noise_stds: jax.Array, random_key:jax.Array, alpha:float = 10.0, samples:int = 100, using_quantiles:bool=True, unroll:int=1, using_CME:bool=True) -> jax.Array:
#     """Returns an estimate of the acquisition function I(X^* ; Y_x | D) at specified points using a variant of Entropy Search (ES) employing CME or E-TSE for PoM entropy estimation

#     points (jax.Array): (n,) the indices (0...|X|-1) of the points for which the acquisition function is computed

#     gaussian_means (jax.Array): (|X|,) current posterior mean (avoids recomputation)
#     gaussian_stds (jax.Array): (|X|, |X|) current posterior stds (avoids recomputation)
#     mean (float): (|X|,) the constant prior mean of the GP
#     kernel (Callable): the prior kernel of the GP
#     kernel_params (dict): the parameters of the prior kernel such as length_scale and amplitude
#     positions (jax.Array): (|X|, d) the positions of the elements of the domain
#     observation_indices (jax.Array): (n,) the indices of the Gaussian components/coordinates that are observed
#     observations (jax.Array): (n,) the actual, observed values
#     observation_noise_stds (jax.Array): (|X|) the standard deviations of the i.i.d. noise associated with observations at each position of the domain

#     random_key (jax.Array): seeds sampling
#     alpha (float): accuracy of binary search in CME subroutines and E-TSE Monte Carlo sampling
#     samples (int): the number of samples or quantiles of Y_x | D used to estimate conditional entropy
#     using_quantiles (bool): whether to use quantiles of Y_x | D instead of independent samples from it to estimate conditional entropy
#     unroll (int): how far to unroll the outer loop over x in X, each element of inner loop demands |X|*samples memory (and compute) for CME and |X|^2*samples memory (and compute) for E-TSE
#     using_CME (bool): whether to use CME for PoM entropy estimation rather than E-TSE

#     Returns:
#         jax.Array: (|X|,) the acquisition function I(X^* ; Y_x | D) evaluated at the provided points
#     """
#     # as help to understand the code, here are the function declarations for cme and etse
#     # cme_poo(gaussian_means:jax.Array, gaussian_stds:jax.Array, alpha:float, scaling:float=1.0)
#     # etse_poo(gaussian_means:jax.Array, gaussian_cov:jax.Array, number_of_samples:int, random_key:jax.Array, unroll:int=1)

#     if using_CME:
#         p_x_star = cme_poo(gaussian_means, gaussian_stds, alpha=alpha)
#         condition_on_several_observations = jax.vmap(gaussians.condition_on_observations_diagonal_w_kernel, in_axes=(None, None, None, None, 0, 0, 0), out_axes=(0, 0))
#         poms_for_several_conditionings = jax.vmap(cme_poo, in_axes=(0, 0, None), out_axes=0)
#     else:
#         assert False, "using_CME=False is not implemented"
#     entropy = jnp.sum(jax.scipy.special.entr(p_x_star))
    
#     # prepare 

#     @jax.jit
#     def body_func(i:int, state:Tuple[jax.Array, jax.Array]):
#         x = points[i]
#         cond_entropies = state[0]
#         key, one_time_key = jax.random.split(state[1])


#         if using_quantiles:
#             y_samples = jnp.expand_dims(gaussian_means[x] + (gaussian_stds[x]**2 + observation_noise_stds[x]**2)**.5 * jax.scipy.special.ndtri(jnp.linspace(start=1/(2*samples), stop=1-1/(2*samples), num=samples, endpoint=True)), axis=1)
#         else:
#             y_samples = gaussians.sample_n_independent(mean_vector=jnp.expand_dims(gaussian_means[x], 0), standard_deviations=jnp.expand_dims(gaussian_stds[x]**2 + observation_noise_stds[x]**2, 0)**.5, n=samples, random_key=one_time_key)

#         means, cov = condition_on_several_observations(mean, kernel, kernel_params, positions, jnp.tile(jnp.expand_dims(jnp.concatenate([x*jnp.ones((1,), dtype=int), observation_indices]), 0), (y_samples.size,1)), jnp.stack([jnp.concatenate([y_sample*jnp.ones((1,)), observations]) for y_sample in y_samples[:,0]]), jnp.tile(jnp.expand_dims(jnp.concatenate([observation_noise_stds[x]*jnp.ones((1,)), observation_noise_stds[observation_indices]]), 0), (y_samples.size,1)))

#         # means and stds are of shape (samples, |X|)
#         if using_CME:
#             p_x_stars = poms_for_several_conditionings(means, cov**.5, alpha)
#         else:
#             assert False, "using_CME=False is not implemented"
#         real_cond_entropies = jnp.sum(jax.scipy.special.entr(p_x_stars), axis=1)
#         cond_entropy = jnp.mean(real_cond_entropies)
#         return (cond_entropies.at[i].set(cond_entropy), key)
    
#     conditional_entropies = jnp.zeros_like(points, dtype=float) # stores H[X^* | Y_x]
#     conditional_entropies, _ = jax.lax.fori_loop(0, conditional_entropies.size, body_func, (conditional_entropies, random_key), unroll=min(conditional_entropies.size, unroll))

#     mutual_information = entropy - conditional_entropies
#     return mutual_information

# @partial(jax.jit, static_argnames=['kernel', 'alpha', 'samples', 'using_quantiles', 'unroll', 'using_CME'])
# def entropy_search_global_kernelized(gaussian_means:jax.Array, gaussian_stds:jax.Array, mean:float, kernel: Callable, kernel_params: dict, positions: jax.Array, observation_indices: jax.Array, observations: jax.Array, observation_noise_stds: jax.Array, random_key:jax.Array, alpha:float = 10.0, samples:int = 100, using_quantiles:bool=True, unroll:int=1, using_CME:bool=True) -> jax.Array:
#     """Returns an estimate of the acquisition function I(X^* ; Y_x | D) over the whole domain using a variant of Entropy Search (ES) employing CME or E-TSE for PoM entropy estimation

#     gaussian_means (jax.Array): (m,) current posterior mean (avoids recomputation)
#     gaussian_stds (jax.Array): (m, m) current posterior stds (avoids recomputation)
#     mean (float): (m,) the constant prior mean of the GP
#     kernel (Callable): the prior kernel of the GP
#     kernel_params (dict): the parameters of the prior kernel such as length_scale and amplitude
#     positions (jax.Array): (|X|, d) the positions of the elements of the domain
#     observation_indices (jax.Array): (n,) the indices of the Gaussian components/coordinates that are observed
#     observations (jax.Array): (n,) the actual, observed values
#     observation_noise_stds (jax.Array): the standard deviations of the i.i.d. noise associated with the observations
    
#     random_key (jax.Array): seeds sampling
#     alpha (float): accuracy of binary search in CME subroutines and E-TSE Monte Carlo sampling
#     samples (int): the number of samples or quantiles of Y_x | D used to estimate conditional entropy
#     using_quantiles (bool): whether to use quantiles of Y_x | D instead of independent samples from it to estimate conditional entropy
#     unroll (int): how far to unroll the outer loop over x in X, each element of inner loop demands |X|*samples memory (and compute) and |X|^2*samples memory (and compute) for E-TSE
#     using_CME (bool): whether to use CME for PoM entropy estimation rather than E-TSE

#     Returns:
#         jax.Array: (m,) the acquisition function I(X^* ; Y_x | D)
#     """
#     points = jnp.arange(positions.shape[0])
#     return entropy_search_kernelized(points, gaussian_means, gaussian_stds, mean, kernel, kernel_params, positions, observation_indices, observations, observation_noise_stds, random_key, alpha, samples, using_quantiles, unroll, using_CME)


poo_estimators_and_BO_listed = "'PI', 'EI', 'UCB', 'TS', 'EST', 'VEST', 'CEST', 'CME', 'CME-2', 'VAPOR', 'OCME', 'CES', 'ES', 'MES', 'IG', 'CIG', 'CIGU'"
def get_acquisition_function(method:str, post_means:jax.Array, post_cov:jax.Array, observation_noise_stds:jax.Array, observation_values:jax.Array, random_key:jax.Array):  
    match method:
        case "PI":
            return probability_of_improvement(post_means, gaussians.standard_deviations(post_cov), jnp.max(observation_values), marginal_threshold=0)
        case "EI":
            return expected_improvement(post_means, gaussians.standard_deviations(post_cov), jnp.max(observation_values))
        case "UCB":
            return ucb(post_means, gaussians.standard_deviations(post_cov), t=len(observation_values)+1, delta=0.1)
        case "TS":
            post_cov_sqrt = gaussians.spsd_matrix_square_root(post_cov)
            return gaussians.sample(post_means, post_cov_sqrt, random_key)
        case "EST":
            return est_poo_with_kappa_tilde(post_means, gaussians.standard_deviations(post_cov), random_key, alpha=100.0, normalised=True, use_integration=True)
        case "VEST": # VAPOR-based EST (uses VAPOR optimisation problem to upper bound E[F^* | D])
            return vest_with_kappa(post_means, gaussians.standard_deviations(post_cov), alpha=100.0)
        case "CEST": # CME-based EST (uses CME optimisation problem to upper bound E[F^* | D])
            return vest_with_kappa(post_means, gaussians.standard_deviations(post_cov), alpha=100.0, use_cme=True)
        case "CME":
            return cme_poo_with_kappa(post_means, gaussians.standard_deviations(post_cov), alpha=100.0)
        case "CME-2":
            poo, kappa = cme_poo_with_kappa(post_means, gaussians.standard_deviations(post_cov), alpha=100.0)
            threshold = max(jnp.max(observation_values), kappa)
            return probability_of_improvement(post_means, gaussians.standard_deviations(post_cov), threshold, marginal_threshold=0), kappa
            #if True:#observation_values.size % 2 == 0: # go for second best instead of best
            #    poo = poo.at[jnp.argmax(poo)].set(0)
            #    poo = poo / jnp.sum(poo)
            #return poo, kappa
        #case "2CME":
        #    return cme_poo_with_kappa(post_means, gaussians.standard_deviations(post_cov), alpha=100.0, scaling=2)
        case "VAPOR":
            return vapor_poo_with_kappa(post_means, gaussians.standard_deviations(post_cov), alpha=100.0)
        case "OCME": # optimistic CME where due to optimism bias kappa > max_x mu_{F_x | D}
            return ocme_poo_with_kappa(post_means, gaussians.standard_deviations(post_cov), alpha=100.0, exploration_factor=1.0)
        case "CES": # CME Entropy Search
            return entropy_search_global(post_means, post_cov, observation_noise_stds, random_key, alpha=10.0, samples=5, using_quantiles=False, unroll=10)
        case "ES": # Vanilla Entropy Search
            return entropy_search_global(post_means, post_cov, observation_noise_stds, random_key, alpha=2.0/post_means.size, samples=5, using_quantiles=False, unroll=10, using_CME=False)
        case "MES": # Max-Value Entropy Search
            return max_value_entropy_search(post_means, gaussians.standard_deviations(post_cov), observation_noise_stds, random_key, gumbel=False, samples=10)
        case "NAMES": # Noise-Aware Max-value Entropy Search
            return max_value_entropy_search(post_means, gaussians.standard_deviations(post_cov), observation_noise_stds, random_key, gumbel=False, samples=10, account_for_sampling_noise=True)
        #case "FCES": # Fast CES only looks at a few candidate points sampled from TS
        #    key1, key2 = jax.random.split(random_key)
        #    positions = jnp.unique(gaussians.sample_n_arg_max(post_means, post_cov, n=2, random_key=key1, unroll=10)) # up to 2 positions
        #    mis = entropy_search(positions, post_means, post_cov, observation_noise_stds, key2, alpha=100.0, samples=10, using_quantiles=False, unroll=10)
        #    acquisition_function = - jnp.finfo(jnp.float32).max * jnp.ones_like(post_means)
        #    return acquisition_function.at[positions].set(mis)
        case "IG":
            information_gain = .5 * jnp.log(1 + jnp.diag(post_cov) / observation_noise_stds**2)
            return information_gain
        case "PIG": # POO weighted information gain
            information_gain = .5 * jnp.log(1 + jnp.diag(post_cov) / observation_noise_stds**2)
            poo, kappa = cme_poo_with_kappa(post_means, gaussians.standard_deviations(post_cov), alpha=100.0)
            return (information_gain * poo, kappa)
        case _:
            assert False, "acquisition function not defined, must be one of " + poo_estimators_and_BO_listed
            return 0