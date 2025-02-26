# Copyright (c) 2025 Nicolas Menet, Jonas Hübotter, Parnian Kassraie, Andreas Krause
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from functools import partial
from typing import Tuple
import jax
import jax.numpy as jnp

@jax.jit
def _p_gaussians_geq_threshold(means:jax.Array, standard_deviations:jax.Array, threshold:float) -> jax.Array:
    return jax.scipy.special.ndtr((means - threshold) / standard_deviations) # Scipy Normal CDF

@jax.jit
def _find_normalising_threshold(means:jax.Array, standard_deviations:jax.Array, epsilon:float) -> Tuple[float, float]:
    """Finds κ such that Σ_i P[N[i] >= κ] ≈ 1 for N[i] ~ N(means[i], standard_deviations[i]^2)
    Args:
        means (jax.Array): (m,)
        standard_deviations (jax.Array): (m,)
        epsilon (float): element-wise absolute convergence threshold 
    Returns:
        Tuple[float, float]: (kappa_lower_bound, kappa_upper_bound)
    """
    min_mu = jnp.min(means)
    min_sigma = jnp.min(standard_deviations)
    max_mu = jnp.max(means)
    max_sigma = jnp.max(standard_deviations)
    beta = jax.scipy.special.ndtri(1/means.size) # Scipy inverse Normal CDF
    kappa_lower_bound = min_mu - beta * min_sigma
    kappa_upper_bound = max_mu - beta * max_sigma
    func = lambda a: _p_gaussians_geq_threshold(means, standard_deviations, threshold=a)

    @jax.jit
    def body_func(kappa_search_window: Tuple[float, float]):
        kappa_low, kappa_up = kappa_search_window
        kappa = (kappa_low + kappa_up) / 2
        probs = func(kappa) 
        normalisation_delta = 1 - jnp.sum(probs) # monotonously increasing in kappa
        kappa_low = jnp.where(normalisation_delta < 0, kappa, kappa_low)
        kappa_up = jnp.where(normalisation_delta >= 0, kappa, kappa_up)

        return kappa_low, kappa_up
    @jax.jit
    def cond_func(kappa_search_window: Tuple[float, float]):
        kappa_low, kappa_up = kappa_search_window
        probs_low = func(kappa_up)
        probs_up = func(kappa_low)
        return jnp.max(probs_up - probs_low) >= epsilon
    
    kappa_low, kappa_up = jax.lax.while_loop(cond_func, body_func, (kappa_lower_bound, kappa_upper_bound))
    return kappa_low, kappa_up

@partial(jax.jit, static_argnames=['epsilon'])
def flite_pom(gaussian_means:jax.Array, gaussian_stds:jax.Array, epsilon:float=None) -> jax.Array:
    """Evaluates the F-LITE estimator of probability of maximality
    Args:
        gaussian_means (jax.Array): (m,)
        gaussian_stds (jax.Array): (m,)
        epsilon (float): element-wise absolute convergence threshold, defaults to 1/(100·m)
    Returns:
        jax.Array: (m,) the probabilities of maximality
    """
    if epsilon is None:
        epsilon = 1/(100 * gaussian_means.size) 
    kappa_low, kappa_up = _find_normalising_threshold(gaussian_means, gaussian_stds, epsilon)
    r_up =  _p_gaussians_geq_threshold(gaussian_means, gaussian_stds, threshold=kappa_low)
    r_low = _p_gaussians_geq_threshold(gaussian_means, gaussian_stds, threshold=kappa_up)
    r = (r_up + r_low) / 2
    r /= r.sum() # evens out rounding errors
    return r

@partial(jax.jit, static_argnames=['epsilon'])
def flite_pom_entropy(gaussian_means:jax.Array, gaussian_stds:jax.Array, epsilon:float=None) -> float:
    """Computes entropy of probability of maximality according to the F-LITE estimator
    Args:
        gaussian_means (jax.Array): (m,)
        gaussian_stds (jax.Array): (m,)
        epsilon (float): element-wise absolute convergence threshold, defaults to 1/(100·m)
    Returns:
        float: the entropy of probability of maximality
    """
    poms = flite_pom(gaussian_means, gaussian_stds, epsilon)
    return jnp.sum(jax.scipy.special.entr(poms))
