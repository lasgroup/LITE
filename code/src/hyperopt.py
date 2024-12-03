from typing import Tuple, Callable
from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, value_and_grad

import optax
from tqdm import tqdm

import src.kernels as kernels

@partial(jax.jit, static_argnames=['kernel'])
def negative_log_marginal_likelihood(
    kernel:Callable,
    params:dict,
    x:jax.Array,
    y:jax.Array) -> float:
    """computes the negative log marginal likelihood of a GP prior. Runtime scales in Theta(n^3) with a memory consumption in Theta(n^2) due to the cholesky decomposition.

    Args:
        kernel (Callable): one of the functions in kernel.py
        params (dict): parameters of the model, should include the keys noise_std, mean, as well as kernel-specific parameters such as length_scale and amplitude
        x (jax.Array): (n, d) observation locations
        y (jax.Array): (n,) observation values

    Returns:
        float: negative log marginal likelihood of data
    """
    noise_std = jnp.exp(params["log_noise_std"])
    mean = jnp.full((y.size,), params["mean"])
    kernel_params = {"length_scale": jnp.exp(params["log_length_scale"]), "amplitude": jnp.exp(params["log_amplitude"])}

    K_theta = kernels.evaluate_kernel(x, x, kernel, kernel_params) + noise_std**2 * jnp.identity(y.size)
    delta = y - mean
    L_theta = jnp.linalg.cholesky(K_theta)
    alpha = jnp.linalg.solve(L_theta.T, jnp.linalg.solve(L_theta, delta)) # K_theta^-1 (y-mean)

    log_marginal_likelihood = - 0.5 * jnp.inner(delta, alpha) - jnp.sum(jnp.log(jnp.diag(L_theta))) - 0.5 * y.size * jnp.log(2*jnp.pi)
    return - log_marginal_likelihood

def optimise_params(
    kernel:Callable,
    params:dict,
    x:jax.Array,
    y:jax.Array,
    optimiser:optax.GradientTransformation,
    num_iters=1_000,
    tol=1e-3,) -> Tuple[dict, list]:
    """optimises the model hyperparameters (kernel length scale, kernel amplitude, mean constant, and (homoscedastic centred) observation noise standard deviation)
    of a Gaussian process through minimisation of the negative log margimal likelihood of the data (x,y)

    Args:
        kernel (Callable): one of the functions in kernel.py
        params (dict): parameters of the GP prior and observation model. Should include 'noise_std', 'mean', and the kernel parameters of 'length_scale' and 'amplitude' 
        x (jax.Array): (n, d) observation locations
        y (jax.Array): (n,) observation values
        optimiser (optax.GradientTransformation): optimiser to use for gradient descent
        num_iters (int): the number of iterations in gradient descent
        tol (float): the tolerance for the stopping condition of gradient descent
    Returns:
        Tuple[dict, list]: (parameters, nlls) the final parameters as well as a list of the negative log marginal likelihood during optimisation
    """
    params = {"log_noise_std":    jnp.log(params["noise_std"]), 
              "mean":             params["mean"], 
              "log_length_scale": jnp.log(params["length_scale"]), 
              "log_amplitude":    jnp.log(params["amplitude"])}
    opt_state = optimiser.init(params) # with respect to which variables to take the gradient
    
    @jit
    def step(params, opt_state):
        nll, grads = value_and_grad(negative_log_marginal_likelihood, argnums=1)(kernel, params, x, y)
        updates, opt_state = optimiser.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, nll

    nlls = []
    pbar = tqdm(range(num_iters)) # indicates the maximum number of iterations in progress bar
    for i in pbar:
        params, opt_state, nll = step(params, opt_state)
        nlls.append(nll)
        pbar.set_description(f"{nll}")
        if i % 100 == 0 and i > 0 and nll + tol > nlls[-100]: # breaks if only 'tol' delta in the last 100 steps
            break

    params = {"noise_std":    jnp.exp(params["log_noise_std"]), 
              "mean":         params["mean"], 
              "length_scale": jnp.exp(params["log_length_scale"]), 
              "amplitude":    jnp.exp(params["log_amplitude"])} 

    return params, nlls