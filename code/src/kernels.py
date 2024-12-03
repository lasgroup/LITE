from functools import partial
from typing import Callable
import jax
import jax.numpy as jnp

@jax.jit
def gaussian_kernel(x1:jax.Array, x2:jax.Array, length_scale:float, amplitude:float) -> jax.Array:
    """Evaluates the kernel on x1 and x2

    Args:
        x1 (jax.Array): (d,) where d the dimensionality of each vector
        x2 (jax.Array): (d,) where d the dimensionality of each vector
        length_scale (float): the length scale of the gaussian kernel
        amplitude (float): the amplitude of the gaussian kernel. k(x,x) = amplitude^2

    Returns:
        jax.Array: the scalar result
    """
    return amplitude**2 * jnp.exp(-jnp.sum(jnp.square(x1-x2))/(2 * length_scale**2))

@jax.jit
def laplacian_kernel(x1:jax.Array, x2:jax.Array, length_scale:float, amplitude:float) -> jax.Array:
    """Evaluates the kernel on x1 and x2

    Args:
        x1 (jax.Array): (d,) where d the dimensionality of each vector
        x2 (jax.Array): (d,) where d the dimensionality of each vector
        length_scale (float): the length scale of the Laplacian kernel
        amplitude (float): the amplitude of the Laplacian kernel. k(x,x) = amplitude^2

    Returns:
        jax.Array: the scalar result
    """
    return amplitude**2 * jnp.exp(-jnp.sum(jnp.abs(x1-x2))/(length_scale))

def matern32_kernel(x1:jax.Array, x2:jax.Array, length_scale:float, amplitude:float) -> jax.Array:
    """Evaluates the kernel on x1 and x2

    Args:
        x1 (jax.Array): (d,) where d the dimensionality of each vector
        x2 (jax.Array): (d,) where d the dimensionality of each vector
        length_scale (float): the length scale of the Matern3/2 kernel
        amplitude (float): the amplitude of the Matern3/2 kernel. k(x,x) = amplitude^2

    Returns:
        jax.Array: the scalar result
    """
    tau = jnp.sqrt(jnp.sum(jnp.square(x1-x2))) / length_scale    
    return amplitude**2 * (1.0 + jnp.sqrt(3.0) * tau) * jnp.exp(-jnp.sqrt(3.0) * tau)

def matern52_kernel(x1:jax.Array, x2:jax.Array, length_scale:float, amplitude:float) -> jax.Array:
    """Evaluates the kernel on x1 and x2

    Args:
        x1 (jax.Array): (d,) where d the dimensionality of each vector
        x2 (jax.Array): (d,) where d the dimensionality of each vector
        length_scale (float): the length scale of the Matern5/2 kernel
        amplitude (float): the amplitude of the Matern5/2 kernel. k(x,x) = amplitude^2

    Returns:
        jax.Array: the scalar result
    """
    tau = jnp.sqrt(jnp.sum(jnp.square(x1-x2))) / length_scale    
    return amplitude**2 * (1.0 + jnp.sqrt(5.0) * tau + 5.0 / 3.0 * jnp.square(tau)) * jnp.exp(-jnp.sqrt(5.0) * tau)

@jax.jit
def linear_kernel(x1:jax.Array, x2:jax.Array, length_scale:float, amplitude:float) -> jax.Array:
    """Evaluates the kernel on x1 and x2

    Args:
        x1 (jax.Array): (d,) where d the dimensionality of each vector
        x2 (jax.Array): (d,) where d the dimensionality of each vector
        length_scale (float): the length scale of the linear kernel
        amplitude (float): the bias amplitude of the linear kernel. k(x,y) = amplitude^2 + x^T y / length_scale^2

    Returns:
        jax.Array: the scalar result
    """
    return amplitude**2 + jnp.dot(x1, x2) / length_scale**2

@jax.jit
def independent_kernel(x1:jax.Array, x2:jax.Array, length_scale:float, amplitude:float) -> jax.Array:
    """Evaluates the kernel on x1 and x2

    Args:
        x1 (jax.Array): (d,) where d the dimensionality of each vector
        x2 (jax.Array): (d,) where d the dimensionality of each vector
        length_scale (float): dummy parameter, not used
        amplitude (float): the amplitude, i.e. k(x,y) = amplitude^2 1_{x=y}

    Returns:
        jax.Array: the scalar result
    """
    return amplitude**2 * jnp.array_equal(x1, x2)

@jax.jit
def brownian_motion_kernel(x1:jax.Array, x2:jax.Array, length_scale:float, amplitude:float) -> jax.Array:
    """Evaluates the kernel on x1 and x2

    Args:
        x1 (jax.Array): (1,)
        x2 (jax.Array): (1,)
        length_scale (float): the length scale of the brownian motion kernel
        amplitude (float): the bias amplitude of the brownian motion kernel. k(x,y) = amplitude^2 + min(x,y) / length_scale^2

    Returns:
        jax.Array: the scalar result
    """
    assert jnp.shape(x1)[0] == 1, "the time-axis in brownian motion cannot be multivariate"
    #assert jnp.all(x1 >= 0), "in brownian motion no negative time is allowed"
    return amplitude**2 + jnp.minimum(x1[0], x2[0]) / length_scale**2

@partial(jax.jit, static_argnums=(2,))
def _evaluate_kernel_1v_1v(x1:jax.Array, x2:jax.Array, kernel:Callable, kernel_args:dict) -> jax.Array:
    return kernel(x1, x2, **kernel_args)

_evaluate_kernel_nv_1v      = jax.jit(jax.vmap(_evaluate_kernel_1v_1v, in_axes=[0, None, None, None], out_axes=0), static_argnums=(2,))

_evaluate_kernel_1v_nv      = jax.jit(jax.vmap(_evaluate_kernel_1v_1v, in_axes=[None, 0, None, None], out_axes=0), static_argnums=(2,))

_evaluate_kernel_nv_nv      = jax.jit(jax.vmap(_evaluate_kernel_nv_1v, in_axes=[None, 0, None, None], out_axes=1), static_argnums=(2,))

_evaluate_kernel_nv_nv_diag = jax.jit(jax.vmap(_evaluate_kernel_1v_1v, in_axes=[0, 0, None, None], out_axes=0), static_argnums=(2,))

@partial(jax.jit, static_argnums=(2,))
def evaluate_kernel(x1:jax.Array, x2:jax.Array, kernel:Callable, kernel_args:dict) -> jax.Array:
    """Evaluates the supplied kernel on x1 and x2

    Args:
        x1 (jax.Array): (d,) or (m, d) where m denotes the number of vectors and d the dimensionality of each vector
        x2 (jax.Array): (d,) or (n, d) where n denotes the number of vectors and d the dimensionality of each vector
        kernel (function): which kernel function, taking (vector, vector, **kernel_args), to evaluate
        args (dict): _description_

    Returns:
        jax.Array: A jax array of shape (m,n) filled with kernel(x1[i, :], x2[j, :], **kernel_args) at position (i,j)
    """
    len_x1 = len(jnp.shape(x1))
    len_x2 = len(jnp.shape(x2))
    assert len_x1 == 1 or len_x1 == 2, "x1 should be of shape (d,) or (m, d), but is of shape " + str(jnp.shape(x1))
    assert len_x2 == 1 or len_x2 == 2, "x2 should be of shape (d,) or (n, d), but is of shape " + str(jnp.shape(x2))
    match (len_x1, len_x2): # python control flow because function is rejitted for each different shape
        case (1,1):
            return _evaluate_kernel_1v_1v(x1, x2, kernel, kernel_args)
        case (2,1):
            return _evaluate_kernel_nv_1v(x1, x2, kernel, kernel_args)
        case (1,2):
            return _evaluate_kernel_1v_nv(x1, x2, kernel, kernel_args)
        case (2,2):
            return _evaluate_kernel_nv_nv(x1, x2, kernel, kernel_args)
        
