# Copyright (c) 2025 Nicolas Menet, Jonas Hübotter, Parnian Kassraie, Andreas Krause
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from functools import partial
import jax
from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn


@jax.jit
def js_div(p:jax.Array, q:jax.Array) -> float:
    """computes the Jensen-Shannon divergence between p and q

    Args:
        p (jax.Array): (m,) a pmf, i.e. non-negative entries that sum to 1
        q (jax.Array): (m,) a pmf, i.e. non-negative entries that sum to 1
        
    Returns:
        float: D_KL(p||q) = 0.5 * (D_KL(p||(p+q)/2) + D_KL(q||(p+q)/2))
    """
    p = p.at[p < 1e-20].set(0)
    q = q.at[q < 1e-20].set(0)
    m = (p+q)/2
    return 0.5 * (kl_div(p, m) + kl_div(q, m))

@jax.jit
def kl_div(p:jax.Array, q:jax.Array) -> float:
    """computes the Kullback-Leibler divergence between p and q
       assumes p<<q, i.e. q(x) = 0 => p(x) = 0

    Args:
        p (jax.Array): (m,) a pmf, i.e. non-negative entries that sum to 1
        q (jax.Array): (m,) a pmf, i.e. non-negative entries that sum to 1
        
    Returns:
        float: D_KL(p||q) = sum_x log(p(x)/q(x)) p(x) in nats
    """
    p = p.at[p < 1e-20].set(0)
    q = q.at[q < 1e-20].set(0)
    return jax.numpy.sum(jax.scipy.special.kl_div(p, q))

@partial(jax.jit, static_argnames=['axis'])
def tv_dist(p:jax.Array, q:jax.Array, axis:int=None) -> float:
    """computes the total variation distance between p and q

    Args:
        p (jax.Array): (m,) a pmf, i.e. non-negative entries that sum to 1
        q (jax.Array): (m,) a pmf, i.e. non-negative entries that sum to 1
        axis (int): Axis along which the TV distance is computed. If None, it is computed along all the axes.
        
    Returns:
        float: d_TV(p||q) = 0.5 * sum_x |p(x) - q(x)|
    """
    return 0.5 * jax.numpy.sum(jax.numpy.abs(p - q), axis=axis)

@jax.jit
def sinkhorn_div(domain_points: jax.Array, p: jax.Array, q:jax.Array) -> float:
    """computes the sinkhorn divergence between p and q

    Args:
        domain_points (jax.Array): (m,d) a set of points that describe the domain (subset of R^d)
        p (jax.Array): (m,) a pmf, i.e. non-negative entries that sum to 1
        q (jax.Array): (m,) a pmf, i.e. non-negative entries that sum to 1
        
    Returns:
        float: div_sinkhorn(p,q)
    """
    geom = pointcloud.PointCloud(domain_points)
    ot_prob = linear_problem.LinearProblem(geom, a=p, b=q)
    solver = sinkhorn.Sinkhorn()
    ot = solver(ot_prob)
    return ot.reg_ot_cost # Entropy regularised OT cost