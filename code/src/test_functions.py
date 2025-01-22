# Copyright (c) 2025 Nicolas Menet, Jonas Hübotter, Parnian Kassraie, Andreas Krause
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from typing import Tuple

import jax

import src.quadrotor as quadrotor


test_functions_listed = ["random_linear", "drop-wave", "drop-wave-mini", "drop-wave-medium", "quadrotor"]

def get_test_function(name:str, return_mesh:bool=False, seed:int=0) -> Tuple[jax.Array, jax.Array]:
    """Returns a set of hardcoded test functions on which to run Bayesian optimisation.

    Args:
        name (str): name of the test function. Must be one of test_functions_listed
        return_mesh (bool): whether to additionally also return the meshgrid x1 and x2, useful for plotting
        seed (int): the seed for generation of the test function

    Returns:
        Tuple[jax.Array, jax.Array, float]: (x, f_true, true_obs_noise_std) where x describes the domain geometry as a tensor of shape (|X|, d),
                                    f_true the function on that domain (as a tensor of shape (|X|,)), and true_obs_noise_std the std of the 
                                    independent homoscedastic centred Gaussian noise added to f_true for each observation
    """
    assert name in test_functions_listed, name + " is not recognised as one of the hardcoded test functions: " + " ".join(test_functions_listed)
    match name:
        case "random_linear":
            dimension = int(1e3) # 1'000 dimensions
            domain_size = int(1e5) # 10'000 points
            key1, key2 = jax.random.split(jax.random.key(seed))
            hyperplane = jax.random.normal(key1, shape=(1,dimension)) # (1, d)
            x = jax.random.ball(key2, dimension, p=2, shape=(domain_size,)) # (|X|, d)
            f_true = jax.numpy.sum(x * hyperplane, axis=1)
            true_obs_noise_std = 0.1

        case "drop-wave":
            x1, x2 = jax.numpy.meshgrid(jax.numpy.linspace(start=-5, stop=4, num=300),
                                        jax.numpy.linspace(start=-5, stop=4, num=300))
            x = jax.numpy.reshape(jax.numpy.dstack((x1, x2)), (300**2, 2)) # (|X|, 2)
            f_true = (1 + jax.numpy.cos(12*jax.numpy.sqrt(x[:,0]**2 + x[:,1]**2))) / (0.5 * (x[:,0]**2 + x[:,1]**2) + 2)
            true_obs_noise_std = 0.1
        case "drop-wave-medium":
            x1, x2 = jax.numpy.meshgrid(jax.numpy.linspace(start=-5, stop=4, num=100),
                                        jax.numpy.linspace(start=-5, stop=4, num=100))
            x = jax.numpy.reshape(jax.numpy.dstack((x1, x2)), (100**2, 2)) # (|X|, 2)
            f_true = (1 + jax.numpy.cos(12*jax.numpy.sqrt(x[:,0]**2 + x[:,1]**2))) / (0.5 * (x[:,0]**2 + x[:,1]**2) + 2)
            true_obs_noise_std = 0.1
        case "drop-wave-mini":
            x1, x2 = jax.numpy.meshgrid(jax.numpy.linspace(start=-2.5, stop=2, num=25),
                                        jax.numpy.linspace(start=-2.5, stop=2, num=25))
            x = jax.numpy.reshape(jax.numpy.dstack((x1, x2)), (25**2, 2)) # (|X|, 2)
            f_true = (1 + jax.numpy.cos(12*jax.numpy.sqrt(x[:,0]**2 + x[:,1]**2))) / (0.5 * (x[:,0]**2 + x[:,1]**2) + 2)
            true_obs_noise_std = 0.1
        case "quadrotor":
            assert return_mesh == False, "quadrotor does not admit a 2d mesh in 4d"
            key = jax.random.key(seed)
            x, f_true = quadrotor.get_domain_and_f(key, 400)
            true_obs_noise_std = quadrotor.NOISE_STD

    if return_mesh:
        return (x, f_true, true_obs_noise_std, x1, x2)
    else:
        return (x, f_true, true_obs_noise_std)
        