import jax
from jax import random
import jax.numpy as jnp
import json
import datetime
import argparse
import time
import os

import src.kernels as kernels
import src.gaussians as gaussians
import src.poo_estimators_and_BO as poo_estimators_and_BO

parser = argparse.ArgumentParser(description='Demonstrates that E-TSE requires a lot of samples for an accurate estimation of entropy. Runs bayesian optimisation using a prior matched to the data generation process. After --bayesian_optimisation_steps steps the posterior is used to estimate the entropy of X^* based on E-TSE with a varying number of samples.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-s', '--seed', type=int, default=0, help='the random seed for reproducibility')
parser.add_argument('-t', '--timeit', action="store_true", help='whether to print the elapsed time for each entropy estimation')
parser.add_argument('-c', '--domain_cardinality', type=int, default=250, help='the cardinality of the domain, i.e. the resolution of the subsampling of a continuous Gaussian process')
parser.add_argument('-k', '--kernel', type=str, default="gaussian", help="which kernel to use, must be one of 'gaussian', 'laplacian', 'linear', or 'independent'")
parser.add_argument('-l', '--length_scale', type=float, default=0.01, help='the length scale of the kernel for the prior')
parser.add_argument('-a', '--prior_amplitude', type=float, default=1, help='the amplitude of the prior belief (true function is sampled from copied prior with fixed amplitude 1)')
parser.add_argument('-bos', '--bayesian_optimisation_steps', type=int, default=15, help='the number of steps of bayesian optimisation (using Thompson sampling) before the actual experiment of entropy estimation takes place')
parser.add_argument('-n', '--observation_noise_std', type=float, default=0.1, help='the homoscedastic noise associated with all observations during bayesian optimisation')
parser.add_argument('-mns', '--max_n_samples', type=int, default=100000, help='the maximum number of Thompson samples in E-TSE that we consider')
parser.add_argument('-nns', '--n_number_of_samples', type=int, default=100, help='the number of different "numer of Thompson samples" in E-TSE that we consider on a log-scale')

args = parser.parse_args()

print(args) # for logging purposes

match args.kernel:
    case "gaussian":
        kernel = kernels.gaussian_kernel
    case "laplacian":
        kernel = kernels.laplacian_kernel
    case "linear":
        kernel = kernels.linear_kernel
    case 'independent':
        kernel = kernels.independent_kernel
    case _:
        assert False, "the provided kernel is unknown, verify that it is one of 'gaussian', 'laplacian', 'linear', or 'independent'"

# seed
random_key = random.key(args.seed)
x = jnp.expand_dims(jnp.arange(args.domain_cardinality) / args.domain_cardinality, axis = 1) # (m, 1)

# sample some f_true
random_key, one_time_key = random.split(random_key)
means = jnp.zeros((args.domain_cardinality,))
cov = gaussians.covariance_matrix_from_kernel(kernel, kernel_args={'length_scale':args.length_scale, 'amplitude':1}, positions=x)
if args.timeit:
    start_time = time.perf_counter()
sqrt_cov = gaussians.spsd_matrix_square_root(cov)
if args.timeit:
    sqrt_cov = sqrt_cov.block_until_ready()
    print(f'{time.perf_counter() - start_time} seconds for matrix square root.')
f_true = gaussians.sample(means, sqrt_cov, one_time_key)

# Gaussian Process prior
prior_means = jnp.zeros((args.domain_cardinality,))
prior_cov = gaussians.covariance_matrix_from_kernel(kernel, kernel_args={'length_scale':args.length_scale, 'amplitude':args.prior_amplitude}, positions=x)

post_means = jnp.copy(prior_means)
post_cov   = jnp.copy(prior_cov)

# run bayesian optimisation using a Thompson sampling acquisition function
if args.timeit:
    bo_start_time = time.perf_counter()

observation_indices = jnp.zeros((args.bayesian_optimisation_steps,), dtype=int)
observation_values  = jnp.zeros((args.bayesian_optimisation_steps,))
for step in range(args.bayesian_optimisation_steps):
    # Thompson sampling
    post_sqrt_cov = gaussians.spsd_matrix_square_root(post_cov)
    random_key, one_time_key1, one_time_key2 = random.split(random_key, 3)
    if args.timeit:
        start_time = time.perf_counter()
    posterior_sample = gaussians.sample(post_means, post_sqrt_cov, one_time_key1)
    if args.timeit:
        posterior_sample = posterior_sample.block_until_ready()
        print(f'{time.perf_counter() - start_time} seconds for sampling from posterior.')
    query = jnp.argmax(posterior_sample)
    observation = f_true[query] + args.observation_noise_std * jax.random.normal(one_time_key2, (1,))

    observation_indices = observation_indices.at[step].set(query)
    observation_values  =  observation_values.at[step].set(observation.item())

    # update posterior
    if args.timeit:
        start_time = time.perf_counter()
    post_means, post_cov = gaussians.condition_on_observations(post_means, post_cov, jnp.ones((1,), dtype=int) * query, jnp.ones((1,), dtype=float) * observation, jnp.ones((1,), dtype=float) * args.observation_noise_std)
    if args.timeit:
        post_cov = post_cov.block_until_ready()
        print(f'{time.perf_counter() - start_time} seconds for conditioning.')
if args.timeit:
    print(f'\n {time.perf_counter() - bo_start_time} seconds for complete bayesian optimisation.')

# run the entropy estimation experiment
n_of_samples_arr = jnp.zeros((args.n_number_of_samples,))
entropy_arr      = jnp.zeros((args.n_number_of_samples,))
for i in range(0, args.n_number_of_samples):
    n_of_samples = int(args.max_n_samples**(i/(args.n_number_of_samples - 1))) # logarithmic interpolation of number of samples between 1 and args.max_n_samples 
    n_of_samples_arr = n_of_samples_arr.at[i].set(n_of_samples) # store for visualisation

    if args.timeit:
        start_time = time.perf_counter()
    random_key, one_time_key = random.split(random_key) 
    etse_poo = poo_estimators_and_BO.etse_poo(post_means, post_cov, n_of_samples, one_time_key)
    entropy = jnp.sum(jax.scipy.special.entr(etse_poo))
    entropy_arr = entropy_arr.at[i].set(entropy)
    if args.timeit:
        print(f'{time.perf_counter() - start_time} seconds to estimate entropy with {n_of_samples} samples (goes up to {args.max_n_samples}) in E-TSE.')

date_time = datetime.datetime.strftime(datetime.datetime.now(), '%d:%b:%Y-%H:%M:%S')
with open(f'results/{date_time}.json', 'w') as fp:
    information = {'script name' : os.path.basename(__file__)}
    information.update(vars(args))
    json.dump(information, fp) # parameters

jax.numpy.savez(f'results/{date_time}', n_of_samples = n_of_samples_arr, entropies = entropy_arr, 
                                        f_true = f_true, observation_indices = observation_indices, observation_values = observation_values,
                                        post_means = post_means, post_stds = gaussians.standard_deviations(post_cov), etse_poo = etse_poo) # results