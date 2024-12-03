import wandb
import jax
jax.config.update("jax_enable_x64", True)
from jax import random
import jax.numpy as jnp
from jax.lib import xla_bridge
import optax
import json
import datetime
import argparse
import time
import os

import src.kernels as kernels
import src.gaussians as gaussians
import src.poo_estimators_and_BO as poo_estimators_and_BO
import src.divergences as divergences
import src.test_functions as test_functions
import src.hyperopt as hyperopt


print("BACKEND USED:", jax.default_backend()) # verifies if GPU was used 

parser = argparse.ArgumentParser(description='Runs Bayesian optimisation on a finite domain according to expected improvement and marginal-likelihood maximisation to fit the kernel parameters. Returns the posteriors at intermediate steps.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("test_function", type=str, help="the test function on which to run Bayesian optimisation based on expected improvement")
parser.add_argument("--domain_cardinality", type=int, default=-1, help="the cardinality of the 'uniformly random' subsampling of the domain, default incurs no subsampling")
parser.add_argument('-s', '--seed', type=int, default=0, help='the random seed for reproducibility, affects the observation noise during sampling and possibly the test function')
parser.add_argument('-d', '--delta', type=int, default=0.1, help='delta parameter of GP-UCB')
parser.add_argument('-rcc', '--record_complete_covariance', action="store_true", help="whether to record the complete covariance matrix instead of just the standard deviations")
parser.add_argument('-t', '--timeit', action="store_true", help='whether to print the elapsed time for substeps (e.g. conditioning etc.)')
parser.add_argument('-k', '--kernel', type=str, default="matern52", help="which kernel class to use, must be one of 'gaussian', 'laplacian', 'matern32', 'matern52', 'linear', or 'independent'")
parser.add_argument('--mlm_n_random_observations', type=int, default=50, help="how many uniformly random observations to perform before starting Bayesian optimisation. Is used to infer reasonable starting hyperparameters.")
parser.add_argument('--max_n_observations_for_mlm', type=int, default=1000, help="up to how many observations we keep refitting marginal likelihood maximisation, which scales in n_observations^3")
parser.add_argument('-n_skipped_observations_for_mlm', type=int, default=1, help="every how many observations we keep refitting marginal likelihood maximisation, which scales in n_observations^3")
parser.add_argument('-no', '--n_observations', type=int, default=500, help='the number of consecutive queries to f_true (after the warm start using mlm_n_random_observations random observations to infer hyperparameters), i.e. the runtime of Bayesian optimisation')
parser.add_argument('-a', '--acquisition_function', type=str, default="EI", help='the acquisition function used for Bayesian optimisation. Must be one of [EI, TS, UCB] , where TS samples from F-LITE')


args = parser.parse_args()

print(args) # for logging purposes

wandb.init(
        project="master-thesis",
        config={
            "test function": args.test_function,
            "domain cardinality": args.domain_cardinality,
            "seed": args.seed,
            #"delta": args.delta,
            "record complete covariance": args.record_complete_covariance,
            "print elapsed time": args.timeit,
            "kernel" : args.kernel,
            "MLM # random_observations": args.mlm_n_random_observations,
            "max # observations for MLM": args.max_n_observations_for_mlm,
            "# observations delta for MLM": args.n_skipped_observations_for_mlm,
            "# of observations": args.n_observations,
            #"acquisition function": args.acquisition_function,
            "device": xla_bridge.get_backend().platform,
        },
        save_code=True,
        name="posterior_during_sampling_of_test_function",
        #mode="offline"
    )

match args.kernel:
    case "gaussian":
        kernel = kernels.gaussian_kernel
    case "laplacian":
        kernel = kernels.laplacian_kernel
    case "matern32":
        kernel = kernels.matern32_kernel
    case "matern52":
        kernel = kernels.matern52_kernel
    case "linear":
        kernel = kernels.linear_kernel
    case 'brown':
        kernel = kernels.brownian_motion_kernel
    case 'independent':
        kernel = kernels.independent_kernel
    case _:
        assert False, "the provided kernel is unknown, verify that it is one of 'gaussian', 'laplacian', 'matern32', 'matern52', 'linear', or 'independent'"

# seed
random_key = random.key(args.seed)

# retrieve x, f_true and subsamle according to desired size
x, f_true, true_obs_noise_std = test_functions.get_test_function(args.test_function, seed=args.seed) # x in (|X|, d) and f_true in (|X|,)
random_key, one_time_key = random.split(random_key)
assert args.domain_cardinality <= f_true.size, "the (random subsample) domain cardinality cannot exceed that of the ground-truth domain"
if args.domain_cardinality != -1:
    subsampling = random.choice(key=one_time_key, a=f_true.size, shape=(args.domain_cardinality,), replace=False)
    x, f_true = x[subsampling, :], f_true[subsampling]

post_means = jnp.zeros((args.n_observations, x.shape[0],))
if args.record_complete_covariance:
    post_cov = jnp.zeros((args.n_observations, x.shape[0], x.shape[0]))
else:
    post_stds = jnp.zeros((args.n_observations, x.shape[0],))
observation_indices = jnp.zeros((args.n_observations + args.mlm_n_random_observations,), dtype=int) # (args.n_observations,)
observation_values  = jnp.zeros((args.n_observations + args.mlm_n_random_observations,), dtype=float) # (args.n_observations,)

# infer a first set of hyper-parameters through random observations of f_true + noise
for i in range(0, args.mlm_n_random_observations):
    random_key, one_time_key1, one_time_key2 = jax.random.split(random_key, 3)
    query = jax.random.randint(one_time_key1, (1,), minval=0, maxval=x.shape[0])
    observation = f_true[query] + jax.random.normal(one_time_key2, (1,)) * true_obs_noise_std
    observation_indices = observation_indices.at[i].set(query.item())
    observation_values = observation_values.at[i].set(observation.item())

# repeatedly query f_true according to UCB while also fitting the parameters of the prior using marginal likelihood maximisation
params = {"length_scale":1.0, "amplitude":1.0, "mean":0.0, "noise_std":1.0} # initial parameters before fitting using gradient descent
for i in range(0, args.n_observations):
    print(f"\n\nITERATION {i}")
    # perform marginal likelihood maximisation in O(n^3)
    if i+args.mlm_n_random_observations <= args.max_n_observations_for_mlm and i%args.n_skipped_observations_for_mlm == 0:
        if args.timeit:
            start_time = time.perf_counter()
        params, nlls = hyperopt.optimise_params(kernel, params, x[observation_indices[:i+args.mlm_n_random_observations], :], observation_values[:i+args.mlm_n_random_observations], 
                                                optimiser=optax.adam(learning_rate=0.01), num_iters=1_000, tol=1e-3)
        #params["length_scale"] = 0.4
        print(f'negative log marginal likelihood: {nlls[-1]} for model parameters:{params}')
        if args.timeit:
            print(f'{time.perf_counter() - start_time} seconds for marginal likelihood maximisation.')
 

    # deriving posterior from prior and data in O(|X| * n^2 + n^3) operations with O(|X|*n + n^2) memory if not args.record_complete_covariance, else in O(|X|^2 * n + |X| * n^2 + n^3) compute and O(|X|^2+n^2) memory
    if args.timeit:
        start_time = time.perf_counter()
    if args.record_complete_covariance:
        curr_post_means, curr_post_cov = gaussians.condition_on_observations(params['mean']*jnp.ones_like(f_true), gaussians.covariance_matrix_from_kernel(kernel, {k: v for k, v in params.items() if k not in {'mean', 'noise_std'}}, x), observation_indices[:i+args.mlm_n_random_observations], observation_values[:i+args.mlm_n_random_observations], jnp.ones((i+args.mlm_n_random_observations,)) * params['noise_std'])
        curr_post_stds = jnp.sqrt(jnp.diag(curr_post_cov))
        post_cov = post_cov.at[i, :, :].set(curr_post_cov) 
    else:
        curr_post_means, curr_post_stds = gaussians.condition_on_observations_diagonal_w_kernel(params['mean'], kernel, {k: v for k, v in params.items() if k not in {'mean', 'noise_std'}}, x, observation_indices[:i+args.mlm_n_random_observations], observation_values[:i+args.mlm_n_random_observations], jnp.ones((i+args.mlm_n_random_observations,)) * params['noise_std'])
        post_stds = post_stds.at[i, :].set(curr_post_stds)
    post_means = post_means.at[i, :].set(curr_post_means)
    if args.timeit:
        curr_post_stds = curr_post_stds.block_until_ready()
        print(f'{time.perf_counter() - start_time} seconds for conditioning on on observations.')

    # obtain a query according to the EI acquisition function in O(|X|)
    if args.timeit:
        start_time = time.perf_counter()

    if args.acquisition_function == "EI":
        a_f = poo_estimators_and_BO.expected_improvement(curr_post_means, curr_post_stds, jnp.max(observation_values[:i+args.mlm_n_random_observations]))
    elif args.acquisition_function == "TS":
        probs = poo_estimators_and_BO.cme_poo(curr_post_means, curr_post_stds, 100)
        random_key, one_time_key = random.split(random_key)
        query = jax.random.choice(one_time_key, curr_post_means.size, p=probs)
    elif args.acquisition_function == "UCB":
        a_f = poo_estimators_and_BO.ucb(curr_post_means, curr_post_stds, i, args.delta)
    elif args.acquisition_function == "ES":
        random_key, one_time_key = random.split(random_key)
        a_f = poo_estimators_and_BO.entropy_search_global_kernelized(curr_post_means, curr_post_stds, params['mean'], kernel, {k: v for k, v in params.items() if k not in {'mean', 'noise_std'}}, x, observation_indices[:i+args.mlm_n_random_observations], observation_values[:i+args.mlm_n_random_observations], jnp.ones((i+args.mlm_n_random_observations,)) * params['noise_std'],
        one_time_key, 100, samples=1, using_quantiles=False, unroll=1, using_CME=True)

    if args.acquisition_function != "TS":
        query = jnp.argmax(a_f)

    if args.timeit:
        query = query.block_until_ready()
        print(f'{time.perf_counter() - start_time} seconds for obtaining acquisition function.')
    #print(f'current query coordinates: {x[query, :]}')

    # simulate noisy observation
    random_key, one_time_key = random.split(random_key)
    query_value = f_true[query] + jax.random.normal(one_time_key, (1,)) * true_obs_noise_std
    observation_indices = observation_indices.at[i+args.mlm_n_random_observations].set(query.item())
    observation_values = observation_values.at[i+args.mlm_n_random_observations].set(query_value.item())
    print(f'current query value: {query_value}')

    # give feedback to human operator (console)
    mu_max_idx = jnp.argmax(post_means[i, :])
    print(f'\mu_F^max: {post_means[i, mu_max_idx]}')# at position {x[mu_max_idx, :]}' )


date_time = datetime.datetime.strftime(datetime.datetime.now(), '%d:%b:%Y-%H:%M:%S')
with open(f'results/{date_time}-{args.seed}-{args.test_function}.json', 'w') as fp:
    information = {'script name' : os.path.basename(__file__)}
    information.update(vars(args))
    json.dump(information, fp) # parameters

if args.record_complete_covariance:
    jax.numpy.savez(f'results/{date_time}-{args.seed}-{args.test_function}', seed=args.seed, test_function=args.test_function, # stored for coherent file naming
                x=x, f_true=f_true, 
                observation_indices=observation_indices, observation_values=observation_values, 
                post_means = post_means, post_cov = post_cov) # results
else:
    jax.numpy.savez(f'results/{date_time}-{args.seed}-{args.test_function}', seed=args.seed, test_function=args.test_function, # stored for coherent file naming
                x=x, f_true=f_true, 
                observation_indices=observation_indices, observation_values=observation_values, 
                post_means = post_means, post_stds = post_stds) # results

wandb.finish()