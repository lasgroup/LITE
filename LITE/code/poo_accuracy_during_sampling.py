import wandb
import jax
jax.config.update("jax_enable_x64", True)
from jax import random
import jax.numpy as jnp
import matplotlib.pyplot as plt
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.size'] = 15
import PIL
import json
import datetime
import argparse
import time
import os

import src.kernels as kernels
import src.gaussians as gaussians
import src.poo_estimators_and_BO as poo_estimators_and_BO
import src.divergences as divergences

parser = argparse.ArgumentParser(description='measures the fidelity of several estimators compared against a groundtruth as established via E-TSE during a sampling procedure according to Thompson Sampling', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-s', '--seed', type=int, default=0, help='the random seed for reproducibility, affects f_true, TS, as well as E-TSE')
parser.add_argument('-t', '--timeit', action="store_true", help='whether to print the elapsed time for the estimators')
parser.add_argument('-c', '--domain_cardinality', type=int, default=300, help='the cardinality of the domain, i.e. the resolution of the subsampling of a continuous Gaussian process')
parser.add_argument('--two_d_domain', action="store_true", help='whether the (continuous) domain is a square instead of a line interval')
parser.add_argument('-k', '--kernel', type=str, default="gaussian", help="which kernel to use, must be one of 'gaussian', 'laplacian', 'linear', or 'independent'")
parser.add_argument('-l', '--length_scale', type=float, default=0.005, help='the length scale of the kernel for the prior')
parser.add_argument('-a', '--prior_amplitude', type=float, default=1.0, help='the amplitude of the prior belief (true function is sampled from copied prior with fixed amplitude 1)')
parser.add_argument('-n', '--observation_noise_std', type=float, default=0.1, help='the homoscedastic noise associated with each observation')
parser.add_argument('-no', '--n_observations', type=int, default=100, help='the number of consecutive samples from f_true')
parser.add_argument('-nos', '--n_observations_showed', type=int, default=20, help='after how many observations to show the estimated posteriors along with the observations')
parser.add_argument('--alpha', type=float, default=10.0, help='reciprocal of the desired relative accuracy, E-TSE scales in O(|alpha^2 |X|^2)')

args = parser.parse_args()

print(args) # for logging purposes

match args.kernel:
    case "gaussian":
        kernel = kernels.gaussian_kernel
    case "laplacian":
        kernel = kernels.laplacian_kernel
    case "linear":
        kernel = kernels.linear_kernel
    case 'brown':
        kernel = kernels.brownian_motion_kernel
    case 'independent':
        kernel = kernels.independent_kernel
    case _:
        assert False, "the provided kernel is unknown, verify that it is one of 'gaussian', 'laplacian', 'linear', or 'independent'"

wandb.init(
        project="master-thesis",
        config={
            "seed": args.seed,
            "print elapsed time": args.timeit,
            "domain cardinality": args.domain_cardinality,
            "2D domain": args.two_d_domain,
            "kernel" : args.kernel,
            "length scale": args.length_scale,
            "prior amplitude": args.prior_amplitude,
            "observation noise standard deviation": args.observation_noise_std,
            "# of observations": args.n_observations,
            "# of observations showed": args.n_observations_showed,
            "alpha": args.alpha
        },
        save_code=True,
        name="poo_accuracy_during_sampling",
        #mode="offline"
    )

from jax.lib import xla_bridge
print("DEVICE USED: ", xla_bridge.get_backend().platform)

# seed
random_key = random.key(args.seed)

# defining domain
if args.two_d_domain:
    sqrt_domain_cardinality = int(args.domain_cardinality**.5)
    assert args.domain_cardinality**.5 == sqrt_domain_cardinality, "if the flag --two_d_domain is set --domain_cardinality must be a square number."
    x1, x2 = jnp.meshgrid(jnp.arange(sqrt_domain_cardinality) / sqrt_domain_cardinality,
                           jnp.arange(sqrt_domain_cardinality) / sqrt_domain_cardinality) # (sqrt(m), sqrt(m)), essentially already broad-cast from (sqrt(m), 1) and (1, sqrt(m)), respectively
    x = jnp.reshape(jnp.dstack((x1, x2)), (args.domain_cardinality, 2)) # (m, 2)
else:
    x = jnp.expand_dims(jnp.arange(args.domain_cardinality) / args.domain_cardinality, axis = 1) # (m, 1)


# sample some f_true
random_key, one_time_key = random.split(random_key)
means = jnp.zeros((args.domain_cardinality,))
cov = gaussians.covariance_matrix_from_kernel(kernel, kernel_args={'length_scale':args.length_scale, 'amplitude':1.0}, positions=x)
if args.timeit:
    start_time = time.perf_counter()
sqrt_cov = gaussians.spsd_matrix_square_root(cov)
if args.timeit:
    sqrt_cov = sqrt_cov.block_until_ready()
    print(f'{time.perf_counter() - start_time} seconds for matrix square root.')
f_true = gaussians.sample(means, sqrt_cov, one_time_key)

# Gaussian process prior (posterior without incorporating any data yet)
post_means = jnp.zeros((args.domain_cardinality,))
post_cov = gaussians.covariance_matrix_from_kernel(kernel, kernel_args={'length_scale':args.length_scale, 'amplitude':args.prior_amplitude}, positions=x)

observation_indices = jnp.zeros((args.n_observations,), dtype=int)
observation_values = jnp.zeros((args.n_observations,))

poo_entropy_etse = jnp.zeros((args.n_observations,))
poo_variance_etse = jnp.zeros((args.n_observations,))

ie_sinkhorn = jnp.zeros((args.n_observations,))
ie_tv = jnp.zeros((args.n_observations,))
poo_entropy_ie = jnp.zeros((args.n_observations,))
poo_variance_ie = jnp.zeros((args.n_observations,))

cme_sinkhorn = jnp.zeros((args.n_observations,))
cme_tv = jnp.zeros((args.n_observations,))
poo_entropy_cme = jnp.zeros((args.n_observations,))
poo_variance_cme = jnp.zeros((args.n_observations,))

#ocme_sinkhorn = jnp.zeros((args.n_observations,))
#ocme_tv = jnp.zeros((args.n_observations,))
#poo_entropy_ocme = jnp.zeros((args.n_observations,))
#poo_variance_ocme = jnp.zeros((args.n_observations,))


vapor_sinkhorn = jnp.zeros((args.n_observations,))
vapor_tv = jnp.zeros((args.n_observations,))
poo_entropy_vapor = jnp.zeros((args.n_observations,))
poo_variance_vapor = jnp.zeros((args.n_observations,))


nest_sinkhorn = jnp.zeros((args.n_observations,))
nest_tv = jnp.zeros((args.n_observations,))
poo_entropy_nest = jnp.zeros((args.n_observations,))
poo_variance_nest = jnp.zeros((args.n_observations,))


nie_sinkhorn = jnp.zeros((args.n_observations,))
nie_tv = jnp.zeros((args.n_observations,))
poo_entropy_nie = jnp.zeros((args.n_observations,))
poo_variance_nie = jnp.zeros((args.n_observations,))


# repeatedly sample according to TS
# it should be noted that we always compute the complete covariance matrix upon conditioning, which for most methods (e.g. IE and CME) is not necessary
for i in range(args.n_observations):
    if args.timeit:
        start_time = time.perf_counter()
    # obtain the Thompson Sampling acquisition function
    random_key, one_time_key = random.split(random_key)
    post_cov_sqrt = gaussians.spsd_matrix_square_root(post_cov)
    a_f =  gaussians.sample(post_means, post_cov_sqrt, one_time_key)
    if args.timeit:
        post_cov = post_cov.block_until_ready()
        print(f'{time.perf_counter() - start_time} seconds for TS.')

    # Compute IE POO
    if args.timeit:
        start_time = time.perf_counter()
    ie_poo = poo_estimators_and_BO.ie_poo(post_means, gaussians.standard_deviations(post_cov), alpha=args.alpha, unroll=1)
    #ie_poo = poo_estimators_and_BO.ie_poo_parallel_guaranteed(post_means, gaussians.standard_deviations(post_cov), alpha=args.alpha)
    ie_poo /= jnp.sum(ie_poo) # ensures exact normalisation
    if args.timeit:
        ie_poo = ie_poo.block_until_ready()
        print(f'{time.perf_counter() - start_time} seconds for IE.')

    # Compute CME POO
    if args.timeit:
        start_time = time.perf_counter()
    cme_poo = poo_estimators_and_BO.cme_poo(post_means, gaussians.standard_deviations(post_cov), alpha=args.alpha)
    cme_poo /= jnp.sum(cme_poo) # ensures exact normalisation
    if args.timeit:
        cme_poo = cme_poo.block_until_ready()
        print(f'{time.perf_counter() - start_time} seconds for CME.')

    # Compute OCME POO
    #if args.timeit:
    #    start_time = time.perf_counter()
    #ocme_poo = poo_estimators_and_BO.ocme_poo(post_means, gaussians.standard_deviations(post_cov), alpha=args.alpha)
    #ocme_poo /= jnp.sum(ocme_poo) # ensures exact normalisation
    #if args.timeit:
    #    ocme_poo = ocme_poo.block_until_ready()
    #    print(f'{time.perf_counter() - start_time} seconds for OCME.')

    # Compute VAPOR POO
    if args.timeit:
        start_time = time.perf_counter()
    vapor_poo = poo_estimators_and_BO.vapor_poo(post_means, gaussians.standard_deviations(post_cov), alpha=args.alpha)
    vapor_poo /= jnp.sum(vapor_poo) # ensures exact normalisation
    if args.timeit:
        vapor_poo = vapor_poo.block_until_ready()
        print(f'{time.perf_counter() - start_time} seconds for VAPOR.')

    # Compute NEST POO
    if args.timeit:
        start_time = time.perf_counter()
    nest_poo = poo_estimators_and_BO.est_poo(post_means, gaussians.standard_deviations(post_cov), random_key=None, alpha=args.alpha, normalised=False, use_integration=True)
    nest_poo /= jnp.sum(nest_poo) # ensures exact normalisation
    if args.timeit:
        nest_poo = nest_poo.block_until_ready()
        print(f'{time.perf_counter() - start_time} seconds for NEST.')

    # Compute NIE POO
    if args.timeit:
        start_time = time.perf_counter()
    nie_poo = poo_estimators_and_BO.nie_poo(post_means, gaussians.standard_deviations(post_cov), alpha=args.alpha)
    nie_poo /= jnp.sum(nie_poo) # ensures exact normalisation
    if args.timeit:
        nie_poo = nie_poo.block_until_ready()
        print(f'{time.perf_counter() - start_time} seconds for NIE.')

    # Compute E-TSE POO
    if args.timeit:
        start_time = time.perf_counter()
    random_key, one_time_key = random.split(random_key)
    etse_poo = poo_estimators_and_BO.etse_poo(post_means, post_cov, number_of_samples=int(args.alpha**2 * args.domain_cardinality**2), random_key=one_time_key)
    etse_poo /= jnp.sum(etse_poo) # ensures exact normalisation
    if args.timeit:
        etse_poo.block_until_ready()
        print(f'{time.perf_counter() - start_time} seconds for E-TSE.')

    if args.timeit:
        start_time = time.perf_counter()
    poo_entropy_etse = poo_entropy_etse.at[i].set(jnp.sum(jax.scipy.special.entr(etse_poo)))
    #poo_variance_etse = poo_variance_etse.at[i].set(jnp.sum(etse_poo * x**2) - jnp.sum(etse_poo * x)**2)

    ie_sinkhorn = ie_sinkhorn.at[i].set(divergences.sinkhorn_div(domain_points=x, p=etse_poo, q=ie_poo))
    ie_tv = ie_tv.at[i].set(divergences.tv_dist(p=etse_poo, q=ie_poo))
    poo_entropy_ie = poo_entropy_ie.at[i].set(jnp.sum(jax.scipy.special.entr(ie_poo)))
    #poo_variance_ie = poo_variance_ie.at[i].set(jnp.sum(ie_poo * x**2) - jnp.sum(ie_poo * x)**2)

    cme_sinkhorn = cme_sinkhorn.at[i].set(divergences.sinkhorn_div(domain_points=x, p=etse_poo, q=cme_poo))
    cme_tv = cme_tv.at[i].set(divergences.tv_dist(p=etse_poo, q=cme_poo))
    poo_entropy_cme = poo_entropy_cme.at[i].set(jnp.sum(jax.scipy.special.entr(cme_poo)))
    #poo_variance_cme = poo_variance_cme.at[i].set(jnp.sum(cme_poo * x**2) - jnp.sum(cme_poo * x)**2)

    #ocme_sinkhorn = ocme_sinkhorn.at[i].set(divergences.sinkhorn_div(domain_points=x, p=etse_poo, q=ocme_poo))
    #ocme_tv = ocme_tv.at[i].set(divergences.tv_dist(p=etse_poo, q=ocme_poo))
    #poo_entropy_ocme = poo_entropy_ocme.at[i].set(jnp.sum(jax.scipy.special.entr(ocme_poo)))
    #poo_variance_ocme = poo_variance_ocme.at[i].set(jnp.sum(ocme_poo * x**2) - jnp.sum(ocme_poo * x)**2)

    vapor_sinkhorn = vapor_sinkhorn.at[i].set(divergences.sinkhorn_div(domain_points=x, p=etse_poo, q=vapor_poo))
    vapor_tv = vapor_tv.at[i].set(divergences.tv_dist(p=etse_poo, q=vapor_poo))
    poo_entropy_vapor = poo_entropy_vapor.at[i].set(jnp.sum(jax.scipy.special.entr(vapor_poo)))
    #poo_variance_vapor = poo_variance_vapor.at[i].set(jnp.sum(vapor_poo * x**2) - jnp.sum(vapor_poo * x)**2)

    nest_sinkhorn = nest_sinkhorn.at[i].set(divergences.sinkhorn_div(domain_points=x, p=etse_poo, q=nest_poo))
    nest_tv = nest_tv.at[i].set(divergences.tv_dist(p=etse_poo, q=nest_poo))
    poo_entropy_nest = poo_entropy_nest.at[i].set(jnp.sum(jax.scipy.special.entr(nest_poo)))
    #poo_variance_nest = poo_variance_nest.at[i].set(jnp.sum(nest_poo * x**2) - jnp.sum(nest_poo * x)**2)

    nie_sinkhorn = nie_sinkhorn.at[i].set(divergences.sinkhorn_div(domain_points=x, p=etse_poo, q=nie_poo))
    nie_tv = nie_tv.at[i].set(divergences.tv_dist(p=etse_poo, q=nie_poo))
    poo_entropy_nie = poo_entropy_nie.at[i].set(jnp.sum(jax.scipy.special.entr(nie_poo)))
    #poo_variance_nie = poo_variance_nie.at[i].set(jnp.sum(nie_poo * x**2) - jnp.sum(nie_poo * x)**2)

    if args.timeit:
        ie_sinkhorn.block_until_ready()
        cme_sinkhorn.block_until_ready()
        #ocme_sinkhorn.block_until_ready()
        vapor_sinkhorn.block_until_ready()
        nest_sinkhorn.block_until_ready()
        nie_sinkhorn.block_until_ready()
        print(f'{time.perf_counter() - start_time} seconds to compute Sinkhorn divergences (and TV distances) to E-TSE for IE, CME, VAPOR, NEST, and NIE.')

    # save f_true, posterior, observations, and POOs for visualisation after args.n_observations_showed (15) observations in first repetition
    if i == args.n_observations_showed:
        vis_f_true = f_true.copy()
        vis_observation_indices = observation_indices.at[:args.n_observations_showed].get()
        vis_observation_values = observation_values.at[:args.n_observations_showed].get()
        vis_post_means = post_means.copy()
        vis_post_stds = gaussians.standard_deviations(post_cov)
        vis_acquisition_function = a_f
        vis_etse = etse_poo
        vis_ie = ie_poo
        vis_cme = cme_poo
        #vis_ocme = ocme_poo
        vis_vapor = vapor_poo
        vis_nest = nest_poo
        vis_nie = nie_poo

    query = jnp.argmax(a_f)

    # simulate noisy observation
    random_key, one_time_key = random.split(random_key)
    observation_noise = args.observation_noise_std * jax.random.normal(one_time_key, (1,))
    query_value = f_true[query] + observation_noise
    observation_indices = observation_indices.at[i].set(query.item())
    observation_values = observation_values.at[i].set(query_value.item())

    # condition Gaussian process on the new observation
    if args.timeit:
        start_time = time.perf_counter()
    post_means, post_cov = gaussians.condition_on_observations(post_means, post_cov, observation_indices[i:i+1], observation_values[i:i+1], args.observation_noise_std * jnp.ones((1,)))
    if args.timeit:
        post_cov = post_cov.block_until_ready()
        print(f'{time.perf_counter() - start_time} seconds for conditioning.')

date_time = datetime.datetime.strftime(datetime.datetime.now(), '%d:%b:%Y-%H:%M:%S')
with open(f'results/{date_time}-{args.seed}.json', 'w') as fp:
    information = {'script name' : os.path.basename(__file__)}
    information.update(vars(args))
    json.dump(information, fp) # parameters

jax.numpy.savez(f'results/{date_time}-{args.seed}', poo_entropy_etse=poo_entropy_etse,    vis_poo_etse=vis_etse,   #poo_variance_etse=poo_variance_etse,  
  ie_sinkhorn=ie_sinkhorn,       ie_tv=ie_tv,       poo_entropy_ie=poo_entropy_ie,        vis_poo_ie=vis_ie,       #poo_variance_ie=poo_variance_ie,      
  cme_sinkhorn=cme_sinkhorn,     cme_tv=cme_tv,     poo_entropy_cme=poo_entropy_cme,      vis_poo_cme=vis_cme,     #poo_variance_cme=poo_variance_cme,    
#ocme_sinkhorn=ocme_sinkhorn,    ocme_tv=ocme_tv,   poo_entropy_ocme=poo_entropy_ocme,    vis_poo_ocme=vis_ocme,   #poo_variance_ocme=poo_variance_ocme   
  vapor_sinkhorn=vapor_sinkhorn, vapor_tv=vapor_tv, poo_entropy_vapor=poo_entropy_vapor,  vis_poo_vapor=vis_vapor, #poo_variance_vapor=poo_variance_vapor,
  nest_sinkhorn=nest_sinkhorn,   nest_tv=nest_tv,   poo_entropy_nest=poo_entropy_nest,    vis_poo_nest=vis_nest,   #poo_variance_nest=poo_variance_nest,  
  nie_sinkhorn=nie_sinkhorn,     nie_tv=nie_tv,     poo_entropy_nie=poo_entropy_nie,      vis_poo_nie=vis_nie,     #poo_variance_nie=poo_variance_nie,    
  vis_f_true=vis_f_true, vis_observation_indices=vis_observation_indices, vis_observation_values = vis_observation_values, vis_post_means=vis_post_means,
  vis_post_stds=vis_post_stds, vis_acquisition_function=vis_acquisition_function
  ) # results

wandb.finish()