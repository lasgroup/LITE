import wandb
import jax
jax.config.update("jax_enable_x64", True)
from jax import random
import jax.numpy as jnp
import json
import datetime
import argparse
import time
import os

import src.poo_estimators_and_BO as poo_estimators_and_BO
import src.divergences as divergences

parser = argparse.ArgumentParser(description='measures the fidelity of several estimators compared against a groundtruth as established via IE as the domain size is increased on a logarithmic scale from 2 to --max_domain_size', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-s', '--seed', type=int, default=0, help='the random seed for reproducibility')
parser.add_argument('-t', '--timeit', action="store_true", help='whether to print the elapsed time for the estimators')
parser.add_argument('-md', '--max_domain_size', type=int, default=2000, help='the maximum domain size on which to evaluate the accuracy')
parser.add_argument('-nd', '--n_domain_sizes', type=int, default=40, help='the number of domain sizes that are evaluated between 2 and --max_domain_size')
parser.add_argument('-mr', '--mean_range', type=float, default=5, help='the range [0, mean_range] of means from which we sample uniformly')
parser.add_argument('-sr', '--std_range', type=float, default=10, help='the range [0.5, std_range] of stds from which we sample uniformly')
parser.add_argument('--alpha', type=float, default=200.0, help='reciprocal of the desired relative accuracy')

args = parser.parse_args()

print(args) # for logging purposes

wandb.init(
        project="master-thesis",
        config={
            "seed": args.seed,
            "print elapsed time": args.timeit,
            "maximum domain size": args.max_domain_size,
            "# of domain sizes to evaluate": args.n_domain_sizes,
            "range of means": args.mean_range,
            "range of stds": args.std_range,
            "alpha": args.alpha,
        },
        save_code=True,
        name="poo_accuracy_for_independent_gp",
        #mode="offline"
    )

# seed
random_key = random.key(args.seed)

domain_sizes = jax.numpy.floor(jax.numpy.logspace(start=1, stop=jax.numpy.log2(args.max_domain_size), num=args.n_domain_sizes, base=2)).astype(int)
domain_sizes = jax.numpy.unique(domain_sizes) # remove duplicates
n_domain_sizes = domain_sizes.size
#js_divs_mean = jnp.zeros((n_domain_sizes,))
#js_divs_stds = jnp.zeros((n_domain_sizes,))
#tv_dists_mean_cme = jnp.zeros((n_domain_sizes,))
#tv_dists_stds_cme = jnp.zeros((n_domain_sizes,))
#tv_dists_mean_ocme = jnp.zeros((n_domain_sizes,))
#tv_dists_stds_ocme = jnp.zeros((n_domain_sizes,))
#tv_dists_mean_vapor = jnp.zeros((n_domain_sizes,))
#tv_dists_stds_vapor = jnp.zeros((n_domain_sizes,))
#tv_dists_mean_est = jnp.zeros((n_domain_sizes,))
#tv_dists_stds_est = jnp.zeros((n_domain_sizes,))
#tv_dists_mean_nest = jnp.zeros((n_domain_sizes,))
#tv_dists_stds_nest = jnp.zeros((n_domain_sizes,))
#tv_dists_mean_nies = jnp.zeros((n_domain_sizes,))
#tv_dists_stds_nies = jnp.zeros((n_domain_sizes,))
#tv_dists_mean_nie = jnp.zeros((n_domain_sizes,))
#tv_dists_stds_nie = jnp.zeros((n_domain_sizes,))

tv_dists_cme = jnp.zeros((n_domain_sizes,))
#tv_dists_ocme = jnp.zeros((n_domain_sizes,))
tv_dists_vapor = jnp.zeros((n_domain_sizes,))
#tv_dists_est = jnp.zeros((n_domain_sizes,))
tv_dists_nest = jnp.zeros((n_domain_sizes,))
#tv_dists_nies = jnp.zeros((n_domain_sizes,))
tv_dists_nie = jnp.zeros((n_domain_sizes,))

for d, domain_size in enumerate(domain_sizes):
    #js_divs_repetitions = jnp.zeros((args.domain_size_repetitions,))
    #tv_dists_repetitions_cme = jnp.zeros((args.domain_size_repetitions,))
    #tv_dists_repetitions_vapor = jnp.zeros((args.domain_size_repetitions,))
    #tv_dists_repetitions_ocme = jnp.zeros((args.domain_size_repetitions,))
    #tv_dists_repetitions_est = jnp.zeros((args.domain_size_repetitions,))
    #tv_dists_repetitions_nest = jnp.zeros((args.domain_size_repetitions,))
    #tv_dists_repetitions_nies = jnp.zeros((args.domain_size_repetitions,))
    #tv_dists_repetitions_nie = jnp.zeros((args.domain_size_repetitions,))

    #for repetition in range(args.domain_size_repetitions):
        random_key, one_time_key = random.split(random_key)
        means = jax.random.uniform(key=one_time_key, shape=(domain_size,), minval=0, maxval=args.mean_range)
        random_key, one_time_key = random.split(random_key)
        stds = jax.random.uniform(key=one_time_key, shape=(domain_size,), minval=0.5, maxval=args.std_range)

        if args.timeit:# and repetition == 0:
            start_time = time.perf_counter()
        ie_probs = poo_estimators_and_BO.ie_poo(means, stds, args.alpha, unroll=1)
        if args.timeit:# and repetition == 0:
            ie_probs = ie_probs.block_until_ready()
            print(f'{time.perf_counter() - start_time} seconds for IE with domain size {domain_size}')

        if args.timeit:# and repetition == 0:
            start_time = time.perf_counter()
        cme_probs = poo_estimators_and_BO.cme_poo(means, stds, args.alpha)
        if args.timeit:# and repetition == 0:
            cme_probs = cme_probs.block_until_ready()
            print(f'{time.perf_counter() - start_time} seconds for CME with domain size {domain_size}')

        #if args.timeit:# and repetition == 0:
        #    start_time = time.perf_counter()
        #ocme_probs = poo_estimators_and_BO.ocme_poo(means, stds, args.alpha, exploration_factor=1.0)
        #if args.timeit:# and repetition == 0:
        #    ocme_probs = ocme_probs.block_until_ready()
        #    print(f'{time.perf_counter() - start_time} seconds for OCME with domain size {domain_size}')

        if args.timeit:# and repetition == 0:
            start_time = time.perf_counter()
        vapor_probs = poo_estimators_and_BO.vapor_poo(means, stds, args.alpha)
        if args.timeit:# and repetition == 0:
            vapor_probs = vapor_probs.block_until_ready()
            print(f'{time.perf_counter() - start_time} seconds for VAPOR with domain size {domain_size}')

        #if args.timeit:# and repetition == 0:
        #    start_time = time.perf_counter()
        #est_probs = poo_estimators_and_BO.est_poo(means, stds, None, args.alpha)
        #if args.timeit:# and repetition == 0:
        #    est_probs = est_probs.block_until_ready()
        #    print(f'{time.perf_counter() - start_time} seconds for EST with domain size {domain_size}')

        if args.timeit:# and repetition == 0:
            start_time = time.perf_counter()
        nest_probs = poo_estimators_and_BO.est_poo(means, stds, None, args.alpha, normalised=True)
        if args.timeit:# and repetition == 0:
            nest_probs = nest_probs.block_until_ready()
            print(f'{time.perf_counter() - start_time} seconds for NEST with domain size {domain_size}')

        #if args.timeit:# and repetition == 0:
        #    start_time = time.perf_counter()
        #random_key, one_time_key = random.split(random_key)
        #nies_probs = poo_estimators_and_BO.nies_poo(means, stds, 100, one_time_key, unroll=1)
        #if args.timeit:# and repetition == 0:
        #    nies_probs = nies_probs.block_until_ready()
        #    print(f'{time.perf_counter() - start_time} seconds for NIES with domain size {domain_size}')

        if args.timeit:# and repetition == 0:
            start_time = time.perf_counter()
        nie_probs = poo_estimators_and_BO.nie_poo(means, stds, args.alpha)
        if args.timeit:# and repetition == 0:
            nie_probs = nie_probs.block_until_ready()
            print(f'{time.perf_counter() - start_time} seconds for NIEQ with domain size {domain_size}')

        #js_divs_repetitions = js_divs_repetitions.at[repetition].set(divergences.js_div(p=ie_probs, q=cme_probs))
        #tv_dists_repetitions_cme = tv_dists_repetitions_cme.at[repetition].set(divergences.tv_dist(p=ie_probs, q=cme_probs))
        #tv_dists_repetitions_ocme = tv_dists_repetitions_ocme.at[repetition].set(divergences.tv_dist(p=ie_probs, q=ocme_probs))
        #tv_dists_repetitions_vapor = tv_dists_repetitions_vapor.at[repetition].set(divergences.tv_dist(p=ie_probs, q=vapor_probs))
        #tv_dists_repetitions_est = tv_dists_repetitions_est.at[repetition].set(divergences.tv_dist(p=ie_probs, q=est_probs))
        #tv_dists_repetitions_nest = tv_dists_repetitions_nest.at[repetition].set(divergences.tv_dist(p=ie_probs, q=nest_probs))
        #tv_dists_repetitions_nies = tv_dists_repetitions_nies.at[repetition].set(divergences.tv_dist(p=ie_probs, q=nies_probs))
        #tv_dists_repetitions_nie = tv_dists_repetitions_nie.at[repetition].set(divergences.tv_dist(p=ie_probs, q=nie_probs))

        tv_dists_cme   = tv_dists_cme.at[d].set(  divergences.tv_dist(p=ie_probs, q=cme_probs))
        #tv_dists_ocme  = tv_dists_ocme.at[d].set( divergences.tv_dist(p=ie_probs, q=ocme_probs))
        tv_dists_vapor = tv_dists_vapor.at[d].set(divergences.tv_dist(p=ie_probs, q=vapor_probs)) 
        #tv_dists_est   = tv_dists_est.at[d].set(  divergences.tv_dist(p=ie_probs, q=est_probs))
        tv_dists_nest  = tv_dists_nest.at[d].set( divergences.tv_dist(p=ie_probs, q=nest_probs))
        #tv_dists_nies  = tv_dists_nies.at[d].set( divergences.tv_dist(p=ie_probs, q=nies_probs))
        tv_dists_nie  = tv_dists_nie.at[d].set( divergences.tv_dist(p=ie_probs, q=nie_probs))


    #js_divs_mean = js_divs_mean.at[d].set(jnp.sum(js_divs_repetitions) / args.domain_size_repetitions)
    #tv_dists_mean_cme = tv_dists_mean_cme.at[d].set(jnp.sum(tv_dists_repetitions_cme) / args.domain_size_repetitions)
    #tv_dists_mean_ocme = tv_dists_mean_ocme.at[d].set(jnp.sum(tv_dists_repetitions_ocme) / args.domain_size_repetitions)
    #tv_dists_mean_vapor = tv_dists_mean_vapor.at[d].set(jnp.sum(tv_dists_repetitions_vapor) / args.domain_size_repetitions)
    #tv_dists_mean_est = tv_dists_mean_est.at[d].set(jnp.sum(tv_dists_repetitions_est) / args.domain_size_repetitions)
    #tv_dists_mean_nest = tv_dists_mean_nest.at[d].set(jnp.sum(tv_dists_repetitions_nest) / args.domain_size_repetitions)
    #tv_dists_mean_nies = tv_dists_mean_nies.at[d].set(jnp.sum(tv_dists_repetitions_nies) / args.domain_size_repetitions)
    #tv_dists_mean_nie = tv_dists_mean_nie.at[d].set(jnp.sum(tv_dists_repetitions_nie) / args.domain_size_repetitions)

    #js_divs_stds = js_divs_stds.at[d].set((jnp.sum((js_divs_repetitions-js_divs_mean[d])**2) / (args.domain_size_repetitions-1))**.5)
    #tv_dists_stds_cme = tv_dists_stds_cme.at[d].set((jnp.sum((tv_dists_repetitions_cme-tv_dists_mean_cme[d])**2) / (args.domain_size_repetitions-1))**.5)
    #tv_dists_stds_ocme = tv_dists_stds_ocme.at[d].set((jnp.sum((tv_dists_repetitions_ocme-tv_dists_mean_ocme[d])**2) / (args.domain_size_repetitions-1))**.5)
    #tv_dists_stds_vapor = tv_dists_stds_vapor.at[d].set((jnp.sum((tv_dists_repetitions_vapor-tv_dists_mean_vapor[d])**2) / (args.domain_size_repetitions-1))**.5)
    #tv_dists_stds_est = tv_dists_stds_est.at[d].set((jnp.sum((tv_dists_repetitions_est-tv_dists_mean_est[d])**2) / (args.domain_size_repetitions-1))**.5)
    #tv_dists_stds_nest = tv_dists_stds_nest.at[d].set((jnp.sum((tv_dists_repetitions_nest-tv_dists_mean_nest[d])**2) / (args.domain_size_repetitions-1))**.5)
    #tv_dists_stds_nies = tv_dists_stds_nies.at[d].set((jnp.sum((tv_dists_repetitions_nies-tv_dists_mean_nies[d])**2) / (args.domain_size_repetitions-1))**.5)
    #tv_dists_stds_nie = tv_dists_stds_nie.at[d].set((jnp.sum((tv_dists_repetitions_nie-tv_dists_mean_nie[d])**2) / (args.domain_size_repetitions-1))**.5)

    #print(f'js-div={js_divs_mean[d]} +- {js_divs_stds[d]} and tv-dist={tv_dists_mean_cme[d]} +- {tv_dists_stds_cme[d]} for domain size {domain_size}')
    #print(f'tv-dist={tv_dists_mean_cme[d]} +- {tv_dists_stds_cme[d]} for domain size {domain_size} (IE vs CME)')
    #print(f'tv-dist={tv_dists_mean_ocme[d]} +- {tv_dists_stds_ocme[d]} for domain size {domain_size} (IE vs OCME)')
    #print(f'tv-dist={tv_dists_mean_vapor[d]} +- {tv_dists_stds_vapor[d]} for domain size {domain_size} (IE vs VAPOR)')
    #print(f'tv-dist={tv_dists_mean_est[d]} +- {tv_dists_stds_est[d]} for domain size {domain_size} (IE vs EST)')
    #print(f'tv-dist={tv_dists_mean_nest[d]} +- {tv_dists_stds_nest[d]} for domain size {domain_size} (IE vs NEST)')
    #print(f'tv-dist={tv_dists_mean_nies[d]} +- {tv_dists_stds_nies[d]} for domain size {domain_size} (IE vs NIES)')
    #print(f'tv-dist={tv_dists_mean_nie[d]} +- {tv_dists_stds_nie[d]} for domain size {domain_size} (IE vs NIEQ)')
    #print("") # new line

date_time = datetime.datetime.strftime(datetime.datetime.now(), '%d:%b:%Y-%H:%M:%S')
with open(f'results/{date_time}.json', 'w') as fp:
    information = {'script name' : os.path.basename(__file__)}
    information.update(vars(args))
    json.dump(information, fp)

jax.numpy.savez(f'results/{date_time}', tv_dists_cme   = tv_dists_cme  ,
                                        #tv_dists_ocme = tv_dists_ocme ,
                                        tv_dists_vapor = tv_dists_vapor,
                                        #tv_dists_est  = tv_dists_est  ,
                                        tv_dists_nest  = tv_dists_nest ,
                                        #tv_dists_nies = tv_dists_nies ,
                                        tv_dists_nie  = tv_dists_nie,
)