import wandb
import jax
jax.config.update("jax_enable_x64", True)
from jax import random
import jax.numpy as jnp
import matplotlib.pyplot as plt
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.size'] = 14
from matplotlib import cm
import PIL
import json
import datetime
import argparse
import time
import os

import src.kernels as kernels
import src.gaussians as gaussians
import src.poo_estimators_and_BO as poo_estimators_and_BO

parser = argparse.ArgumentParser(description='visualises the querying process for a specified method', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('method', type=str, help="the method according to which querying takes place, one of " + poo_estimators_and_BO.poo_estimators_and_BO_listed)
parser.add_argument('-s', '--seed', type=int, default=0, help='the random seed for reproducibility')
parser.add_argument('-t', '--timeit', action="store_true", help='whether to print the elapsed time for the given method')
parser.add_argument('-c', '--domain_cardinality', type=int, default=250, help='the cardinality of the domain, i.e. the resolution of the subsampling of a continuous Gaussian process')
parser.add_argument('-k', '--kernel', type=str, default="gaussian", help="which kernel to use, must be one of 'gaussian', 'laplacian', 'linear', or 'independent'")
parser.add_argument('-l', '--length_scale', type=float, default=0.02, help='the length scale of the kernel for the prior')
parser.add_argument('-a', '--prior_amplitude', type=float, default=1.0, help='the amplitude of the prior belief (true function is sampled from copied prior with fixed amplitude 1)')
parser.add_argument('-o', '--number_of_observations', type=int, default=1000, help='the number of consecutive observations to perform')
parser.add_argument('--observations_showed', type=int, default=20, help='after how many observations to show the posterior along with the observations')
parser.add_argument('-n', '--observation_noise_std', type=float, default=0.1, help='the homoscedastic noise associated with all observations')
parser.add_argument('-het', '--observation_noise_heteroscedasticity', type=float, default=0, help='the heteroscedasticity in the observation noise (in [0,1), since 1 will lead to division by zero in most algorithms)')
parser.add_argument('-r', '--repetitions', type=int, default=20, help='the number of times the sampling experiment is repeated to estimate mean and stdv of regrets')

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
            "method": args.method,
            "seed": args.seed,
            "print elapsed time": args.timeit,
            "domain cardinality": args.domain_cardinality,
            "kernel" : args.kernel,
            "length scale": args.length_scale,
            "prior amplitude": args.prior_amplitude,
            "# of observations": args.number_of_observations,
            "observation noise standard deviation": args.observation_noise_std,
            "observation noise heteroscedasticity": args.observation_noise_heteroscedasticity,
            "repetitions": args.repetitions
        },
        save_code=True,
        name="regret experiment",
        #mode="offline"
    )

# seed
random_key = random.key(args.seed)

# defining domain
x = jnp.expand_dims(jnp.arange(args.domain_cardinality) / args.domain_cardinality, axis = 1) # (m, 1)

regrets = jnp.zeros((args.repetitions, args.number_of_observations))
simple_regrets = jnp.zeros((args.repetitions, args.number_of_observations))
cumulative_regrets = jnp.zeros((args.repetitions, args.number_of_observations))

entropies = jnp.zeros((args.repetitions, args.number_of_observations))
kappa_deltas = jnp.zeros((args.repetitions, args.number_of_observations))

# on average the observation noise std is always args.observation_noise_std, but for args.observation_noise_heteroscedasticity = 1 the heteroscedastic coefficient goes from 2 to 1
observation_noise_std = (args.observation_noise_heteroscedasticity * jnp.linspace(start=2, stop=0, num=args.domain_cardinality)\
                        + (1-args.observation_noise_heteroscedasticity) * jnp.ones(args.domain_cardinality)) * args.observation_noise_std
random_key, one_time_key = random.split(random_key)
observation_noise_std = jax.random.permutation(one_time_key, observation_noise_std)

# repeat to produce mean and stdv estimates
for r in range(args.repetitions):
    # sample some f_true
    random_key, one_time_key = random.split(random_key)
    means = jnp.zeros((args.domain_cardinality,))
    cov = gaussians.covariance_matrix_from_kernel(kernel, kernel_args={'length_scale':args.length_scale, 'amplitude':1.0}, positions=x)
    if args.timeit:
        start_time = time.time()
    sqrt_cov = gaussians.spsd_matrix_square_root(cov)
    if args.timeit:
        sqrt_cov = sqrt_cov.block_until_ready()
        print(f'[{r}] {time.time() - start_time} seconds for matrix square root.')
    f_true = gaussians.sample(means, sqrt_cov, one_time_key)
    f_max = jnp.max(f_true)

    # Gaussian process prior
    prior_means = jnp.zeros((args.domain_cardinality,))
    prior_cov = gaussians.covariance_matrix_from_kernel(kernel, kernel_args={'length_scale':args.length_scale, 'amplitude':args.prior_amplitude}, positions=x)

    observation_indices = jnp.zeros((args.number_of_observations,), dtype=int)
    observation_values = jnp.zeros((args.number_of_observations,))

    regret = jnp.zeros((args.number_of_observations,))
    simple_regret = jnp.zeros((args.number_of_observations,))

    entropy = jnp.zeros((args.number_of_observations,))
    kappa_delta = jnp.zeros((args.number_of_observations,))

    # first observation is made deterministically in the center,
    # this ensures that POI and EI are well-defined and that generally the maximum of any acquisition function is a.s. unique (up to numerical precision))
    query = args.domain_cardinality//2

    # simulate noisy observation
    random_key, one_time_key = random.split(random_key)
    observation_noise = observation_noise_std[query] * jax.random.normal(one_time_key, (1,))
    query_value = f_true[query] + observation_noise
    observation_indices = observation_indices.at[0].set(query)
    observation_values = observation_values.at[0].set(query_value.item())
    f_max_sampled = f_true[query]

    # update regret
    regret = regret.at[0].set(f_max - f_true[query])
    simple_regret = simple_regret.at[0].set(f_max - f_max_sampled)

    # Gaussian Process posterior
    post_means, post_cov = gaussians.condition_on_observations(prior_means, prior_cov, 
                                                            observation_indices[0:1], observation_values[0:1], observation_noise_std[query:query+1])
    
    # update entropy
    if args.timeit:
        start_time = time.time()
        #random_key, one_time_key = random.split(random_key)
        #pom = poo_estimators_and_BO.etse_poo(post_means, post_cov, int(relative_accuracy**2 * post_means.size**2), one_time_key, unroll=1)
        pom = poo_estimators_and_BO.cme_poo(post_means, gaussians.standard_deviations(post_cov), alpha=100)
    if args.timeit:
        pom = pom.block_until_ready()
        print(f'[{r}] {time.time() - start_time} seconds for estimating entropy of posterior POO/PoM with CME/F-LITE.')
    entropy = entropy.at[0].set(jnp.sum(jax.scipy.special.entr(pom)))

    # repeatedly sample and recompute acquisition function
    # it should be noted that we always compute the complete covariance matrix upon conditioning, which for many methods (e.g. IE or CME) is not necessary
    for i in range(1, args.number_of_observations):
        print(f"ITERATION {i}")
        if args.timeit:
            start_time = time.time()
        random_key, one_time_key = random.split(random_key)
        if "CME" in args.method or "PIG" in args.method or "VAPOR" in args.method or "EST" in args.method:
            a_f, kappa = poo_estimators_and_BO.get_acquisition_function(args.method, post_means, post_cov, observation_noise_std, observation_values[:i], one_time_key)
            kappa_delta = kappa_delta.at[i].set(kappa - jnp.max(f_true))
        else:
            a_f = poo_estimators_and_BO.get_acquisition_function(args.method, post_means, post_cov, observation_noise_std, observation_values[:i], one_time_key)
        if args.timeit:
            post_cov = post_cov.block_until_ready()
            print(f'[{r}] {time.time() - start_time} seconds for obtaining acquisition function.')

        # save f_true, posterior, and observations for visualisation after args.observations_showed (20) observations in first repetition
        if r == 0 and i == args.observations_showed:
            vis_f_true = f_true.copy()
            vis_observation_indices = observation_indices.at[:args.observations_showed].get()
            vis_observation_values = observation_values.at[:args.observations_showed].get()
            vis_post_means = post_means.copy()
            vis_post_stds = gaussians.standard_deviations(post_cov)
            vis_acquisition_function = a_f

        query = jnp.argmax(a_f)

        # simulate noisy observation
        random_key, one_time_key = random.split(random_key)
        observation_noise = observation_noise_std[query] * jax.random.normal(one_time_key, (1,))
        query_value = f_true[query] + observation_noise
        observation_indices = observation_indices.at[i].set(query.item())
        observation_values = observation_values.at[i].set(query_value.item())
        f_max_sampled = max(f_max_sampled, f_true[query])

        # update regret
        regret = regret.at[i].set(f_max - f_true[query])
        simple_regret = simple_regret.at[i].set(f_max - f_max_sampled)

        # condition Gaussian process on the new observation
        if args.timeit:
            start_time = time.time()
        post_means, post_cov = gaussians.condition_on_observations(post_means, post_cov, observation_indices[i:i+1], observation_values[i:i+1], observation_noise_std[query:query+1])
        if args.timeit:
            post_cov = post_cov.block_until_ready()
            print(f'[{r}] {time.time() - start_time} seconds for conditioning.')

        # update PoM entropy
        relative_accuracy = 1
        if args.timeit:
            start_time = time.time()
        #random_key, one_time_key = random.split(random_key)
        #pom = poo_estimators_and_BO.etse_poo(post_means, post_cov, int(relative_accuracy**2 * post_means.size**2), one_time_key, unroll=1)
        pom = poo_estimators_and_BO.cme_poo(post_means, gaussians.standard_deviations(post_cov), alpha=100)
        if args.timeit:
            pom = pom.block_until_ready()
            print(f'[{r}] {time.time() - start_time} seconds for estimating entropy of posterior POO/PoM with CME/F-LITE.')
        entropy = entropy.at[i].set(jnp.sum(jax.scipy.special.entr(pom)))

    regrets = regrets.at[r, :].set(regret)
    simple_regrets = simple_regrets.at[r, :].set(simple_regret)
    cumulative_regrets = cumulative_regrets.at[r, :].set(jnp.cumsum(regret))

    entropies = entropies.at[r, :].set(entropy)
    kappa_deltas = kappa_deltas.at[r, :].set(kappa_delta)

    print(entropy)

# compute statistics of regrets
regret_mean = jnp.mean(regrets, axis=0)
regret_std = jnp.std(regrets, axis=0, ddof=1)
simple_regret_mean = jnp.mean(simple_regrets, axis=0)
simple_regret_std = jnp.std(simple_regrets, axis=0, ddof=1)
cumulative_regret_mean = jnp.mean(cumulative_regrets, axis=0)
cumulative_regret_std = jnp.std(cumulative_regrets, axis=0, ddof=1)

entropies_mean = jnp.mean(entropies, axis=0)
entropies_std = jnp.std(entropies, axis=0, ddof=1)
kappa_delta_mean = jnp.mean(kappa_deltas, axis=0)
kappa_delta_std = jnp.std(kappa_deltas, axis=0, ddof=1)


# set up plotting
if "CME" in args.method or "PIG" in args.method or "VAPOR" in args.method or "EST" in args.method: # adds axis to plot kappa
    fig, axs = plt.subplots(4, 1, figsize=(18, 8), dpi=400)
else:
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), dpi=400)

twin_axs0 = axs[0].twinx()
twin_axs1 = axs[1].twinx()
plt.subplots_adjust(top = 0.9, bottom=0.1, hspace=0.5, wspace=0.3)

# plot f_true of last repetition
axs[0].plot(x[:, 0], vis_f_true, c="k", label=r"$f_{true}$")

# plot f_posterior as well as observations after 20 observations in first repetition
axs[0].scatter(x[vis_observation_indices, 0], vis_observation_values, c="k", label="observations")
for n in range(len(vis_observation_indices)):
    axs[0].annotate(n, (x[vis_observation_indices[n], 0], vis_observation_values[n]+0.2))
axs[0].plot(x[:, 0], vis_post_means, "b", label=r"$\mu_{f\ |\ \mathcal{D}}$")
axs[0].fill_between(x[:, 0], vis_post_means - vis_post_stds, vis_post_means + vis_post_stds, color='b', alpha=.1, label=r"$\sigma_{f\ |\ \mathcal{D}}$")
twin_axs0.plot(x[:, 0], vis_acquisition_function, "r:", label=rf"$\alpha_{{{args.method}}}$")

# plot statistics for cumulative regret and simple regret
axs[1].plot(range(args.number_of_observations), cumulative_regret_mean, c="b", label=r"$\mu_{cum\, reg}$")
axs[1].fill_between(range(args.number_of_observations), cumulative_regret_mean - cumulative_regret_std, cumulative_regret_mean + cumulative_regret_std, color='b', alpha=.1, label=r"$\sigma_{cum\, reg}$")
twin_axs1.plot(range(args.number_of_observations), simple_regret_mean, c="k", label=r"$\mu_{sim\, reg}$")
twin_axs1.fill_between(range(args.number_of_observations), simple_regret_mean - simple_regret_std, simple_regret_mean + simple_regret_std, color='k', alpha=.1, label=r"$\sigma_{sim\, reg}$")

# plot statistics for entropy
axs[2].plot(range(args.number_of_observations), entropies_mean, c="b", label=r"$\mu_{H[X^*|\mathcal{D}]}$")
axs[2].fill_between(range(args.number_of_observations), entropies_mean - entropies_std, entropies_mean + entropies_std, color="b", alpha=.1, label=r"$\sigma_{H[X^*|\mathcal{D}]}$")

if "CME" in args.method or "PIG" in args.method or "VAPOR" in args.method or "EST" in args.method:
    axs[3].plot(range(1, args.number_of_observations), jnp.zeros((args.number_of_observations - 1,)), 'k.')
    axs[3].plot(range(1, args.number_of_observations), kappa_delta_mean[1:], "b-.", label=r"$\mu_{\kappa - f_{true}^*}$")
    axs[3].fill_between(range(1, args.number_of_observations), kappa_delta_mean[1:] - kappa_delta_std[1:], kappa_delta_mean[1:] + kappa_delta_std[1:], color='b', alpha=.1, label=r"$\sigma_{\kappa - f_{true}^*}$")

# adjust style
legend0 = axs[0].legend(loc="upper left")
legend0.remove() # avoids plotting over it
axs[0].set_xlabel("x")
axs[0].set_ylabel("")
#axs[0].set_title(rf"Querying $f_{{true}}$ based on {args.method}")
twin_axs0.legend(loc="upper right")
twin_axs0.add_artist(legend0)


legend1 = axs[1].legend(loc="upper left")
legend1.remove() # avoids plotting over it
axs[1].set_xlabel("t")
axs[1].set_ylabel("cumulative regret")
#axs[1].set_title(r"Regret")
twin_axs1.legend(loc="upper right")
twin_axs1.set_ylabel("simple regret")
twin_axs1.add_artist(legend1)
twin_axs1.set_yscale("log")

axs[2].legend(loc="upper right")
axs[2].set_xlabel("t")
axs[2].set_ylabel("differential entropy")

if "CME" in args.method or "PIG" in args.method or "VAPOR" in args.method or "EST" in args.method:
    axs[3].set_xlabel("t")
    axs[3].set_ylabel("")
    #axs[3].set_title(r"Difference between $\kappa$ and $\max f_{true}(x)$")
    axs[3].set_yscale("log")


date_time = datetime.datetime.strftime(datetime.datetime.now(), '%d:%b:%Y-%H:%M:%S')
plt.savefig(f'results/{date_time}.pdf', format='pdf') # figure
with open(f'results/{date_time}.json', 'w') as fp:
    information = {'script name' : os.path.basename(__file__)}
    information.update(vars(args))
    json.dump(information, fp) # parameters

if "CME" in args.method or "PIG" in args.method or "VAPOR" in args.method or "EST" in args.method:
    jax.numpy.savez(f'results/{date_time}', method=args.method, regret_mean=regret_mean, regret_std=regret_std, simple_regret_mean=simple_regret_mean, simple_regret_std=simple_regret_std, 
                cumulative_regret_mean=cumulative_regret_mean, cumulative_regret_std=cumulative_regret_std, entropies_mean=entropies_mean, entropies_std=entropies_std, kappa_delta_mean=kappa_delta_mean, kappa_delta_std=kappa_delta_std) # results
else:
    jax.numpy.savez(f'results/{date_time}', method=args.method, regret_mean=regret_mean, regret_std=regret_std, simple_regret_mean=simple_regret_mean, simple_regret_std=simple_regret_std, 
                cumulative_regret_mean=cumulative_regret_mean, cumulative_regret_std=cumulative_regret_std, entropies_mean=entropies_mean, entropies_std=entropies_std) # results

plt.show()

wandb.log({"querying f_true": wandb.Image(PIL.Image.frombytes('RGBA', fig.canvas.get_width_height(), fig.canvas.buffer_rgba(), 'raw'))})
wandb.finish()