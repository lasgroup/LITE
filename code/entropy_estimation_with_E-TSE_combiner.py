import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.size'] = 22
import json
import datetime
import argparse
import time
import os

parser = argparse.ArgumentParser(description='combines the results of several "entropy estimation with E-TSE" experiments into a single scatter plot', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("results", type=str, nargs="+", help="npz files of results of entropy estimation experiments")
parser.add_argument('-c', '--domain_cardinality', type=int, default=250, help='the cardinality of the domain, i.e. the resolution of the subsampling of a continuous Gaussian process. Used to visualise POOs of a particular run.')
parser.add_argument("-vr", "--visualisation_run", type=int, default=0, help="the index of the run that is used for the illustration of the POO")
args = parser.parse_args()

print(args) # for logging purposes

# defining domain for visualisation
x = jnp.expand_dims(jnp.arange(args.domain_cardinality) / args.domain_cardinality, axis = 1) # (m, 1)

# opens files
results = []
for file in args.results:
    results.append(jnp.load(file=file))

n_of_samples = jnp.concatenate([result["n_of_samples"] for result in results])
entropies    = jnp.concatenate([result["entropies"]    for result in results])

vis_result = results[args.visualisation_run]
vis_f_true = vis_result['f_true']
vis_observation_indices = vis_result['observation_indices']
vis_observation_values = vis_result['observation_values']
vis_post_means = vis_result['post_means']
vis_post_stds = vis_result['post_stds']
vis_etse_poo = vis_result['etse_poo']

fig, axs = plt.subplots(2, 1, figsize=(12, 8), dpi=400)

plt.subplots_adjust(top = 0.9, bottom=0.1, hspace=0.5, wspace=0.3)

twin_axs0 = axs[0].twinx()

# plot f_true
axs[0].plot(x[:, 0], vis_f_true, c="k", label=r"$f_{true}$")

# plot f_posterior as well as observations after ? samples from f_true
axs[0].scatter(x[vis_observation_indices, 0], vis_observation_values, c="k", label="observations")
for n in range(len(vis_observation_indices)):
    axs[0].annotate(n, (x[vis_observation_indices[n], 0], vis_observation_values[n]+0.2))
axs[0].plot(x[:, 0], vis_post_means, "b", label=r"$\mu_{F\ |\ \mathcal{D}}$")
axs[0].fill_between(x[:, 0], vis_post_means - vis_post_stds, vis_post_means + vis_post_stds, color='b', alpha=.1, label=r"$\sigma_{F\ |\ \mathcal{D}}$")

twin_axs0.plot(x[:, 0], vis_etse_poo, "r:", label=r"$\mathbb{P}[x \in X^* | \mathcal{D}]$")

# scatter plot the entropies with respect to the number of samples
axs[1].scatter(n_of_samples, entropies, label=r"$H[X^* | \mathcal{D}]$")

axs[0].set_xlabel("x")
leg0 = axs[0].legend(loc="upper left")
leg0.remove()

twin_axs0.legend(loc="upper right")
twin_axs0.add_artist(leg0)

axs[1].set_xlabel("# samples")
axs[1].legend(loc="upper left")
axs[1].set_xscale('log')

date_time = datetime.datetime.strftime(datetime.datetime.now(), '%d:%b:%Y-%H:%M:%S')
plt.savefig(f'results/combined_experiments-{date_time}.pdf', format='pdf') # figure

plt.show()

with open(f'results/combined_experiments-{date_time}.json', 'w') as fp:
    information = {'script name' : os.path.basename(__file__)}
    information.update(vars(args))
    json.dump(information, fp) # parameters