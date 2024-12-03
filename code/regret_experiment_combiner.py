import jax
#jax.config.update("jax_enable_x64", True)
from jax import random
import jax.numpy as jnp
import matplotlib.pyplot as plt
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.size'] = 16
from matplotlib import cm
import PIL
import json
import datetime
import argparse
import time
import os

parser = argparse.ArgumentParser(description='combines the results of several regret experiments into a single set of plots', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("results", type=str, nargs="+", help="npz files of results of regret experiments")
parser.add_argument("--std_factor", type=float, default=10.0**.5, help="the factor by which standard deviations are downscaled for visualisation")
args = parser.parse_args()
print(args) # for logging purposes
s_fac = args.std_factor

# opens files
results = []
for file in args.results:
    results.append(jax.numpy.load(file=file))

# prepares plots
fig, axs = plt.subplots(4, 1, figsize=(12, 26), dpi=400)
plt.subplots_adjust(top = 0.9, bottom=0.1, hspace=0.5, wspace=0.3)

# plot statistics for cumulative regret
for result in results:
    axs[0].plot(range(len(result['cumulative_regret_mean'])), result['cumulative_regret_mean'], label=rf"$\mu_{{cum\, reg}} \pm \sigma_{{cum\, reg}} / {s_fac} \ ({result['method']})$")
    axs[0].fill_between(range(len(result['cumulative_regret_mean'])), result['cumulative_regret_mean'] - result['cumulative_regret_std']/s_fac, result['cumulative_regret_mean'] + result['cumulative_regret_std']/s_fac, alpha=.1)

# plot statistics for simple regret
for result in results:
    axs[1].plot(range(len(result['simple_regret_mean'])), result['simple_regret_mean'], label=rf"$\mu_{{sim\, reg}} \pm \sigma_{{sim\, reg}} / {s_fac} \ ({result['method']})$")
    axs[1].fill_between(range(len(result['simple_regret_mean'])), result['simple_regret_mean'] - result['simple_regret_std']/s_fac, result['simple_regret_mean'] + result['simple_regret_std']/s_fac, alpha=.1)

# plot statistics for entropy
for result in results:
    axs[2].plot(range(len(result['entropies_mean'])), result['entropies_mean'], label=rf"$\mu_{{H[X^*|\mathcal{{D}}]}} \pm \sigma_{{H[X^*|\mathcal{{D}}]}} / {s_fac} \ ({result['method']})$")
    axs[2].fill_between(range(len(result['entropies_mean'])), result['entropies_mean'] - result['entropies_std']/s_fac, result['entropies_mean'] + result['entropies_std']/s_fac, alpha=.1)

# plot statistcs for difference between kappa and max f_true for the runs that use a kappa
plotted_zero_line = False
for result in results:
    if not "kappa_delta_mean" in result:
        continue
    if not plotted_zero_line:
        axs[3].plot(range(1, len(result['kappa_delta_mean'])), jnp.zeros((len(result['kappa_delta_mean']) - 1,)), 'k-.')
        plotted_zero_line = True
    axs[3].plot(range(1, len(result['kappa_delta_mean'])), result['kappa_delta_mean'][1:], label=rf"$\mu_{{\kappa - f_{{true}}^*}} \pm \sigma_{{\kappa - f_{{true}}^*}} / {s_fac} \ ({result['method']})$")
    axs[3].fill_between(range(1, len(result['kappa_delta_mean'])), result['kappa_delta_mean'][1:] - result['kappa_delta_std'][1:]/s_fac, result['kappa_delta_mean'][1:] + result['kappa_delta_std'][1:]/s_fac, alpha=.1)

# close files
for result in results:
    result.close()

# adjust style
axs[0].legend(loc="upper left", fontsize=14)
axs[0].set_xlabel("t", fontsize=18)
#axs[0].set_xscale("log")
axs[0].set_ylabel("cumulative regret")
axs[0].tick_params(which='major', length=7, width=2)
axs[0].tick_params(which='minor', length=4, width=1.5)
axs[1].legend(loc="lower left", fontsize=14)
axs[1].set_xlabel("t", fontsize=18)
#axs[1].set_xscale("log")
axs[1].set_ylabel("simple regret")
axs[1].set_yscale("log")
axs[1].tick_params(which='major', length=7, width=2)
axs[1].tick_params(which='minor', length=4, width=1.5)
axs[2].legend(loc="lower left", fontsize=14)
axs[2].set_xlabel("t", fontsize=18)
#axs[2].set_xscale("log")
axs[2].set_ylabel("PoM Entropy")
#axs[2].set_yscale("log")
axs[2].tick_params(which='major', length=7, width=2)
axs[2].tick_params(which='minor', length=4, width=1.5)
axs[3].legend(loc="upper right", fontsize=14)
axs[3].set_xlabel("t", fontsize=18)
axs[3].set_ylabel(r"$\kappa - f_{true}^*$")
axs[3].set_yscale("symlog", linthresh=1e-2)
axs[3].set_yticks([1e-1 * i for i in range(-9, 10) if abs(i) % 2 == 0] + [1e-2 * i for i in range(-9, 10) if abs(i) % 2 == 0] + [1e-3 * i for i in range(-9, 10) if abs(i) % 2 == 0], minor=True)
axs[3].tick_params(which='major', length=7, width=2)
axs[3].tick_params(which='minor', length=4, width=1.5)




date_time = datetime.datetime.strftime(datetime.datetime.now(), '%d:%b:%Y-%H:%M:%S')
plt.savefig(f'results/combined_experiments-{date_time}.pdf', format='pdf') # figure

with open(f'results/combined_experiments-{date_time}.json', 'w') as fp:
    information = {'script name' : os.path.basename(__file__)}
    information.update(vars(args))
    json.dump(information, fp) # parameters

plt.show()
