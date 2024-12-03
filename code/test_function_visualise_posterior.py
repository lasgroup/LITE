import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import json
import datetime
import argparse
import time
import os

import matplotlib.pyplot as plt
from matplotlib import cm
#import matplotlib.animation as animation

import src.test_functions as test_fs 

parser = argparse.ArgumentParser(description='visualises the posterior of a test function on a 2d domain', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("bayes_opt_report", type=str, help="npz file containing the posteriors as well as some additional information recorded during Bayesian optimisation as produced by 'test_function_posterior_during_sampling'")
parser.add_argument("poo_report", type=str, help="npz file containing the poo as produced by 'test_function_poo_during_sampling'")
parser.add_argument("visualisation_step", type=int, help="the bayesian optimisation step that is depicted")
args = parser.parse_args()

print(args) # for logging purposes

bayes_opt_report    = jax.numpy.load(file=args.bayes_opt_report)
seed                = bayes_opt_report['seed']
test_function       = bayes_opt_report['test_function']
#x                   = bayes_opt_report['x']
#f_true              = bayes_opt_report['f_true']
observation_indices = bayes_opt_report['observation_indices']
observation_values  = bayes_opt_report['observation_values']
post_means          = bayes_opt_report['post_means']
if "post_cov" in bayes_opt_report:
    post_cov          = bayes_opt_report['post_cov']
    post_stds         = jnp.sqrt(jnp.diagonal(bayes_opt_report['post_cov'], axis1=-2, axis2=-1))
else:
    post_stds         = bayes_opt_report['post_stds']

assert test_function in ["drop-wave", "drop-wave-mini"], "plotting is only enabled for test functions in [drop-wave]"

poo_report = jax.numpy.load(file=args.poo_report)
assert str(poo_report['bayes_opt_report']) == args.bayes_opt_report, "poo_report must match bayes_opt_report"
poos = poo_report['poos']

fig, ax = plt.subplots(1, 3, subplot_kw=dict(projection='3d'), figsize=(10, 7.5), dpi=400)

x, f_true, true_obs_noise_std, x1, x2 = test_fs.get_test_function(test_function, return_mesh=True)
sqrt_domain_cardinality = int(f_true.size ** 0.5)

# plot f_true
Z = f_true.reshape((sqrt_domain_cardinality, sqrt_domain_cardinality))
colors = cm.viridis(plt.Normalize(jnp.min(Z), jnp.max(Z))(Z))
surf1 = ax[0].plot_surface(x1, x2, Z, rcount = sqrt_domain_cardinality, ccount = sqrt_domain_cardinality, shade=False, facecolors=colors, label=r"$f_{true}$")
ax[0].legend(loc="upper left")
ax[0].set_xlabel(r"$x_1$")#, fontsize=35)
ax[0].set_ylabel(r"$x_2$")#, fontsize=35)


# plot f_posterior
Z = post_means[args.visualisation_step, :].reshape((sqrt_domain_cardinality, sqrt_domain_cardinality))
stds = post_stds[args.visualisation_step, :].reshape((sqrt_domain_cardinality, sqrt_domain_cardinality))
norm = plt.Normalize(jnp.min(stds), jnp.max(stds))
colors = cm.inferno(norm(stds))
surf2 = ax[1].plot_surface(x1, x2, Z, rcount = sqrt_domain_cardinality, ccount = sqrt_domain_cardinality, shade=False, facecolors=colors, label=r"$p(f\ |\ \mathcal{D})$")
cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.inferno), ax=ax[1], pad=0.3, fraction=0.025)
cbar.ax.set_title(r"$\sigma_{F\ |\ \mathcal{D}}$", rotation=0, fontsize=12)
ax[1].set_xlabel(r"$x_1$")#, fontsize=35)
#ax[1].xaxis.set_tick_params(labelsize=25)
#ax[1].xaxis.labelpad = 15
ax[1].set_ylabel(r"$x_2$")#, fontsize=35)
#ax[1].yaxis.set_tick_params(labelsize=25)
#ax[1].yaxis.labelpad = 15
#ax[1].zaxis.set_tick_params(pad=10, labelsize=25)
ax[1].set_zlabel(r"$\mu_{F\ |\ \mathcal{D}}$")
ax[1].legend(loc="upper left")#, bbox_to_anchor=(0.05, 0.9), fontsize=40)

# plot poo
Z = poos[args.visualisation_step, :].reshape((sqrt_domain_cardinality, sqrt_domain_cardinality))
colors = cm.viridis(plt.Normalize(jnp.min(Z), jnp.max(Z))(Z))
surf3 = ax[2].plot_surface(x1, x2, Z, rcount = sqrt_domain_cardinality, ccount = sqrt_domain_cardinality, shade=False, facecolors=colors, label=r"$p(x^* |\ \mathcal{D})$")
ax[2].set_xlabel(r"$x_1$")#, fontsize=35)
#ax[1].xaxis.set_tick_params(labelsize=25)
#ax[1].xaxis.labelpad = 15
ax[2].set_ylabel(r"$x_2$")#, fontsize=35)
#ax[1].yaxis.set_tick_params(labelsize=25)
#ax[1].yaxis.labelpad = 15
#ax[1].zaxis.set_tick_params(pad=10, labelsize=25)
ax[2].legend(loc="upper left")#, bbox_to_anchor=(0.05, 0.9), fontsize=40)

#def update(frame):
#    # for each frame, update the data stored on each artist.
#    global surf2
#    surf2.remove()
#    global cbar
#    Z = post_means[frame, :].reshape((sqrt_domain_cardinality, sqrt_domain_cardinality))
#    stds = post_stds[frame, :].reshape((sqrt_domain_cardinality, sqrt_domain_cardinality))
#    norm = plt.Normalize(jnp.min(stds), jnp.max(stds))
#    colors = cm.inferno(norm(stds))
#    surf2 = ax[1].plot_surface(x1, x2, Z, rcount = sqrt_domain_cardinality, ccount = sqrt_domain_cardinality, shade=False, facecolors=colors, label=r"$p(f\ |\ \mathcal{D})$")
#    cbar.remove()
#    cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.inferno), ax=ax[1], pad=0.5, fraction=0.1)
#    cbar.ax.set_title(r"$\sigma_{F\ |\ \mathcal{D}}$", rotation=0, fontsize=12)

#ani = animation.FuncAnimation(fig=fig, func=update, interval=10000)

date_time = datetime.datetime.strftime(datetime.datetime.now(), '%d:%b:%Y-%H:%M:%S')
plt.savefig(f'results/{date_time}.pdf', format='pdf')
with open(f'results/{date_time}.json', 'w') as fp:
    information = {'script name' : os.path.basename(__file__)}
    information.update(vars(args))
    json.dump(information, fp)

plt.show()