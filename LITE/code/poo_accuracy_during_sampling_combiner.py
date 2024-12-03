import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib import cm
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.size'] = 22
import json
import datetime
import argparse
import time
import os

parser = argparse.ArgumentParser(description='combines the results of several "POO accuracy during sampling" experiments into a single set of plots', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("results", type=str, nargs="+", help="npz files of results from experiments")
parser.add_argument('-c', '--domain_cardinality', type=int, default=300, help='the cardinality of the domain, i.e. the resolution of the subsampling of a continuous Gaussian process')
parser.add_argument('--two_d_domain', action="store_true", help='whether the (continuous) domain is a square instead of a line interval')
parser.add_argument("-vr", "--visualisation_run", type=int, default=0, help="the index of the run that is used for the illustration of the various POOs")
args = parser.parse_args()

print(args) # for logging purposes

# defining domain
if args.two_d_domain:
    sqrt_domain_cardinality = int(args.domain_cardinality**.5)
    assert args.domain_cardinality**.5 == sqrt_domain_cardinality, "if the flag --two_d_domain is set --domain_cardinality must be a square number."
    x1, x2 = jnp.meshgrid(jnp.arange(sqrt_domain_cardinality) / sqrt_domain_cardinality,
                           jnp.arange(sqrt_domain_cardinality) / sqrt_domain_cardinality) # (sqrt(m), sqrt(m)), essentially already broad-cast from (sqrt(m), 1) and (1, sqrt(m)), respectively
    x = jnp.reshape(jnp.dstack((x1, x2)), (args.domain_cardinality, 2)) # (m, 2)
else:
    x = jnp.expand_dims(jnp.arange(args.domain_cardinality) / args.domain_cardinality, axis = 1) # (m, 1)

# opens files
results = []
for file in args.results:
    results.append(jnp.load(file=file))

poo_entropy_etse_mean = sum(result["poo_entropy_etse"] for result in results) / len(results)
poo_entropy_etse_std = ( sum((result["poo_entropy_etse"] - poo_entropy_etse_mean)**2 for result in results) / (len(results)-1) )**.5

ie_sinkhorn_mean = sum(result["ie_sinkhorn"] for result in results) / len(results)
ie_sinkhorn_std = ( sum((result["ie_sinkhorn"] - ie_sinkhorn_mean)**2 for result in results) / (len(results)-1) )**.5
ie_tv_mean = sum(result["ie_tv"] for result in results) / len(results)
ie_tv_std = ( sum((result["ie_tv"] - ie_tv_mean)**2 for result in results) / (len(results)-1) )**.5
poo_entropy_ie_mean = sum(result["poo_entropy_ie"] for result in results) / len(results)
poo_entropy_ie_std = ( sum((result["poo_entropy_ie"] - poo_entropy_ie_mean)**2 for result in results) / (len(results)-1) )**.5
poo_entropy_ie_msre = sum( (result["poo_entropy_ie"]  - result["poo_entropy_etse"])**2 / result["poo_entropy_etse"]**2  for result in results) / len(results)  # mean squared relative error
poo_entropy_ie_stdsre = ( sum( (((result["poo_entropy_ie"]  - result["poo_entropy_etse"])**2 / result["poo_entropy_etse"]**2) - poo_entropy_ie_msre)**2  for result in results) / (len(results)-1) )**.5  # standard deviation of squared relative error

cme_sinkhorn_mean = sum(result["cme_sinkhorn"] for result in results) / len(results)
cme_sinkhorn_std = ( sum((result["cme_sinkhorn"] - cme_sinkhorn_mean)**2 for result in results) / (len(results)-1) )**.5
cme_tv_mean = sum(result["cme_tv"] for result in results) / len(results)
cme_tv_std = ( sum((result["cme_tv"] - cme_tv_mean)**2 for result in results) / (len(results)-1) )**.5
poo_entropy_cme_mean = sum(result["poo_entropy_cme"] for result in results) / len(results)
poo_entropy_cme_std = ( sum((result["poo_entropy_cme"] - poo_entropy_cme_mean)**2 for result in results) / (len(results)-1) )**.5
poo_entropy_cme_msre = sum( (result["poo_entropy_cme"]  - result["poo_entropy_etse"])**2 / result["poo_entropy_etse"] **2  for result in results) / len(results)  # mean squared relative error
poo_entropy_cme_stdsre = ( sum( (((result["poo_entropy_cme"]  - result["poo_entropy_etse"])**2 / result["poo_entropy_etse"]**2) - poo_entropy_cme_msre)**2  for result in results) / (len(results)-1) )**.5  # standard deviation of squared relative error

#ocme_sinkhorn_mean = sum(result["ocme_sinkhorn"] for result in results) / len(results)
#ocme_sinkhorn_std = ( sum((result["ocme_sinkhorn"] - ocme_sinkhorn_mean)**2 for result in results) / (len(results)-1) )**.5
#ocme_tv_mean = sum(result["ocme_tv"] for result in results) / len(results)
#ocme_tv_std = ( sum((result["ocme_tv"] - ocme_tv_mean)**2 for result in results) / (len(results)-1) )**.5
#poo_entropy_ocme_mean = sum(result["poo_entropy_ocme"] for result in results) / len(results)
#poo_entropy_ocme_std = ( sum((result["poo_entropy_ocme"] - poo_entropy_ocme_mean)**2 for result in results) / (len(results)-1) )**.5
#poo_entropy_ocme_msre = sum( (result["poo_entropy_ocme"]  - result["poo_entropy_etse"])**2 / result["poo_entropy_etse"] **2  for result in results) / len(results)  # mean squared relative error
#poo_entropy_ocme_stdsre = ( sum( (((result["poo_entropy_ocme"]  - result["poo_entropy_etse"])**2 / result["poo_entropy_etse"]**2) - poo_entropy_ocme_msre)**2  for result in results) / (len(results)-1) )**.5  # standard deviation of squared relative error

vapor_sinkhorn_mean = sum(result["vapor_sinkhorn"] for result in results) / len(results)
vapor_sinkhorn_std = ( sum((result["vapor_sinkhorn"] - vapor_sinkhorn_mean)**2 for result in results) / (len(results)-1) )**.5
vapor_tv_mean = sum(result["vapor_tv"] for result in results) / len(results)
vapor_tv_std = ( sum((result["vapor_tv"] - vapor_tv_mean)**2 for result in results) / (len(results)-1) )**.5
poo_entropy_vapor_mean = sum(result["poo_entropy_vapor"] for result in results) / len(results)
poo_entropy_vapor_std = ( sum((result["poo_entropy_vapor"] - poo_entropy_vapor_mean)**2 for result in results) / (len(results)-1) )**.5
poo_entropy_vapor_msre = sum( (result["poo_entropy_vapor"]  - result["poo_entropy_etse"])**2 / result["poo_entropy_etse"] **2  for result in results) / len(results)  # mean squared relative error
poo_entropy_vapor_stdsre = ( sum( (((result["poo_entropy_vapor"]  - result["poo_entropy_etse"])**2 / result["poo_entropy_etse"]**2) - poo_entropy_vapor_msre)**2  for result in results) / (len(results)-1) )**.5  # standard deviation of squared relative error

nest_sinkhorn_mean = sum(result["nest_sinkhorn"] for result in results) / len(results)
nest_sinkhorn_std = ( sum((result["nest_sinkhorn"] - nest_sinkhorn_mean)**2 for result in results) / (len(results)-1) )**.5
nest_tv_mean = sum(result["nest_tv"] for result in results) / len(results)
nest_tv_std = ( sum((result["nest_tv"] - nest_tv_mean)**2 for result in results) / (len(results)-1) )**.5
poo_entropy_nest_mean = sum(result["poo_entropy_nest"] for result in results) / len(results)
poo_entropy_nest_std = ( sum((result["poo_entropy_nest"] - poo_entropy_nest_mean)**2 for result in results) / (len(results)-1) )**.5
poo_entropy_nest_msre = sum( (result["poo_entropy_nest"]  - result["poo_entropy_etse"])**2 / result["poo_entropy_etse"] **2  for result in results) / len(results)  # mean squared relative error
poo_entropy_nest_stdsre = ( sum( (((result["poo_entropy_nest"]  - result["poo_entropy_etse"])**2 / result["poo_entropy_etse"]**2) - poo_entropy_nest_msre)**2  for result in results) / (len(results)-1) )**.5  # standard deviation of squared relative error

nie_sinkhorn_mean = sum(result["nie_sinkhorn"] for result in results) / len(results)
nie_sinkhorn_std = ( sum((result["nie_sinkhorn"] - nie_sinkhorn_mean)**2 for result in results) / (len(results)-1) )**.5
nie_tv_mean = sum(result["nie_tv"] for result in results) / len(results)
nie_tv_std = ( sum((result["nie_tv"] - nie_tv_mean)**2 for result in results) / (len(results)-1) )**.5
poo_entropy_nie_mean = sum(result["poo_entropy_nie"] for result in results) / len(results)
poo_entropy_nie_std = ( sum((result["poo_entropy_nie"] - poo_entropy_nie_mean)**2 for result in results) / (len(results)-1) )**.5
poo_entropy_nie_msre = sum( (result["poo_entropy_nie"]  - result["poo_entropy_etse"])**2 / result["poo_entropy_etse"] **2  for result in results) / len(results)  # mean squared relative error
poo_entropy_nie_stdsre = ( sum( (((result["poo_entropy_nie"]  - result["poo_entropy_etse"])**2 / result["poo_entropy_etse"]**2) - poo_entropy_nie_msre)**2  for result in results) / (len(results)-1) )**.5  # standard deviation of squared relative error

vis_result = results[args.visualisation_run]
vis_f_true = vis_result['vis_f_true']
vis_observation_indices = vis_result['vis_observation_indices']
vis_observation_values = vis_result['vis_observation_values']
vis_post_means = vis_result['vis_post_means']
vis_post_stds = vis_result['vis_post_stds']
vis_acquisition_function = vis_result['vis_acquisition_function']

vis_etse = vis_result['vis_poo_etse']
vis_ie = vis_result['vis_poo_ie']
vis_cme = vis_result['vis_poo_cme']
vis_vapor = vis_result['vis_poo_vapor']
vis_nest = vis_result['vis_poo_nest']
vis_nie = vis_result['vis_poo_nie']

poo_entropy_nest_fraction_of_under_estimations  = sum( jnp.count_nonzero(result["poo_entropy_nest"]   < result["poo_entropy_etse"])/result["poo_entropy_etse"].size for result in results) / len(results)
poo_entropy_vapor_fraction_of_under_estimations = sum( jnp.count_nonzero(result["poo_entropy_vapor"]  < result["poo_entropy_etse"])/result["poo_entropy_etse"].size for result in results) / len(results)
poo_entropy_cme_fraction_of_under_estimations   = sum( jnp.count_nonzero(result["poo_entropy_cme"]    < result["poo_entropy_etse"])/result["poo_entropy_etse"].size for result in results) / len(results)
poo_entropy_nie_fraction_of_under_estimations   = sum( jnp.count_nonzero(result["poo_entropy_nie"]    < result["poo_entropy_etse"])/result["poo_entropy_etse"].size for result in results) / len(results)
poo_entropy_ie_fraction_of_under_estimations    = sum( jnp.count_nonzero(result["poo_entropy_ie"]     < result["poo_entropy_etse"])/result["poo_entropy_etse"].size for result in results) / len(results)

print(f"percentage of entropy under estimations using NEST:  {poo_entropy_nest_fraction_of_under_estimations}")
print(f"percentage of entropy under estimations using VAPOR: {poo_entropy_vapor_fraction_of_under_estimations}")
print(f"percentage of entropy under estimations using CME:   {poo_entropy_cme_fraction_of_under_estimations}")
print(f"percentage of entropy under estimations using NIE:   {poo_entropy_nie_fraction_of_under_estimations}")
print(f"percentage of entropy under estimations using IE:    {poo_entropy_ie_fraction_of_under_estimations}")

# factor to correct standard deviation to standard error
s_fac = len(results)**.5

if not args.two_d_domain:
    fig, axs = plt.subplots(5, 1, figsize=(12, 40), dpi=400)
    twin_axs3 = axs[3].twinx()
else:
    fig, axs = plt.subplots(7, 1, figsize=(12, 100), dpi=400, gridspec_kw={'height_ratios': [1, 1, 1, 3, 3, 3, 3]}) 
plt.subplots_adjust(top = 0.9, bottom=0.1, hspace=0.5, wspace=0.3)

# plot TV distances
axs[0].plot(range(len(nest_tv_mean)), nest_tv_mean, 'g--', label=rf"$NEST$")
axs[0].fill_between(range(len(nest_tv_mean)), nest_tv_mean - nest_tv_std/s_fac, nest_tv_mean + nest_tv_std/s_fac, color='g', linestyle='--', linewidth=2, alpha=.1)
axs[0].plot(range(len(vapor_tv_mean)), vapor_tv_mean, 'r:', label=rf"$VAPOR$")
axs[0].fill_between(range(len(vapor_tv_mean)), vapor_tv_mean - vapor_tv_std/s_fac, vapor_tv_mean + vapor_tv_std/s_fac, color='r', linestyle=':', linewidth=2, alpha=.1)
axs[0].plot(range(len(cme_tv_mean)), cme_tv_mean, 'b--', label=rf"$CME$")
axs[0].fill_between(range(len(cme_tv_mean)), cme_tv_mean - cme_tv_std/s_fac, cme_tv_mean + cme_tv_std/s_fac, color='b', linestyle='--', linewidth=2, alpha=.1)
#axs[0].plot(range(len(ocme_tv_mean)), ocme_tv_mean, 'm-.', label=rf"OCME$")
#axs[0].fill_between(range(len(ocme_tv_mean)), ocme_tv_mean - ocme_tv_std/s_fac, ocme_tv_mean + ocme_tv_std/s_fac, color='m', linestyle='-.', linewidth=2, alpha=.1)
axs[0].plot(range(len(nie_tv_mean)), nie_tv_mean, 'm-.', label=rf"$NIE$")
axs[0].fill_between(range(len(nie_tv_mean)), nie_tv_mean - nie_tv_std/s_fac, nie_tv_mean + nie_tv_std/s_fac, color='k', linestyle='-', linewidth=2, alpha=.1)
axs[0].plot(range(len(ie_tv_mean)), ie_tv_mean, 'k-', label=rf"$IE$")
axs[0].fill_between(range(len(ie_tv_mean)), ie_tv_mean - ie_tv_std/s_fac, ie_tv_mean + ie_tv_std/s_fac, color='k', linestyle='-', linewidth=2, alpha=.1)

# plot Sinkhorn divergences
axs[1].plot(range(len(nest_sinkhorn_mean)), nest_sinkhorn_mean, 'g--', label=rf"$NEST$")
axs[1].fill_between(range(len(nest_sinkhorn_mean)), nest_sinkhorn_mean - nest_sinkhorn_std/s_fac, nest_sinkhorn_mean + nest_sinkhorn_std/s_fac, color='g', linestyle='--', linewidth=2, alpha=.1)
axs[1].plot(range(len(vapor_sinkhorn_mean)), vapor_sinkhorn_mean, 'r:', label=rf"$VAPOR$")
axs[1].fill_between(range(len(vapor_sinkhorn_mean)), vapor_sinkhorn_mean - vapor_sinkhorn_std/s_fac, vapor_sinkhorn_mean + vapor_sinkhorn_std/s_fac, color='r', linestyle=':', linewidth=2, alpha=.1)
axs[1].plot(range(len(cme_sinkhorn_mean)), cme_sinkhorn_mean, 'b--', label=rf"$CME$")
axs[1].fill_between(range(len(cme_sinkhorn_mean)), cme_sinkhorn_mean - cme_sinkhorn_std/s_fac, cme_sinkhorn_mean + cme_sinkhorn_std/s_fac, color='b', linestyle='--', linewidth=2, alpha=.1)
#axs[1].plot(range(len(ocme_sinkhorn_mean)), ocme_sinkhorn_mean, 'm-.', label=rf"$OCME$")
#axs[1].fill_between(range(len(ocme_sinkhorn_mean)), ocme_sinkhorn_mean - ocme_sinkhorn_std/s_fac, ocme_sinkhorn_mean + ocme_sinkhorn_std/s_fac, color='m', linestyle='-.', linewidth=2, alpha=.1)
axs[1].plot(range(len(nie_sinkhorn_mean)), nie_sinkhorn_mean, 'm-.', label=rf"$NIE$")
axs[1].fill_between(range(len(nie_sinkhorn_mean)), nie_sinkhorn_mean - nie_sinkhorn_std/s_fac, nie_sinkhorn_mean + nie_sinkhorn_std/s_fac, color='m', linestyle='-.', linewidth=2, alpha=.1)
axs[1].plot(range(len(ie_sinkhorn_mean)), ie_sinkhorn_mean, 'k-', label=rf"$IE$")
axs[1].fill_between(range(len(ie_sinkhorn_mean)), ie_sinkhorn_mean - ie_sinkhorn_std/s_fac, ie_sinkhorn_mean + ie_sinkhorn_std/s_fac, color='k', linestyle='-', linewidth=2, alpha=.1)

# plot entropy root mean squared relative error
axs[2].plot(range(len(poo_entropy_nest_msre)),  jnp.sqrt(poo_entropy_nest_msre),  'g--', label=rf"$NEST$")
axs[2].plot(range(len(poo_entropy_vapor_msre)), jnp.sqrt(poo_entropy_vapor_msre), 'r:', label=rf"$VAPOR$")
axs[2].plot(range(len(poo_entropy_cme_msre)),   jnp.sqrt(poo_entropy_cme_msre),   'b--', label=rf"$CME$")
#axs[2].plot(range(len(poo_entropy_ocme_msre)), jnp.sqrt(poo_entropy_ocme_msre),  'm-.', label=rf"$OCME$")
axs[2].plot(range(len(poo_entropy_nie_msre)),   jnp.sqrt(poo_entropy_nie_msre),   'm-.', label=rf"$NIE$")
axs[2].plot(range(len(poo_entropy_ie_msre)),    jnp.sqrt(poo_entropy_ie_msre),    'k-', label=rf"$IE$")

axs[2].fill_between(range(len(poo_entropy_nest_msre)),  jnp.sqrt(poo_entropy_nest_msre  - poo_entropy_nest_stdsre/s_fac ), jnp.sqrt(poo_entropy_nest_msre  + poo_entropy_nest_stdsre/s_fac ), color='g', linestyle='--', linewidth=2, alpha=.1)
axs[2].fill_between(range(len(poo_entropy_vapor_msre)), jnp.sqrt(poo_entropy_vapor_msre - poo_entropy_vapor_stdsre/s_fac), jnp.sqrt(poo_entropy_vapor_msre + poo_entropy_vapor_stdsre/s_fac), color='r', linestyle=':', linewidth=2, alpha=.1)
axs[2].fill_between(range(len(poo_entropy_cme_msre)),   jnp.sqrt(poo_entropy_cme_msre   - poo_entropy_cme_stdsre/s_fac  ), jnp.sqrt(poo_entropy_cme_msre   + poo_entropy_cme_stdsre/s_fac  ), color='b', linestyle='--', linewidth=2, alpha=.1)
#axs[2].fill_between(range(len(poo_entropy_ocme_msre)), jnp.sqrt(poo_entropy_ocme_msre  - poo_entropy_ocme_stdsre/s_fac ), jnp.sqrt(poo_entropy_ocme_msre  + poo_entropy_ocme_stdsre/s_fac ), color='m', linestyle='-.', linewidth=2, alpha=.1)
axs[2].fill_between(range(len(poo_entropy_nie_msre)),   jnp.sqrt(poo_entropy_nie_msre   - poo_entropy_nie_stdsre/s_fac  ), jnp.sqrt(poo_entropy_nie_msre   + poo_entropy_nie_stdsre/s_fac  ), color='k', linestyle='-', linewidth=2, alpha=.1)
axs[2].fill_between(range(len(poo_entropy_ie_msre)),    jnp.sqrt(poo_entropy_ie_msre    - poo_entropy_ie_stdsre/s_fac   ), jnp.sqrt(poo_entropy_ie_msre    + poo_entropy_ie_stdsre/s_fac   ), color='k', linestyle='-', linewidth=2, alpha=.1)

if not args.two_d_domain: #1d plotting

    # plot f_true
    axs[3].plot(x[:, 0], vis_f_true, c="k", label=r"$f_{true}$")

    # plot f_posterior as well as observations after ? samples from f_true
    axs[3].scatter(x[vis_observation_indices, 0], vis_observation_values, c="k", label="observations")
    for n in range(len(vis_observation_indices)):
        axs[3].annotate(n, (x[vis_observation_indices[n], 0], vis_observation_values[n]+0.2))
    axs[3].plot(x[:, 0], vis_post_means, "b", label=r"$\mu_{F\ |\ \mathcal{D}}$")
    axs[3].fill_between(x[:, 0], vis_post_means - vis_post_stds, vis_post_means + vis_post_stds, color='b', alpha=.1, label=r"$\sigma_{F\ |\ \mathcal{D}}$")
    twin_axs3.plot(x[:, 0], vis_acquisition_function, "r:", label=r"$\alpha_{TS}(x;\mathcal{D})$")

    # plot POOs after ? samples from f_true
    axs[4].plot(x[:, 0], vis_nest, 'g--', label=r"$NEST$")
    axs[4].plot(x[:, 0], vis_vapor, 'r:', label=r"$VAPOR$")
    axs[4].plot(x[:, 0], vis_cme, 'b--', label=r"$CME$")
    #axs[4].plot(x[:, 0], vis_ocme, 'm-.', label=r"$OCME$")
    axs[4].plot(x[:, 0], vis_nie, 'm-.', label=r"$NIE$")
    axs[4].plot(x[:, 0], vis_ie, 'c-', label=r"$IE$")
    axs[4].plot(x[:, 0], vis_etse, 'k-', label=r"$E-TSE$")
else:
    axs[3].remove()
    axs[3] = fig.add_subplot(7, 1, 4, projection='3d')
    axs[4].remove()
    axs[4] = fig.add_subplot(7, 1, 5, projection='3d')
    axs[5].remove()
    axs[5] = fig.add_subplot(7, 1, 6, projection='3d')
    axs[6].remove()
    axs[6] = fig.add_subplot(7, 1, 7, projection='3d')

    # plot f_posterior as well as observations
    axs[3].scatter(x[vis_observation_indices, 0], x[vis_observation_indices, 1], vis_observation_values, c="k", label="observations")
    Z = vis_post_means.reshape((sqrt_domain_cardinality, sqrt_domain_cardinality))
    stds = vis_post_stds.reshape((sqrt_domain_cardinality, sqrt_domain_cardinality))
    norm = plt.Normalize(jnp.min(stds), jnp.max(stds))
    colors = cm.inferno(norm(stds))
    surf = axs[3].plot_surface(x1, x2, Z, rcount = sqrt_domain_cardinality, ccount = sqrt_domain_cardinality, shade=True, facecolors=colors, label=r"$\mu_{F\ |\ \mathcal{D}}$")
    cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.inferno), ax=axs[3], pad=0.1, fraction=0.025)
    cbar.ax.set_title(r"$\sigma_{F\ |\ \mathcal{D}}$", rotation=0)

    # plot true POO after ? samples from f_true
    Z = vis_etse.reshape((sqrt_domain_cardinality, sqrt_domain_cardinality))
    #colors = cm.viridis(plt.Normalize(jnp.min(Z), jnp.max(Z))(Z))
    surf = axs[4].plot_surface(x1, x2, Z, rcount = sqrt_domain_cardinality, ccount = sqrt_domain_cardinality, shade=True, edgecolor="w", label=r"$E-TSE$")
    #surf.set_edgecolor((0,0,0,0.2))

    # plot NIE POO after ? samples from f_true
    Z = vis_nie.reshape((sqrt_domain_cardinality, sqrt_domain_cardinality))
    #colors = cm.viridis(plt.Normalize(jnp.min(Z), jnp.max(Z))(Z))
    surf = axs[5].plot_surface(x1, x2, Z, rcount = sqrt_domain_cardinality, ccount = sqrt_domain_cardinality, shade=True, edgecolor="w", label=r"$NIE$")
    #surf.set_edgecolor((0,0,0,0.2))

    # plot f_true
    Z = vis_f_true.reshape((sqrt_domain_cardinality, sqrt_domain_cardinality))
    colors = cm.viridis(plt.Normalize(jnp.min(Z), jnp.max(Z))(Z))
    surf = axs[6].plot_surface(x1, x2, Z, rcount = sqrt_domain_cardinality, ccount = sqrt_domain_cardinality, shade=True, facecolors=colors, label=r"$f_{true}$")
    #surf.set_facecolor((0,0,0,0.5))

axs[0].set_xlabel("t")
axs[0].set_ylabel(r"$d_{TV}(E-TSE,\, \cdot\, )$")
axs[0].legend(loc="upper left")

axs[1].set_xlabel("t")
axs[1].set_ylabel(r"$S^{\varepsilon}(E-TSE,\, \cdot\, )$")
axs[1].legend(loc="upper left")

axs[2].set_xlabel("t")
axs[2].set_ylabel(r"RMSRE of $H[X^* | \mathcal{D}]$")
axs[2].legend(loc="upper left")

leg3 = axs[3].legend(loc="upper left", fontsize=40)
if args.two_d_domain:
    leg3.set_bbox_to_anchor((0.05, 0.9))
    axs[3].set_xlabel(r"$x_1$", fontsize=35)
    axs[3].xaxis.set_tick_params(labelsize=25)
    axs[3].xaxis.labelpad = 15
    axs[3].set_ylabel(r"$x_2$", fontsize=35)
    axs[3].yaxis.set_tick_params(labelsize=25)
    axs[3].yaxis.labelpad = 15
    axs[3].zaxis.set_tick_params(pad=10, labelsize=25)
else:
    axs[3].set_xlabel("x")
    leg3.remove()
    twin_axs3.legend(loc="upper right")
    twin_axs3.add_artist(leg3)

if args.two_d_domain:
    axs[4].set_xlabel(r"$x_1$", fontsize=35)
    axs[4].xaxis.set_tick_params(labelsize=25)
    axs[4].xaxis.labelpad = 15
    axs[4].set_ylabel(r"$x_2$", fontsize=35)
    axs[4].yaxis.set_tick_params(labelsize=25)
    axs[4].yaxis.labelpad = 15
    axs[4].zaxis.set_tick_params(pad=15, labelsize=25)
    axs[4].legend(loc="upper left", bbox_to_anchor=(0.05, 0.9), fontsize=40)

else:
    axs[4].set_xlabel("x")
#axs[4].legend(loc="upper left")

if args.two_d_domain:
    axs[5].set_xlabel(r"$x_1$", fontsize=35)
    axs[5].xaxis.set_tick_params(labelsize=25)

    axs[5].xaxis.labelpad = 15
    axs[5].set_ylabel(r"$x_2$", fontsize=35)
    axs[5].yaxis.set_tick_params(labelsize=25)
    axs[5].yaxis.labelpad = 15
    axs[5].zaxis.set_tick_params(pad=15, labelsize=25)
    axs[5].legend(loc="upper left", bbox_to_anchor=(0.05, 0.9), fontsize=40)

    axs[6].set_xlabel(r"$x_1$", fontsize=35)
    axs[6].xaxis.set_tick_params(labelsize=25)
    axs[6].xaxis.labelpad = 15
    axs[6].set_ylabel(r"$x_2$", fontsize=35)
    axs[6].yaxis.set_tick_params(labelsize=25)
    axs[6].yaxis.labelpad = 15
    axs[6].zaxis.set_tick_params(pad=10, labelsize=25)
    axs[6].legend(loc="upper left", bbox_to_anchor=(0.05, 0.9), fontsize=40)

date_time = datetime.datetime.strftime(datetime.datetime.now(), '%d:%b:%Y-%H:%M:%S')
plt.savefig(f'results/combined_experiments-{date_time}.pdf', format='pdf') # figure

plt.show()

with open(f'results/combined_experiments-{date_time}.json', 'w') as fp:
    information = {'script name' : os.path.basename(__file__)}
    information.update(vars(args))
    json.dump(information, fp) # parameters