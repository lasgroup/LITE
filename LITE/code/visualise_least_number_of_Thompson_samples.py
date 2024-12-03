import wandb
import jax
#jax.config.update("jax_enable_x64", True)
from jax import random
import jax.numpy as jnp
import matplotlib.pyplot as plt
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
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

parser = argparse.ArgumentParser(description='computes the probability of optimality according to E-TSE (and CME for additional comparison). Meant to be called for different --thompson_sampling_n_exponent and --alpha to compare the fidelity of the estimation', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-s', '--seed', type=int, default=0, help='the random seed for reproducibility')
parser.add_argument('-t', '--timeit', action="store_true", help='whether to print the elapsed time for E-TSE and CME')
parser.add_argument('-c', '--domain_cardinality', type=int, default=225, help='the cardinality of the domain, i.e. the resolution of the subsampling of a continuous Gaussian process')
parser.add_argument('--two_d_domain', action="store_true", help='whether the (continuous) domain is a square instead of a line interval')
parser.add_argument('-k', '--kernel', type=str, default="gaussian", help="which kernel to use, must be one of 'gaussian', 'laplacian', 'linear', or 'independent'")
parser.add_argument('-l', '--length_scale', type=float, default=0.02, help='the length scale of the kernel for the prior')
parser.add_argument('-a', '--prior_amplitude', type=float, default=2, help='the amplitude of the prior belief (true function is sampled from copied prior with fixed amplitude 1)')
parser.add_argument('-o', '--number_of_observation_points', type=int, default=10, help='the number of equally spaced observation points to condition on')
parser.add_argument('-n', '--observation_noise_std', type=float, default=0.5, help='the homoscedastic noise associated with all observations')
parser.add_argument('-te', '--thompson_sampling_n_exponent', type=float, default=2.0, help='E-TSE uses n=alpha**2 * domain_cardinality^thompson_sampling_n_exponent many samples to estimate the probability of optimality yielding an accuracy of about 1/(alpha * domain_cardinality^(thompson_sampling_n_exponent/2))')
parser.add_argument('-ta', '--alpha', type=float, default=5.0, help='E-TSE uses n=alpha**2 * domain_cardinality^thompson_sampling_n_exponent many samples to estimate the probability of optimality yielding an accuracy of about 1/(alpha * domain_cardinality^(thompson_sampling_n_exponent/2))')

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
            "# of observation points": args.number_of_observation_points,
            "observation noise standard deviation": args.observation_noise_std,
            "E-TSE # of samples exponent": args.thompson_sampling_n_exponent,
            "E-TSE # of samples alpha coefficient": args.alpha,
        },
        save_code=True,
        name="visualise_least_number_of_Thompson_samples",
        #mode="offline"
    )

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

# observe f_true at specified points
observation_indices = (jnp.arange(args.number_of_observation_points) * args.domain_cardinality) // (args.number_of_observation_points-1)
random_key, one_time_key = random.split(random_key)
observation_noise = args.observation_noise_std * jax.random.normal(one_time_key, (args.number_of_observation_points,))
observation_values = f_true[observation_indices] + observation_noise

# compute posterior and sample from it
if args.timeit:
    start_time = time.perf_counter()
post_means, post_cov = gaussians.condition_on_observations(prior_means, prior_cov, observation_indices, observation_values, args.observation_noise_std * jnp.ones(args.number_of_observation_points))
if args.timeit:
    post_cov = post_cov.block_until_ready()
    print(f'{time.perf_counter() - start_time} seconds for conditioning.')
if args.timeit:
    start_time = time.perf_counter()
post_sqrt_cov = gaussians.spsd_matrix_square_root(post_cov)
if args.timeit:
    post_sqrt_cov = post_sqrt_cov.block_until_ready()
    print(f'{time.perf_counter() - start_time} seconds for second square root.')
n_o_post_samples = 5
random_key, *one_time_keys = random.split(random_key, n_o_post_samples+1)
if args.timeit:
    start_time = time.perf_counter()
posterior_samples = [gaussians.sample(post_means, post_sqrt_cov, one_time_keys[i]) for i in range(n_o_post_samples)]
if args.timeit:
    posterior_samples[-1] = posterior_samples[-1].block_until_ready()
    print(f'{time.perf_counter() - start_time} seconds for sampling from posterior.')

# compute true probability of optimality according to Thompson sampling
if args.timeit:
    start_time = time.perf_counter()
random_key, one_time_key = random.split(random_key)
thompson_probs = poo_estimators_and_BO.etse_poo(post_means, post_cov, number_of_samples=int(args.alpha**2 * args.domain_cardinality**args.thompson_sampling_n_exponent), random_key=one_time_key)
if args.timeit:
    thompson_probs.block_until_ready()
    time_etse = time.perf_counter() - start_time
    print(f'{time_etse} seconds for E-TSE.')

# compute probability of optimality according to the concentrated maximum method
if args.timeit:
    start_time = time.perf_counter()
cm_probs = poo_estimators_and_BO.cme_poo(post_means, gaussians.standard_deviations(post_cov), alpha=args.alpha)
if args.timeit:
    cm_probs.block_until_ready()
    time_cme = time.perf_counter() - start_time
    print(f'{time_cme} seconds for CME.')

if args.two_d_domain: # 3d plotting
    # set up plotting
    fig, axs = plt.subplots(2, 2, subplot_kw=dict(projection='3d'), figsize=(12, 15), dpi=400)

    # plot f_true
    Z = f_true.reshape((sqrt_domain_cardinality, sqrt_domain_cardinality))
    colors = cm.viridis(plt.Normalize(jnp.min(Z), jnp.max(Z))(Z))
    surf = axs[0, 0].plot_surface(x1, x2, Z, rcount = sqrt_domain_cardinality, ccount = sqrt_domain_cardinality, shade=True, facecolors=colors, label=r"$f_{true}$")
    #surf.set_facecolor((0,0,0,0.5))

    #plot observations
    axs[0, 0].scatter(x[observation_indices, 0], x[observation_indices, 1], observation_values, c="k", label="observations")

    # plot f_posterior
    Z = post_means.reshape((sqrt_domain_cardinality, sqrt_domain_cardinality))
    stds = gaussians.standard_deviations(post_cov).reshape((sqrt_domain_cardinality, sqrt_domain_cardinality))
    norm = plt.Normalize(jnp.min(stds), jnp.max(stds))
    colors = cm.inferno(norm(stds))
    surf = axs[0, 1].plot_surface(x1, x2, Z, rcount = sqrt_domain_cardinality, ccount = sqrt_domain_cardinality, shade=True, facecolors=colors, label=r"$p(f\ |\ \mathcal{D})$")
    cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.inferno), ax=axs[0, 1], pad=0.2, fraction=0.025)
    cbar.ax.set_title(r"$\sigma_{F\ |\ \mathcal{D}}$", rotation=0, fontsize=12)

    # plot probability of optimality acquisition functions
    Z = thompson_probs.reshape((sqrt_domain_cardinality, sqrt_domain_cardinality)) * args.domain_cardinality
    colors = cm.viridis(plt.Normalize(jnp.min(Z), jnp.max(Z))(Z))
    axs[1, 0].plot_surface(x1, x2, Z, rcount = sqrt_domain_cardinality, ccount = sqrt_domain_cardinality, shade=True, facecolors=colors, label=rf'E-TSE with ${args.alpha if int(args.alpha) != args.alpha else int(args.alpha)}^2 \cdot | \mathcal{{X}}\, |^{{{args.thompson_sampling_n_exponent if int(args.thompson_sampling_n_exponent) != args.thompson_sampling_n_exponent else int(args.thompson_sampling_n_exponent)}}}$ samples')
    Z = cm_probs.reshape((sqrt_domain_cardinality, sqrt_domain_cardinality)) * args.domain_cardinality
    colors = cm.viridis(plt.Normalize(jnp.min(Z), jnp.max(Z))(Z))
    axs[1, 1].plot_surface(x1, x2, Z, rcount = sqrt_domain_cardinality, ccount = sqrt_domain_cardinality, shade=True, facecolors=colors, label="CME")

    # adjust style
    axs[0, 0].legend(loc="upper left")
    axs[0, 0].set_xlabel("x1", fontsize=12)
    axs[0, 0].set_ylabel("x2", fontsize=12)
    axs[0, 0].set_zlabel("", fontsize=12)
    axs[0, 0].set_title(r"Target Function $f_{true}$")
    axs[0, 1].legend(loc="upper left")
    axs[0, 1].set_xlabel("x1", fontsize=12)
    axs[0, 1].set_ylabel("x2", fontsize=12)
    axs[0, 1].set_zlabel(r"$\mu_{f\ |\ \mathcal{D}}$", fontsize=12)
    axs[0, 1].set_title(r"Modelling $f_{true}$ with a Gaussian Process")
    axs[1, 0].legend(loc="upper left")
    axs[1, 0].set_xlabel("x1", fontsize=12)
    axs[1, 0].set_ylabel("x2", fontsize=12)
    axs[1, 0].set_zlabel(r'E-TSE(x) [$\frac{1}{| \mathcal{X}\, |}$]', fontsize=12)
    axs[1, 0].set_title(r'E-TSE for $\mathrm{\mathbb{P}}[x \in X^*\!]$')
    axs[1, 1].legend(loc="upper left")
    axs[1, 1].set_xlabel("x1", fontsize=12)
    axs[1, 1].set_ylabel("x2", fontsize=12)
    axs[1, 1].set_zlabel(r'IE(x) [$\frac{1}{| \mathcal{X}\, |}$]', fontsize=12)
    axs[1, 1].set_title(r'IE for $\mathrm{\mathbb{P}}[x \in X^*\!]$')

else: # 1d plotting
    # set up plotting
    fig, axs = plt.subplots(2, 1, figsize=(12, 15), dpi=400)
    plt.subplots_adjust(top = 0.9, bottom=0.1, hspace=0.5, wspace=0.3)

    # plot f_true
    axs[0].plot(x[:, 0], f_true, c="k", label=r"$f_{true}$")

    # plot f_posterior
    axs[0].scatter(x[observation_indices, 0], observation_values, c="k", label="observations")
    axs[0].plot(x[:, 0], post_means, "b", label=r"$\mathrm{\mathbb{E}}[f\ |\ \mathcal{D}]$")
    axs[0].fill_between(x[:, 0], post_means - gaussians.standard_deviations(post_cov), post_means + gaussians.standard_deviations(post_cov), color='b', alpha=.1, label=r"$\sigma_{f\ |\ \mathcal{D}}$")
    for i in range(n_o_post_samples):
        axs[0].plot(x[:, 0], posterior_samples[i], "--")
    
    #plot probability of optimality acquisition functions
    axs[1].plot(x[:, 0], thompson_probs * args.domain_cardinality, "g--", label="E-TSE")
    axs[1].plot(x[:, 0], cm_probs * args.domain_cardinality, "r", label="CME")

    # adjust style
    axs[0].legend(loc="upper left", fontsize=12)
    axs[0].set_xlabel("x", fontsize=12)
    axs[0].set_ylabel("", fontsize=12)
    axs[0].set_title(r"Modelling $f_{true}$ with a Gaussian Process")
    axs[1].legend(loc="upper left", fontsize=12)
    axs[1].set_xlabel("x", fontsize=12)
    axs[1].set_ylabel(r'$\mathrm{\mathbb{P}}[x \in X^*]$ [$\frac{1}{| \mathcal{X}\, |}$]', fontsize=12)
    axs[1].set_title("Probability of Optimality")

date_time = datetime.datetime.strftime(datetime.datetime.now(), '%d:%b:%Y-%H:%M:%S')
plt.savefig(f'results/{date_time}.pdf', format='pdf')
with open(f'results/{date_time}.json', 'w') as fp:
    information = {'script name' : os.path.basename(__file__)}
    information.update(vars(args))
    json.dump(information, fp)

plt.show()

if args.timeit:
    wandb.log({"time[s] elapsed for E-TSE and CME": wandb.Table(columns=["E-TSE", "CME"], data=[[time_etse, time_cme], ])})
wandb.log({"required Thompson samples visualisation": wandb.Image(PIL.Image.frombytes('RGBA', fig.canvas.get_width_height(), fig.canvas.buffer_rgba(), 'raw'))})
wandb.finish()