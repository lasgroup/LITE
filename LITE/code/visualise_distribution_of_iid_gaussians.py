import wandb
import jax
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
import PIL
import json
import datetime
import argparse
import os

parser = argparse.ArgumentParser(description='computes and visualises the distribution of the maximum of n i.i.d. Gaussians. Can be set either to show the pdf or to plot the mean and standard deviation', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--mean_std', action="store_true", help='displays a graph of the mean and standard deviation instead of the pdfs')
parser.add_argument('-e', '--n_max_exponent', type=int, default=50, help='exponent of largest number of i.i.d. Gaussians (n) that are considered, i.e. n_max = 10^n_max_exponent')
parser.add_argument('-r', '--n_density', type=int, default=20, help='the discretisation of n, i.e. the number of points in [1, 10^n_max_exponent] on a logarithmic scale')
parser.add_argument('-ds', '--domain_start', type=int, default=-5, help='the start of the domain on which the pdfs are computed and displayed')
parser.add_argument('-df', '--domain_stop', type=int, default=20, help='the stop of the domain on which the pdfs are computed and displayed')
parser.add_argument('-d', '--domain_density', type=int, default=10000, help='the number of points describing the discretisation of the domain on which the pdfs are computed and displayed')
parser.add_argument('-ep', '--epsilon', type=float, default=0.1, help='the epsilon for tail probabilities P[|max F_x - mode| > epsilon] that are printed for each n. Make sure that (domain_stop - domain_start)/domain_density << epsilon to ensure faithful approximation')

args = parser.parse_args()

print(args) # for logging purposes

wandb.init(
        project="master-thesis",
        config={
            "show mean & std": args.mean_std,
            "n max exponent": args.n_max_exponent,
            "n density": args.n_density,
            "domain start": args.domain_start,
            "domain stop": args.domain_stop,
            "domain density": args.domain_density,
        },
        save_code=True,
        name="visualise_distribution_of_iid_gaussians",
        #mode="offline"
    )


def cdf(z:float, n:float) -> float:
    """Computes the cdf of max_{i=1,...,n} X_i where X_i ~ N(0,1) at z

    Args:
        z (float): evaluation point
        n (float): number of Gaussians

    Returns:
        float: cdf of max of i.i.d. standard normals
    """
    return jax.numpy.exp(n*jax.scipy.stats.norm.logcdf(z))

def pdf(z:float, n:float) -> float:
    """Computes pdf of max_{i=1,...,n} X_i for independent X_i ~ N(0,1) at z

    Args:
        z (float): evaluation point
        n (float): number of Gaussians

    Returns:
        float: pdf of max of i.i.d. standard normals
    """
    return (cdf(z, n-1) * n * jax.scipy.stats.norm.pdf(z))

def delta(pdf:jax.Array, dz:float, epsilon:float):
    """Computes the probability of the tails P[|Z-mode(pdf)| > epsilon] where Z ~ pdf

    Args:
        pdf (jax.Array): pdf of Z
        dz (float): the (equal) measure of each point in the discretised pdf
        epsilon (float): the tail that we want to establish

    Returns:
        float: approximation to P[|Z-mode| > epsilon]
    """
    mode = jax.numpy.argmax(pdf)
    epsilon_idx = int(epsilon/dz) # integer approximation due to discretisation
    return 1-jax.numpy.sum(pdf[mode-epsilon_idx:mode+epsilon_idx])*dz


evaluation_points = jax.numpy.linspace(start=args.domain_start, stop=args.domain_stop, num=args.domain_density)
dz = (args.domain_stop - args.domain_start) / args.domain_density
n_points = jax.numpy.logspace(start=0, stop=args.n_max_exponent, num=args.n_density, base=10)

means = []
stds = []

if args.mean_std:
    fig, ax = plt.subplots(1, 1, dpi=400)
    twin_ax = ax.twinx()
else:
    fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection='3d'), dpi=400)
for zi, pdf_zi, ni in ((evaluation_points, pdf(evaluation_points, n), n) for n in n_points):
    if args.mean_std:
        mean = jax.numpy.sum(zi * pdf_zi * dz)
        variance = jax.numpy.sum(zi ** 2 * pdf_zi * dz) - mean**2
        means.append(mean)
        stds.append(variance**.5)
    else:
        ax.plot(zi, [jax.numpy.log10(ni)]*len(zi), pdf_zi)
    print(f"P[|max_{{i=1,...,10^{jax.numpy.log10(ni):0.2f}}} F_x - mode| > {args.epsilon}] = {delta(pdf_zi, dz, args.epsilon):0.5f}")
if args.mean_std:
    ax.plot(n_points, means, color='k', label=r'$\mu_{X^*}$')
    twin_ax.plot(n_points, stds, color='b', label=r'$\sigma_{X^*}$')
    ax.set_xlabel(r'$n$', fontsize=12)
    ax.set_xscale('log')
    ax.set_ylabel(r'$\mu_{X^*}$', fontsize=12)
    ax.legend(loc="upper left")
    twin_ax.set_ylabel(r'$\sigma_{X^*}$', fontsize=12)
    #twin_ax.set_yscale('log')
    twin_ax.legend(loc="upper right")
    ax.set_title("Mean and Standard Deviation of Maximum of i.i.d. Standard Gaussians")
else:
    ax.set_xlabel(r"$f$")
    ax.set_ylabel(r'$\log_{10} n$')
    ax.set_zlabel(r"$\frac{d}{df}\mathrm{\mathbb{P}} [X^*\! \leq\ f\ ]$")
    ax.set_title("Distribution of Maximum of i.i.d. Standard Gaussians")

date_time = datetime.datetime.strftime(datetime.datetime.now(), '%d:%b:%Y-%H:%M:%S')
plt.savefig(f'results/{date_time}.pdf', format='pdf')
with open(f'results/{date_time}.json', 'w') as fp:
    information = {'script name' : os.path.basename(__file__)}
    information.update(vars(args))
    json.dump(information, fp)

plt.show()

wandb.log({"distribution of iid Gaussians visualisation": wandb.Image(PIL.Image.frombytes('RGBA', fig.canvas.get_width_height(), fig.canvas.buffer_rgba(), 'raw'))})
wandb.finish()