import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random
import json
import datetime
import argparse
import time
import os
import matplotlib.pyplot as plt
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.size'] = 16
from cycler import cycler
plt.rc('axes', prop_cycle=(cycler('color', ['tab:cyan', 'tab:pink', 'tab:green', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:red', 'tab:blue']) +
                           cycler('linestyle', [':', ':', ':', ':', ':', ':', ':', ':']) + # ['-', '--', ':', '-.', ':', '--', '-', '-.']
                           cycler('marker', ['o', 's', '^', 'D', 'v', '>', '<', 'p'])))
from tqdm import tqdm

import src.divergences as div
import src.gaussians as gaussians

parser = argparse.ArgumentParser(description='', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("poos", type=str, nargs="+", help="npz files as returned by 'test_function_poo_during_sampling' that contain the estimated probabilities of optimality during Bayesian optimisation")
parser.add_argument("-k", type=int, default=20, help="the number of set cardinalities that are considered")
parser.add_argument("-gte", "--ground_truth_estimator", type=str, default="E-TSE", help="which estimator to take as ground truth of PoM")
parser.add_argument('-s', '--seed', type=int, default=0, help='the random seed for reproducibility, affects the TS selection strategy')
parser.add_argument("--use_last_instead_of_mean", action="store_true", help="whether to use the last posterior of BO instead of the average across the runtime")
args = parser.parse_args()

print(args) # for logging purposes

fractions = jnp.linspace(start=0, stop=1, num=args.k)

# opens files containing poos
poo_reports = []
for file_name in args.poos:
    poo_reports.append(jnp.load(file=file_name))

# sort poo reports by estimation method
report_for_est_dict = {}
ground_truth_poos_for_bayes_opt_report = {}
for report in poo_reports:
    estimation_method = str(report['estimation_method'])
    if estimation_method not in report_for_est_dict:
        report_for_est_dict[estimation_method] = []
    report_for_est_dict[estimation_method] += [report]
    if estimation_method == args.ground_truth_estimator:
        bayes_opt_report = str(report['bayes_opt_report'])
        alpha = float(report['alpha'])
        if bayes_opt_report not in ground_truth_poos_for_bayes_opt_report\
                            or ground_truth_poos_for_bayes_opt_report[bayes_opt_report]['alpha'] < alpha:
            ground_truth_poos_for_bayes_opt_report[bayes_opt_report] = report
bayes_opt_reports = {str(report['bayes_opt_report']): jax.numpy.load(file=str(report['bayes_opt_report'])) for report in next(iter(report_for_est_dict.values()))}
# synthetically add poo report where the poo is set to the posterior mean, just to test recall under selection of largest means
#reported_steps = next(iter(report_for_est_dict.values()))[0]['poos'].shape[0]
report_for_est_dict['Means'] = [{'bayes_opt_report': key,'poos': value['post_means']} for key, value in bayes_opt_reports.items()]

# synthetically add poo report where the poo is filled with -arange in the order of repeated TS without repetition, just to test recall under TS selection strategy
def ts_selection(post_means, post_cov, random_key):
    r = jnp.zeros_like(post_means)
    random_keys = random.split(random_key, num=r.size)
    for i in tqdm(range(r.shape[0]), desc ="Using repeated TS to produce set of high recall (most expensive part)"): # goes over BO-steps
        @jax.jit
        def ts_selection_body_func(j:int, xstars:jax.Array):
            f = gaussians.sample(post_means[i, :], post_cov[i, :, :], random_keys[j + i * r.shape[1]])
            f = f.at[xstars].set(-jnp.inf) # avoids repeatedly giving same xstar, should skip indices -1 which stand for None
            xstar = jnp.argmax(f)
            xstars = xstars.at[j].set(xstar)
            return xstars
        xstars = jax.lax.fori_loop(0, r.shape[1], ts_selection_body_func, -jnp.ones((r.shape[1],), dtype=int), unroll=None) # goes over selected set cardinality
        r = r.at[i, xstars].set(-jnp.arange(r.shape[1])) # labels the xstars positions in decreasing order
    return r
#ts_for_multiple_posteriors = jax.jit(jax.vmap(gaussians.sample, in_axes=(0, 0, 0), out_axes=0))
random_keys = random.split(random.key(args.seed), num=len(bayes_opt_reports))
report_for_est_dict['TS-Select'] = [{'bayes_opt_report': key, 'poos': ts_selection(value['post_means'], value['post_cov'], random_keys[idx])} for idx, (key, value) in enumerate(bayes_opt_reports.items())]


statistics = {key: (None, None) for key in report_for_est_dict.keys()}

# computes recalls
for estimation_method, reports_list in report_for_est_dict.items():
    reports_recalls_over_time = []
    report_recalls_avg_mean = []
    report_recalls_avg_stderr = []
    for report in reports_list:
        associated_posterior_report = bayes_opt_reports[str(report['bayes_opt_report'])]
        f_true = associated_posterior_report['f_true']
        xstar = jnp.argmax(f_true)

        associated_ground_truth_poo = ground_truth_poos_for_bayes_opt_report[str(report['bayes_opt_report'])]['poos']

        max_poo_indices = jnp.argsort(report['poos'], descending=True)
        reports_recalls_over_time += [jnp.zeros((report['poos'].shape[0], args.k))]
        for idx, frac in enumerate(fractions):
            cardinality = int(frac * report['poos'].shape[1])
            #reports_recalls_over_time[-1] = reports_recalls_over_time[-1].at[:, idx].set(jnp.any(max_poo_indices[:, :cardinality] == xstar, axis=-1))
            reports_recalls_over_time[-1] = reports_recalls_over_time[-1].at[:, idx].set(jnp.sum(jnp.take_along_axis(associated_ground_truth_poo, max_poo_indices[:, :cardinality], axis=1), axis=1))
    if use_last_instead_of_mean:
        reports_recalls_avgs = [report_recalls_over_time[-1, :] for report_recalls_over_time in reports_recalls_over_time]
    else:
        reports_recalls_avgs = [jnp.mean(report_recalls_over_time[:, :], axis=0) for report_recalls_over_time in reports_recalls_over_time]

    report_recalls_avg_mean = sum(reports_recalls_avgs) / len(reports_recalls_avgs)
    report_recalls_avg_var = sum((reports_recalls_avg - report_recalls_avg_mean)**2 for reports_recalls_avg in reports_recalls_avgs) / (len(reports_recalls_avgs) - 1)
    report_recalls_avg_stderr = report_recalls_avg_var**.5 / len(reports_recalls_avgs)**.5

    for idx, frac in enumerate(fractions):
        #cardinality = int(frac * report['poos'].shape[1])
        print(f"For cardinality fraction {frac}, {estimation_method} has recall {report_recalls_avg_mean[idx]} +- {report_recalls_avg_stderr[idx]}")
    
    statistics[estimation_method] = (report_recalls_avg_mean, report_recalls_avg_stderr)

fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=400)
for estimation_method in statistics:
    ax.plot(fractions, statistics[estimation_method][0], label=rf"${estimation_method}$")
    ax.fill_between(fractions, statistics[estimation_method][0] - statistics[estimation_method][1], statistics[estimation_method][0] + statistics[estimation_method][1], linewidth=2, alpha=.1)

ax.set_xlabel(r'fraction of $|\mathcal{X}|$')
ax.set_ylabel(r'recall')
ax.legend(loc="upper left", fontsize=13)
date_time = datetime.datetime.strftime(datetime.datetime.now(), '%d:%b:%Y-%H:%M:%S')
plt.savefig(f'results/{date_time}.pdf', format='pdf')
with open(f'results/{date_time}.json', 'w') as fp:
    information = {'script name' : os.path.basename(__file__)}
    information.update(vars(args))
    json.dump(information, fp)
plt.show()