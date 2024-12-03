import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
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
                           cycler('linestyle', ['-', '--', ':', '-.', ':', '--', '-', '-.']))) #+  ['-', '--', ':', '-.', ':', '--', '-', '-.']
                           #cycler('marker', ['o', 's', '^', 'D', 'v', '>', '<', 'p'])))
#lines = {'linestyle': 'None'}
#plt.rc('lines', **lines)

import src.divergences as div

parser = argparse.ArgumentParser(description='plots the entropy rmsre induced by different methods for estimating probability of maximality', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("poos_with_runtimes", type=str, nargs="+", help="npz files as returned by 'test_function_poo_during_sampling' that contain the estimated probabilities of optimality as well as associated runtimes")
parser.add_argument("-gte", "--ground_truth_estimator", type=str, default="E-TSE", help="which estimator to take as ground truth")
parser.add_argument("-instead_plot_entropy", action="store_true", help="whether to just plot entropy instead of the rmsre of entropy w.r.t. some ground-truth estimator")
parser.add_argument("--logy", action="store_true", help="whether to use a logarithmic y-axis")
parser.add_argument("--n_warmup_steps", type=int, default=5, help="how many warmup steps are skipped in the plot")
parser.add_argument("--max_steps", type=int, default=-1, help="up to how far to draw the plot")

args = parser.parse_args()
print(args) # for logging purposes

# opens files for poos_with_runtimes
poo_during_sampling_reports = []
for file_name in args.poos_with_runtimes:
    poo_during_sampling_reports.append(jnp.load(file=file_name))

# sort reports first by estimation method, then by alpha. also sets up ground_truth_poos_for_bayes_opt_report dict
report_dict = {}
if not args.instead_plot_entropy:
    ground_truth_poos_for_bayes_opt_report = {}
for report in poo_during_sampling_reports:
    estimation_method = str(report['estimation_method'])
    alpha             = float(report["alpha"])
    seed              = int(report['seed'])

    if estimation_method not in report_dict:
        report_dict[estimation_method] = {}
    if alpha not in report_dict[estimation_method]:
        if not args.instead_plot_entropy:
            report_dict[estimation_method][alpha] = []
        else:
            report_dict[estimation_method][alpha] = {}

    if not args.instead_plot_entropy:
    
        report_dict[estimation_method][alpha] += [report]

        if estimation_method == args.ground_truth_estimator:
            bayes_opt_report = str(report['bayes_opt_report'])
            if bayes_opt_report not in ground_truth_poos_for_bayes_opt_report\
                                or ground_truth_poos_for_bayes_opt_report[bayes_opt_report]['alpha'] < alpha:
                ground_truth_poos_for_bayes_opt_report[bayes_opt_report] = report

    else: # also distinguish between different methods for obtaining the posteriors        
        with open(os.path.splitext(str(report['bayes_opt_report']))[0]+'.json') as js:
            acquisition_function = json.load(js).get("acquisition_function", "EI") # add default of expected improvement because in early runs this was not added to json

        if acquisition_function not in report_dict[estimation_method][alpha]:
            report_dict[estimation_method][alpha][acquisition_function] = []
        else:
            report_dict[estimation_method][alpha][acquisition_function] += [report]
        
if not args.instead_plot_entropy:
    for bayes_opt_report, poos_report in ground_truth_poos_for_bayes_opt_report.items():
        print(f"ground-truth POO is computed using {poos_report['estimation_method']} with alpha={poos_report['alpha']} for bayes_opt_report at path {bayes_opt_report}")

statistics = {}

if not args.instead_plot_entropy:
    # computes entropy rmsre data (as well as TV distance data)
    for estimation_method, reports_dict in report_dict.items():
        # compute accuracy lists
        for alpha, reports_list in reports_dict.items(): # at this point all elements in reports_list share the same alpha and estimation method, but possibly different seeds and posteriors on which they were run                         
            # averages over BO steps
            run_times_per_step = [jnp.mean(report['runtimes']) for report in reports_list]
            mean_run_time = sum(run_times_per_step) / len(run_times_per_step)
            std_run_time  = ( sum((rtps - mean_run_time)**2 for rtps in run_times_per_step) / (len(run_times_per_step) - 1) )**.5
            # does not average over BO steps
            poos = [report['poos'][args.n_warmup_steps:args.max_steps, :] for report in reports_list]
            ground_truth_poos = [ground_truth_poos_for_bayes_opt_report[str(report['bayes_opt_report'])]['poos'] for report in reports_list]
            estimated_entropies = [jnp.sum(jax.scipy.special.entr(poo), axis=1) for poo in poos]
            ground_truth_entropies = [jnp.sum(jax.scipy.special.entr(poo), axis=1) for poo in ground_truth_poos]
            entropies_sre = [ ((est_entr - gt_entr) / gt_entr)**2 for est_entr, gt_entr in zip(estimated_entropies, ground_truth_entropies)]
            entropies_msre = sum(entropies_sre) / len(entropies_sre)
            entropies_std_sre = ( sum( (entropy_sre - entropies_msre)**2 for entropy_sre in entropies_sre) / (len(entropies_sre) - 1) )**.5
            entropies_stderr_sre = entropies_std_sre / len(entropies_sre)**.5
            tv_dists = [div.tv_dist(ground_truth_poo, poo, axis=1) for poo, ground_truth_poo in zip(poos, ground_truth_poos)]
            tv_dist_mean = sum(tv_dists) / len(tv_dists)
            tv_dist_std = ( sum( (tv_dist - tv_dist_mean)**2 for tv_dist in tv_dists) / (len(tv_dists) - 1) )**.5
            tv_dist_stderr = tv_dist_std / len(tv_dists)**.5

            statistics[(estimation_method, alpha)] = (mean_run_time, std_run_time, entropies_msre, entropies_stderr_sre, tv_dist_mean, tv_dist_stderr)
            
            # removes comparison against one-self
            statistics = {key: value for key,value in statistics.items() if not (value[4] == 0).all()} # checks if tv_dist == 0 everywhere

if args.instead_plot_entropy:
    # computes entropy
    for estimation_method, reports_dict in report_dict.items():
        # compute accuracy lists
        for alpha, inner_reports_dict in reports_dict.items(): 
            for acquisition_function, reports_list in inner_reports_dict.items():# at this point all elements in reports_list share the same alpha, estimation method, and acquisition function, but possibly different seeds and hence posteriors on which they were run                         
                print(f"{estimation_method}, {alpha}, {acquisition_function} has {len(reports_list)} samples")
                # averages over BO steps
                run_times_per_step = [jnp.mean(report['runtimes']) for report in reports_list]
                mean_run_time = sum(run_times_per_step) / len(run_times_per_step)
                std_run_time  = ( sum((rtps - mean_run_time)**2 for rtps in run_times_per_step) / (len(run_times_per_step) - 1) )**.5
                # does not average over BO steps
                poos = [report['poos'][args.n_warmup_steps:args.max_steps, :] for report in reports_list]
                estimated_entropies = [jnp.sum(jax.scipy.special.entr(poo), axis=1) for poo in poos]
                entropies_means = sum(estimated_entropies) / len(estimated_entropies)
                entropies_stds = ( sum( (entr - entropies_means)**2 for entr in estimated_entropies) / (len(estimated_entropies) - 1) )**.5

                entropies_stderrs = entropies_stds / len(estimated_entropies)**.5

                statistics[(estimation_method, alpha, acquisition_function)] = (mean_run_time, std_run_time, entropies_means, entropies_stderrs)





fig, axs = plt.subplots(1, 2, figsize=(20, 5.5), dpi=400)

for setting, stats in statistics.items():
    if args.instead_plot_entropy:
        est_method, alpha, acquisition_function = setting
        mean_run_time, std_run_time, entropies_means, entropies_stderrs = stats
        size = entropies_means.size if args.max_steps == -1 else args.max_steps
        axs[0].plot(range(args.n_warmup_steps, size), entropies_means, label=f"{acquisition_function}")# with $\alpha={alpha}$")
        axs[0].fill_between(range(args.n_warmup_steps, size), (entropies_means - entropies_stderrs), (entropies_means + entropies_stderrs), linewidth=2, alpha=.1)
    else:
        est_method, alpha = setting
        mean_run_time, std_run_time, entropies_msre, entropies_stderr_sre, tv_dist_mean, tv_dist_stderr = stats
        # plot TV distances
        size = entropies_msre.size if args.max_steps == -1 else args.max_steps
        axs[0].plot(range(args.n_warmup_steps, size), entropies_msre**.5, label=rf"${est_method}$")# with $\alpha={alpha}$")
        axs[0].fill_between(range(args.n_warmup_steps, size), (entropies_msre - entropies_stderr_sre)**.5, (entropies_msre + entropies_stderr_sre)**.5, linewidth=2, alpha=.1)
        axs[1].plot(range(args.n_warmup_steps, size), tv_dist_mean, label=rf"${est_method}$")# with $\alpha={alpha}$")
        axs[1].fill_between(range(args.n_warmup_steps, size), tv_dist_mean - tv_dist_stderr, tv_dist_mean + tv_dist_stderr, linewidth=2, alpha=.1)

axs[0].set_xlabel("steps")
if args.instead_plot_entropy:
    axs[0].set_ylabel("PoM Entropy")
else:
    axs[0].set_ylabel("RMSRE of $H[X^* | \mathcal{D}]$")
if args.logy:
    axs[0].set_yscale("log")
axs[0].legend(loc="upper right", fontsize=13)
axs[1].set_xlabel("steps")
axs[1].set_ylabel("$d_{TV}(TS,\; \cdot\; )$")
if args.logy:
    axs[1].set_yscale("log")
axs[1].legend(loc="upper right", fontsize=13)

date_time = datetime.datetime.strftime(datetime.datetime.now(), '%d:%b:%Y-%H:%M:%S')
plt.savefig(f'results/{date_time}.pdf', format='pdf')
with open(f'results/{date_time}.json', 'w') as fp:
    information = {'script name' : os.path.basename(__file__)}
    information.update(vars(args))
    json.dump(information, fp)

plt.show()