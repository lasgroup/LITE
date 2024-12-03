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
                           cycler('linestyle', [':', ':', ':', ':', ':', ':', ':', ':']) + # ['-', '--', ':', '-.', ':', '--', '-', '-.']
                           cycler('marker', ['o', 's', '^', 'D', 'v', '>', '<', 'p'])))
#lines = {'linestyle': 'None'}
#plt.rc('lines', **lines)

import src.divergences as div

parser = argparse.ArgumentParser(description='plots accuracy against compute time on a fixed GPU for different methods of estimating probability of maximality', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("poos_with_runtimes", type=str, nargs="+", help="npz files as returned by 'test_function_poo_during_sampling' that contain the estimated probabilities of optimality as well as associated runtimes")
parser.add_argument("-gte", "--ground_truth_estimator", type=str, default="E-TSE", help="which estimator to take as ground truth")
parser.add_argument("-ermsre", "--entropy_rmsre_instead_of_tv_distance", action="store_true", help="whether to report on the root mean squared relative error of entropy rather than on the TV distance")
parser.add_argument("--logy", action="store_true", help="whether to use a logarithmic y-axis")
parser.add_argument("--n_warmup_steps", type=int, default=5, help="how many warmup steps are skipped in the plot")

args = parser.parse_args()
print(args) # for logging purposes

# opens files for poos_with_runtimes
poo_during_sampling_reports = []
for file_name in args.poos_with_runtimes:
    poo_during_sampling_reports.append(jnp.load(file=file_name))

# sort reports first by estimation method, then by alpha. also sets up accuracy-compute-points dict and ground_truth_poos_for_bayes_opt_report dict
report_dict = {}
accuracy_compute_points = {}
ground_truth_poos_for_bayes_opt_report = {}
for report in poo_during_sampling_reports:
    estimation_method = str(report['estimation_method'])
    alpha             = float(report["alpha"])
    seed              = int(report['seed'])

    if estimation_method not in report_dict:
        report_dict[estimation_method] = {}
        accuracy_compute_points[estimation_method] = [[], [], [], [], []] # x-axis compute mean, x-axis compute std, y-axis accuracy mean, y-axis accuracy std, and alphas
    if alpha not in report_dict[estimation_method]:
        report_dict[estimation_method][alpha] = []
    report_dict[estimation_method][alpha] += [report]

    if estimation_method == args.ground_truth_estimator:
        bayes_opt_report = str(report['bayes_opt_report'])
        if bayes_opt_report not in ground_truth_poos_for_bayes_opt_report\
                            or ground_truth_poos_for_bayes_opt_report[bayes_opt_report]['alpha'] < alpha:
            ground_truth_poos_for_bayes_opt_report[bayes_opt_report] = report

for bayes_opt_report, poos_report in ground_truth_poos_for_bayes_opt_report.items():
    print(f"ground-truth POO is computed using {poos_report['estimation_method']} with alpha={poos_report['alpha']} for bayes_opt_report at path {bayes_opt_report}")

# fills accuracy-compute-points dict
for estimation_method, reports_dict in report_dict.items():
     # compute and accuracy lists
    for alpha, reports_list in reports_dict.items(): # at this point all elements in al share the same alpha and estimation method, but possibly different seeds and posteriors on which they were run

        #mean_run_time = sum(jnp.sum(report['runtimes']) for report in reports_list)\
        #              / sum(report['runtimes'].size     for report in reports_list)
        #mean_tv_dist  = sum(sum(div.tv_dist(report['poos'][i, :], ground_truth_poos_for_bayes_opt_report[str(report['bayes_opt_report'])]['poos'][i, :]) for i in range(report['poos'].shape[0])) for report in reports_list)\
        #              / sum(report['poos'].shape[0]                                                                                                                                               for report in reports_list)
        #var_run_time  = sum(jnp.sum((report['runtimes'] - mean_run_time)**2) for report in reports_list)\
        #              /(sum(report['runtimes'].size                          for report in reports_list) - 1)
        #var_tv_dist   = sum(sum((div.tv_dist(report['poos'][i, :],ground_truth_poos_for_bayes_opt_report[str(report['bayes_opt_report'])]['poos'][i, :]) - mean_tv_dist)**2 for i in range(report['poos'].shape[0])) for report in reports_list)\
        #              /(sum(report['poos'].shape[0]                                  
        
        # averages over BO steps                                                                                                                                for report in reports_list) - 1)
        run_times_per_step = [jnp.mean(report['runtimes'][args.n_warmup_steps:]) for report in reports_list]

        if args.entropy_rmsre_instead_of_tv_distance:
            estimated_entropies =    [[jnp.sum(jax.scipy.special.entr(report['poos'][i, :]))                                                                  for i in range(args.n_warmup_steps, report['poos'].shape[0])] for report in reports_list]
            ground_truth_entropies = [[jnp.sum(jax.scipy.special.entr(ground_truth_poos_for_bayes_opt_report[str(report['bayes_opt_report'])]['poos'][i, :])) for i in range(args.n_warmup_steps, report['poos'].shape[0])] for report in reports_list]
            # of course not TV-dists, but can override the same variable for an analogous plot with rmsre (across optimisation path, still separate for different seeds)
            tv_dists_per_step = [( sum( ((est_entropy - ground_truth_entropy) / ground_truth_entropy)**2 for est_entropy, ground_truth_entropy in zip(est_report_entropies, ground_truth_report_entropies)) / len(est_report_entropies))**.5 for est_report_entropies, ground_truth_report_entropies in zip(estimated_entropies, ground_truth_entropies)]
        else: # mean TV dist across optimisation, standard deviation is taken across different seeds
            tv_dists_per_step = [jnp.mean(div.tv_dist(report['poos'], ground_truth_poos_for_bayes_opt_report[str(report['bayes_opt_report'])]['poos'], axis=1)[args.n_warmup_steps:]) for report in reports_list]

        # averages across different BO paths (different seed => different posterior)
        mean_run_time = sum(run_times_per_step) / len(run_times_per_step)
        mean_tv_dist  = sum(tv_dists_per_step)  / len(tv_dists_per_step)
        var_run_time  = sum((rtps - mean_run_time)**2 for rtps in run_times_per_step) / (len(run_times_per_step) - 1)
        var_tv_dist  = sum((tvdps - mean_tv_dist)**2 for tvdps in tv_dists_per_step) / (len(tv_dists_per_step) - 1)

        std_run_time   = var_run_time**.5
        stderr_run_time= std_run_time / len(run_times_per_step)**.5 #/ sum(report['runtimes'].size for report in reports_list)
        std_tv_dist    = var_tv_dist**.5
        stderr_tv_dist = std_tv_dist / len(tv_dists_per_step)**.5 #/ sum(report['poos'].shape[0] for report in reports_list) 

        accuracy_compute_points[estimation_method][0] += [mean_run_time]
        accuracy_compute_points[estimation_method][1] += [stderr_run_time] # could also use std
        accuracy_compute_points[estimation_method][2] += [mean_tv_dist]
        accuracy_compute_points[estimation_method][3] += [stderr_tv_dist] # could also use std
        accuracy_compute_points[estimation_method][4] += [alpha]
    # sort according to alpha
    zipped_lists = zip(accuracy_compute_points[estimation_method][0], # mean run_time
                       accuracy_compute_points[estimation_method][1], # std/stderr run_time
                       accuracy_compute_points[estimation_method][2], # mean tv_dist
                       accuracy_compute_points[estimation_method][3], # std/stderr tv_dist
                       accuracy_compute_points[estimation_method][4]) # alphas
    sorted_zipped_lists = sorted(zipped_lists, key=lambda x: x[4]) # sort according to alphas
    l0,l1,l2,l3,l4 = zip(*sorted_zipped_lists)
    accuracy_compute_points[estimation_method][0] = list(l0)
    accuracy_compute_points[estimation_method][1] = list(l1)
    accuracy_compute_points[estimation_method][2] = list(l2)
    accuracy_compute_points[estimation_method][3] = list(l3)
    accuracy_compute_points[estimation_method][4] = list(l4)

    # removes comparison against one-self
    if estimation_method == args.ground_truth_estimator:
        mean_runtime = accuracy_compute_points[estimation_method][0].pop()
        err_runtime = accuracy_compute_points[estimation_method][1].pop()
        accuracy_compute_points[estimation_method][2].pop()
        accuracy_compute_points[estimation_method][3].pop()
        accuracy_compute_points[estimation_method][4].pop()
        print(f"computing ground-truth with {args.ground_truth_estimator} took {mean_run_time} +- {err_runtime} seconds")

fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=400)

for estimation_method, a_c_points in accuracy_compute_points.items():
    print(estimation_method, a_c_points)
    err_bar = ax.errorbar(x=a_c_points[0], y=a_c_points[2], xerr=a_c_points[1], yerr=a_c_points[3], label=f"{estimation_method}", linewidth=1.0, markersize=4)#,fmt='')
    #ax.scatter(a_c_points[0], a_c_points[2], s=1, c='k', label=r'accuracy $\alpha$' if estimation_method == "E-TSE" else '')
    line_color = err_bar[0].get_color()
    for i, alpha in enumerate(a_c_points[4]):
        if i%2 != 0:
            continue
        if estimation_method == "E-TSE":
            offset = 6
        else:
            offset = 10 if i % 4 == 0 else -15
        #eta = round(1 / (alpha * report_dict[estimation_method][alpha][0]['poos'].shape[1]), 3)
        plt.annotate(alpha, (a_c_points[0][i], a_c_points[2][i]), textcoords="offset points", xytext=(0,offset), ha='center', color=line_color, fontsize=8)

ax.set_xscale('log')
#ax.set_xlim([3e-4, 4])
if args.logy:
    ax.set_yscale('log')
#ax.set_ylim([0.03, 0.08])
ax.set_xlabel('seconds on NVIDIA TITAN RTX')
ax.set_ylabel("RMSRE of $H[X^* | \mathcal{D}]$" if args.entropy_rmsre_instead_of_tv_distance else r'$d_{TV}$')
ax.legend(loc="upper left", fontsize=13)

date_time = datetime.datetime.strftime(datetime.datetime.now(), '%d:%b:%Y-%H:%M:%S')
plt.savefig(f'results/{date_time}.pdf', format='pdf')
with open(f'results/{date_time}.json', 'w') as fp:
    information = {'script name' : os.path.basename(__file__)}
    information.update(vars(args))
    json.dump(information, fp)

plt.show()