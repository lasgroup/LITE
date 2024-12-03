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
plt.rcParams['font.size'] = 24
from cycler import cycler
plt.rc('axes', prop_cycle=(cycler('color', ['tab:cyan', 'tab:pink', 'tab:green', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:red', 'tab:blue']) +
                           cycler('linestyle', [':', ':', ':', ':', ':', ':', ':', ':']) +
                           cycler('marker', ['o', 's', '^', 'D', 'v', '>', '<', 'p'])))
#lines = {'linestyle': 'None'}
#plt.rc('lines', **lines)

import src.divergences as div

parser = argparse.ArgumentParser(description='plots the runtime of different methods for estimating probability of maximality against the domain cardinality', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("poos_with_runtimes", type=str, nargs="+", help="npz files as returned by 'test_function_poo_during_sampling' that contain the estimated probabilities of optimality as well as associated runtimes")
args = parser.parse_args()
print(args) # for logging purposes

# opens files for poos_with_runtimes
poo_during_sampling_reports = []
for file_name in args.poos_with_runtimes:
    poo_during_sampling_reports.append(jnp.load(file=file_name))

# adds reports sorted by estimation method and alpha to runtimes_dict
runtimes_dict = {}
for report in poo_during_sampling_reports:
    estimation_method = str(report['estimation_method'])
    alpha             = float(report["alpha"])
    domain_size = jnp.load(file=str(report['bayes_opt_report']))['f_true'].size
    runtime = jnp.mean(report['runtimes'])
        
    if (estimation_method, alpha) not in runtimes_dict:
        runtimes_dict[(estimation_method, alpha)] = {}
    if domain_size not in runtimes_dict[(estimation_method, alpha)]:
        runtimes_dict[(estimation_method, alpha)][domain_size] = []
    runtimes_dict[(estimation_method, alpha)][domain_size].append(runtime)
    
statistics = {}
for (estimation_method, alpha), domainsize_runtimes_dict in runtimes_dict.items():
    statistics[(estimation_method, alpha)] = ([], [], []) # domain_size, mean_runtime, stderr_runtime / std_runtime
    for domain_size, runtimes in domainsize_runtimes_dict.items():
        mean_runtime = sum(runtimes) / len(runtimes)
        var_runtime = sum((runtime - mean_runtime)**2 for runtime in runtimes) / (len(runtimes) - 1)
        stderr_runtime = var_runtime**.5 / len(runtimes)
        statistics[(estimation_method, alpha)][0].append(domain_size)
        statistics[(estimation_method, alpha)][1].append(mean_runtime)
        statistics[(estimation_method, alpha)][2].append(stderr_runtime)
    statistics[(estimation_method, alpha)] = tuple(map(list, zip(*sorted(zip(*statistics[(estimation_method, alpha)]), key=lambda x: x[0])))) # sort according to domain_size
    
fig, ax = plt.subplots(1, 1, figsize=(10, 8), dpi=400)

for (estimation_method, alpha), stats in statistics.items():
    domain_sizes, mean_run_times, stderr_run_times = stats
    # plot runtimes
    ax.errorbar(x=domain_sizes, y=mean_run_times, xerr=0, yerr=stderr_run_times, label=rf"${estimation_method}$", linewidth=3.0, markersize=16)#,fmt='') #with $\alpha={alpha}$

ax.set_xlabel(r"domain size $|\mathcal{X}|$")
ax.set_xscale("log")
ax.set_ylabel("seconds on NVIDIA TITAN RTX")
ax.set_yscale("log")
ax.legend(loc="upper left")

date_time = datetime.datetime.strftime(datetime.datetime.now(), '%d:%b:%Y-%H:%M:%S')
plt.savefig(f'results/{date_time}.pdf', format='pdf')
with open(f'results/{date_time}.json', 'w') as fp:
    information = {'script name' : os.path.basename(__file__)}
    information.update(vars(args))
    json.dump(information, fp)

plt.show()