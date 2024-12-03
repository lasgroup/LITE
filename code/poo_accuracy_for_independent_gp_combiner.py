import jax
import PIL
import json
import datetime
import argparse
import time
import os
import matplotlib.pyplot as plt
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.size'] = 20

parser = argparse.ArgumentParser(description='combines the results of several "POO accuracy for independent GP" experiments into a single set of plots', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("results", type=str, nargs="+", help="npz files of results from experiments")
parser.add_argument('-md', '--max_domain_size', type=int, default=2000, help='the maximum domain size on which the accuracy was evaluated')
parser.add_argument('-nd', '--n_domain_sizes', type=int, default=40, help='the number of domain sizes that were evaluated between 2 and --max_domain_size')
args = parser.parse_args()

print(args) # for logging purposes

domain_sizes = jax.numpy.floor(jax.numpy.logspace(start=1, stop=jax.numpy.log2(args.max_domain_size), num=args.n_domain_sizes, base=2)).astype(int)
domain_sizes = jax.numpy.unique(domain_sizes) # remove duplicates

# opens files
results = []
for file in args.results:
    results.append(jax.numpy.load(file=file))

tv_dists_mean_cme   = sum(result["tv_dists_cme"]   for result in results) / len(results)   
#tv_dists_mean_ocme  = sum(result["tv_dists_ocme"]  for result in results) / len(results) 
tv_dists_mean_vapor = sum(result["tv_dists_vapor"] for result in results) / len(results) 
tv_dists_mean_nest  = sum(result["tv_dists_nest"]  for result in results) / len(results) 
#tv_dists_mean_est   = sum(result["tv_dists_est"]   for result in results) / len(results) 
#tv_dists_mean_nies  = sum(result["tv_dists_nies"]  for result in results) / len(results) 
tv_dists_mean_nie  = sum(result["tv_dists_nie"]  for result in results) / len(results) 

tv_dists_stds_cme   = ( sum((result["tv_dists_cme"]   - tv_dists_mean_cme  )**2 for result in results) / (len(results) - 1) )**.5   
#tv_dists_stds_ocme  = ( sum((result["tv_dists_ocme"]  - tv_dists_mean_ocme )**2 for result in results) / (len(results) - 1) )**.5 
tv_dists_stds_vapor = ( sum((result["tv_dists_vapor"] - tv_dists_mean_vapor)**2 for result in results) / (len(results) - 1) )**.5 
tv_dists_stds_nest  = ( sum((result["tv_dists_nest"]  - tv_dists_mean_nest )**2 for result in results) / (len(results) - 1) )**.5 
#tv_dists_stds_est   = ( sum((result["tv_dists_est"]   - tv_dists_mean_est  )**2 for result in results) / (len(results) - 1) )**.5 
#tv_dists_stds_nies  = ( sum((result["tv_dists_nies"]  - tv_dists_mean_nies )**2 for result in results) / (len(results) - 1) )**.5 
tv_dists_stds_nie  = ( sum((result["tv_dists_nie"]  - tv_dists_mean_nie )**2 for result in results) / (len(results) - 1) )**.5 

tv_dists_sterr_cme   = tv_dists_stds_cme   / len(results)**.5
#tv_dists_sterr_ocme  = tv_dists_stds_ocme  / len(results)**.5
tv_dists_sterr_vapor = tv_dists_stds_vapor / len(results)**.5
tv_dists_sterr_nest  = tv_dists_stds_nest  / len(results)**.5
#tv_dists_sterr_est   = tv_dists_stds_est   / len(results)**.5
#tv_dists_sterr_nies  = tv_dists_stds_nies  / len(results)**.5
tv_dists_sterr_nie  = tv_dists_stds_nie  / len(results)**.5

fig, ax = plt.subplots(1, 1, dpi=400, figsize=(8.25, 6))

#ax.plot(domain_sizes, tv_dists_mean_est,   'k-.', label=r'$EST$')
ax.plot(domain_sizes, tv_dists_mean_nest,  'r:', label=r'$NEST$')
ax.plot(domain_sizes, tv_dists_mean_vapor, 'g-.', label=r'$VAPOR$')
ax.plot(domain_sizes, tv_dists_mean_cme,   'b--', label=r'$CME$')
#ax.plot(domain_sizes, tv_dists_mean_ocme,  'c-.', label=r'$OCME$')
#ax.plot(domain_sizes, tv_dists_mean_nies,  'y-.', label=r'$NIES$')
ax.plot(domain_sizes, tv_dists_mean_nie,  'm-', label=r'$NIE$')

#ax.fill_between(domain_sizes, tv_dists_mean_est   - tv_dists_sterr_est,   tv_dists_mean_est   + tv_dists_sterr_est,   color='k', alpha=.1)
ax.fill_between(domain_sizes, tv_dists_mean_nest  - tv_dists_sterr_nest,  tv_dists_mean_nest  + tv_dists_sterr_nest,  color='r', alpha=.1)
ax.fill_between(domain_sizes, tv_dists_mean_vapor - tv_dists_sterr_vapor, tv_dists_mean_vapor + tv_dists_sterr_vapor, color='g', alpha=.1)
ax.fill_between(domain_sizes, tv_dists_mean_cme   - tv_dists_sterr_cme,   tv_dists_mean_cme   + tv_dists_sterr_cme,   color='b', alpha=.1)
#ax.fill_between(domain_sizes, tv_dists_mean_ocme  - tv_dists_sterr_ocme,  tv_dists_mean_ocme  + tv_dists_sterr_ocme,  color='c', alpha=.1)
#ax.fill_between(domain_sizes, tv_dists_mean_nies  - tv_dists_sterr_nies,  tv_dists_mean_nies  + tv_dists_sterr_nies,  color='y', alpha=.1)
ax.fill_between(domain_sizes, tv_dists_mean_nie  - tv_dists_sterr_nie,  tv_dists_mean_nie  + tv_dists_sterr_nie,  color='m', alpha=.1)

ax.set_xlabel(r'$|\mathcal{X}|$')
ax.set_xscale('log')
ax.legend(loc="upper right")
ax.set_ylabel(r'$d_{TV}$')

date_time = datetime.datetime.strftime(datetime.datetime.now(), '%d:%b:%Y-%H:%M:%S')
plt.savefig(f'results/{date_time}.pdf', format='pdf')

plt.show()