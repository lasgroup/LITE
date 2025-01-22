
# Introduction

This is the official codebase for the experimental section of LITE.

The outermost scripts are intended to be called from the main directory `LITE` and provide a manual, e.g.

> python code/visualise_distribution_of_iid_gaussians.py -h

They set up the various experiments conducted for the development of the associated paper and combine their respective results. 

A practitioner that wants to use LITE in their codebase only needs the files src/poo_estimators_and_BO.py, src/gaussians.py, and src/kernels. Indeed, only the methods nie_poo, cme_poo, and vapor_poo in src/poo_estimators_and_BO.py and their few dependencies are essential for running LITE and F-VAPOR, the other functions are adaptions thereof and alternative ideas that did not make it into the paper. We still provide them at the user's discretion. Alternatively, `flite.py` in the directory `LITE` provides short self-contained code for F-LITE, one of the variants of LITE.

# Digging into the code
Compared to the paper version, there are some differences in the nomenclature of the PoM estimators.
TS-MC                   <-> E-TSE (Exhaustive Thompson Sampling Estimator)
Independence Assumption <-> IE (Independence Estimator)
A-LITE                  <-> NIE (Normal Independence Estimator)
F-LITE                  <-> CME (Concentrated Maximum Estimator)
EST                     <-> NEST (Normalised EST)
Also, whereas the code speaks of probability of optimality (POO), the paper called the same concept probability of maximality (PoM).
When going through the code, it will become apparent that we have explored much more estimators and procedures than indicated in the paper submission. The practicioner is advised to cherry-pick the estimation methods that they find most useful. However, A-LITE, F-LITE, and F-VAPOR, the methods emphasised in the paper, can be run by calling the following functions in src/poo_estimators_and_BO.py:

- nie_poo(gaussian_means:jax.Array, gaussian_stds:jax.Array, alpha:float, simplified=False)
- cme_poo(gaussian_means:jax.Array, gaussian_stds:jax.Array, alpha:float, scaling:float=1.0)
- vapor_poo(gaussian_means:jax.Array, gaussian_stds:jax.Array, alpha:float)

Here, alpha describes the relative convergence of the algorithms to their analytical expressions (elementwise precision of estimates at 1 / (alpha |X|)). In practice, alpha = 100 ensures good convergence in our experience. The scaling factor can be used to adapt the prudence of the estimator by globally rescaling the uncertainty in the form of the standard deviation of the Gaussian vector. If simplified is set to true for NIE/A-LITE, only A-LITE I is used. This runs slightly more quickly by avoiding the nested quartile fitting, but is less accurate.

# Install requirements

The project is based on Python 3.11.6. Create a virtual python environment using

> python -m venv .venv

Activate the environment with

> source .venv/bin/activate

Finally, install the requirements with

> python -m pip install -r code/requirements.txt

To be able to log and visualise the results we use [Weights & Biases](wandb.ai). After creating an account on their website, log in with

> wandb login

before running the scripts


# Reproducing Results
In the following we provide the commands to reproduce our results assuming they are executed from the main directory. Depending on the settings one may need GPU support and sufficient time to reproduce our results, the baseline TS-MC runs really slowly (which is what motivated the development of LITE in the first place).

## 1-dim GP and 2-dim GP experiment
Plots the TV distance and Sinkhorn distance to the groundtruth during Bayesian optimisation based on Thompson Sampling

> python code/poo_accuracy_during_sampling.py -s=$seed -no=200

> python code/poo_accuracy_during_sampling.py -s=$seed -c=400 -no=50 --two_d_domain -k=laplacian -l=0.1

In order to combine the results of several regret experiments into one plot, use 

> python code/poo_accuracy_during_sampling_combiner.py file1.npz file2.npz ...

## Synthetic Distributions experiment

Plots the TV distance between IE and CME (as well as its competitors) as the domain size increases. We use the settings:
--std_range=10 --mean_range=5
--std_range=2 --mean_range=5
--std_range=0.5 --mean_range=5
--std_range=0.5 --mean_range=0.1

for the script

> python code/poo_accuracy_for_independent_gp.py -s=?

where we run s from 0 to 19 (for 20 repetitions). In order to combine the results of several regret experiments into one plot, use 

> python code/poo_accuracy_for_independent_gp_combiner.py file1.npz file2.npz ...

## DropWave and Quadcopter
To analyze the PoM estimation for Bayesian Optimization of fixed test functions such as those provided by DropWave and Quadrotor, one can first run Bayesian optimisation with Expected Improvement to obtain the Gaussian posteriors at intermediate time steps:

> python code/test_function_posterior_during_sampling.py drop-wave-mini -rcc -s=0 -t --mlm_n_random_observations=10 --n_observations=30

Further information on the individual settings, including for marginal likelihood maximisation, can be obtained by querying the help manual from the program. Once one has stored the posteriors, one can run the following script to derive the associated PoM estimates according to the various considered PoM estimators:

> python code/test_function_poo_during_sampling.py

With the PoMs (POOs) savely stored, we can examine various metrics to compare the relative performance of different PoM estimators. To that end, we provide the following programs (each with their own -h manual)

> python code/test_function_accuracy_vs_compute.py

> python code/test_function_entropy_rmsre_over_time.py

> python code/test_function_recall.py

> python code/test_function_runtime_per_domain_size.py

> python code/test_function_visualise_posterior.py

## Appendix, illustrate the number of samples necessary for accurate TS-MC

We evaluate the exhaustive Thompson sampling estimator (E-TSE) for a different number of samples $\alpha^2 | \mathcal X |^2$ where we vary $\alpha$ in (5.0, 1.0, 0.2, 0.04). To give a more complete picture, and compare against CME, we repeat the process for two different kernels (gaussian, laplacian)

> python code/visualise_least_number_of_Thompson_samples.py -c 2500 --two_d_domain --kernel=gaussian --alpha=5.0

> python code/visualise_least_number_of_Thompson_samples.py -c 2500 --two_d_domain --kernel=laplacian --alpha=5.0


> python code/visualise_least_number_of_Thompson_samples.py -c 2500 --two_d_domain --kernel=gaussian --alpha=1.0

> python code/visualise_least_number_of_Thompson_samples.py -c 2500 --two_d_domain --kernel=laplacian --alpha=1.0


> python code/visualise_least_number_of_Thompson_samples.py -c 2500 --two_d_domain --kernel=gaussian --alpha=0.2

> python code/visualise_least_number_of_Thompson_samples.py -c 2500 --two_d_domain --kernel=laplacian --alpha=0.2


> python code/visualise_least_number_of_Thompson_samples.py -c 2500 --two_d_domain --kernel=gaussian --alpha=0.04

> python code/visualise_least_number_of_Thompson_samples.py -c 2500 --two_d_domain --kernel=laplacian --alpha=0.04

## Appendix, illustrate the distribution of the maximum of standard normals
To illustrate the distribution of the maximum of standard normals run the following two commands:

> python code/visualise_distribution_of_iid_gaussians.py

> python code/visualise_distribution_of_iid_gaussians.py --n_density 250 --domain_density 1000000 --mean_std

The domain density is increased to reduce errors in the numerical estimation of first and second moments, see the bump at around $n=10^{15}$.

## Appendix, Impact of the Independence Assumption
In order to understand the impact of falsely dropping all correlation structure, run the following commands. You will obtain a comparison between TS-MC and estimation with Independence Assumption for various settings. Notice that qualitatively, the estimation remains truthful despite neglecting correlation information.

Boundary Discrepancies

> python code/visualise_independence_estimator.py -c 200 --kernel=gaussian --length_scale 0.02

> python code/visualise_independence_estimator.py -c 200 --kernel=laplacian --length_scale 0.02

> python code/visualise_independence_estimator.py -c 200 --kernel=laplacian --length_scale 0.005

> python code/visualise_independence_estimator.py -c 50 --kernel=gaussian --length_scale 0.02

Domain size

> python code/visualise_independence_estimator.py -c 100 --kernel=gaussian --length_scale 0.04 --number_of_observation_points=2

> python code/visualise_independence_estimator.py -c 200 --kernel=gaussian --length_scale 0.02 --number_of_observation_points=4

> python code/visualise_independence_estimator.py -c 400 --kernel=gaussian --length_scale 0.01 --number_of_observation_points=8

> python code/visualise_independence_estimator.py -c 800 --kernel=gaussian --length_scale 0.005 --number_of_observation_points=16

Uniformity of the posterior

> python code/visualise_independence_estimator.py -c 200 --kernel=gaussian --length_scale 0.02 --number_of_observation_points=2

> python code/visualise_independence_estimator.py -c 200 --kernel=gaussian --length_scale 0.02 --number_of_observation_points=4

> python code/visualise_independence_estimator.py -c 200 --kernel=gaussian --length_scale 0.02 --number_of_observation_points=8

> python code/visualise_independence_estimator.py -c 200 --kernel=gaussian --length_scale 0.02 --number_of_observation_points=16

## EntropySearch Regret Experiments
In order to compare Entropy Search using TS-MC and Entropy Search using F-LITE, run the following commands:

> python code/regret_experiment.py CES -c=250 -o=50 -n=0.2 -r=10 --timeit

> python code/regret_experiment.py ES -c=250 -o=50 -n=0.2 -r=10 --timeit

Here, CES refers to using F-LITE and ES refers to using TS-MC. In order to combine the results of the two experiments into one plot, use 

> python code/regret_experiment_combiner.py file1.npz file2.npz ...

## 1k-dim linear kernel experiment
For the large scale experiment, a lot of storage, RAM, and compute are necessary. Run the following commands to generate posteriors from 800 steps of Bayesian optimization (based on EI and UCB) on a domain of 10k elements in 1k dimensions:

> python code/test_function_posterior_during_sampling.py random_linear -a=EI -s=$i -t -k=linear -no=800 --max_n_observations_for_mlm=150

> python code/test_function_posterior_during_sampling.py random_linear -a=UCB -s=$i -t -k=linear -no=800 --max_n_observations_for_mlm=150

Next, process these posteriors by running PoM estimation on them. Here, running the Independence Assumption (IE) would incur a runtime of 500 hours on an A100 GPU, so we will only run F-LITE (CME).

> python code/test_function_poo_during_sampling.py FILENAME CME --alpha=100
