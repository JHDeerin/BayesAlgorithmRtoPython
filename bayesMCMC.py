'''
Implements a basic MCMC Bayesian algorithm to sample means/standard deviations from a dataset with binary categories
TODO: Verify the above intent is correct (via Mat)

This rewrite doesn't attempt to refactor the code structurally or significantly increase performance, and is instead meant to be a fairly close 1:1 conversion of Mat's original R script into python
'''

# Numpy, a numerical package for doing generic computations efficiently
import numpy as np
# Pandas, a library for doing dataset manipulation
import pandas as pd
# Scipy's statistics package, containing many canonical distributions
import scipy.stats as sp

# TODO: Remove (for debugging only)
import time

# Read observed data CSV with 3 columns: a case ID, a "status" (1st category, 2nd category, or NA/unknown), and a "measure"
start_time = time.perf_counter()

df_data_observed = pd.read_csv('finaldata_observed.csv')

max_iterations = 1000 #10000
burn_in_iterations = 100 #1000
num_samples = len(df_data_observed.index)

# Set up matrices that will store results from each iteration
parameter_estimates = np.zeros((max_iterations, 4))
hidden_variables = np.zeros((max_iterations, num_samples))

# Initial guess for parameters
avg0_old = 11
sd0_old = 3
avg1_old = 10
sd1_old = 3
parameter_estimates[0, :] = [avg0_old, sd0_old, avg1_old, sd1_old]

# Initialize priors for means
# TODO: Look over this again to better understand it
avg0_0 = 11
sigma0_0 = 100
avg0_1 = 10
sigma0_1 = 100

# Initialize priors for standard deviations
# TODO: Look over this again to better understand it
alpha_0 = 0.001
beta_0 = 0.001
alpha_1 = 0.001
beta_1 = 0.001

# Randomly seed hidden observations with 0 or 1 (initial category guess)
hidden_variables[0,:] = np.random.choice([0, 1], size=(num_samples))

# Mark known variables with the correct category
hidden_variables[0, np.where(df_data_observed['status'] == 0)] = 0
hidden_variables[0, np.where(df_data_observed['status'] == 1)] = 1

# Get measurement data for both categories
status_0_measures = df_data_observed.loc[df_data_observed['status'] == 0]['measure']
status_1_measures = df_data_observed.loc[df_data_observed['status'] == 0]['measure']
all_data_measures = df_data_observed['measure']

for i in range(1, max_iterations):
    # Get previous iteration's estimates
    hidden_variables_old = hidden_variables[i-1, :]
    avg0_old, sd0_old, avg1_old, sd1_old = parameter_estimates[i-1, :]

    # Sample from posterior to get new estimates of our parameters
    # (Uses conjugate priors to mathe-magically speed things up)
    # TODO: Clean this up and refactor into a function
    n0 = np.count_nonzero(hidden_variables_old == 0)
    mean0 = np.mean(status_0_measures)

    # TODO: Review to understand how this is calculated?
    avg0_new = np.random.normal(
        loc=(1/sigma0_0**2 + n0/sd0_old**2)**(-1) * (avg0_0/sigma0_0**2 + mean0*n0/sd0_old**2),
        scale=np.sqrt((1/sigma0_0**2 + 1/sd0_old**2)**-1),
        size=1)
    sum_squares0 = np.sum((status_0_measures - avg0_new)**2)
    # invgamma takes arguments(alpha, beta)
    sd0_new = np.sqrt(sp.invgamma.rvs(alpha_0 + n0/2, beta_0 + sum_squares0/2))

    # Do the same thing for status 1
    n1 = np.count_nonzero(hidden_variables_old == 1)
    mean1 = np.mean(status_1_measures)

    # TODO: Review to understand how this is calculated?
    avg1_new = np.random.normal(
        loc=(1/sigma0_1**2 + n1/sd1_old**2)**(-1) * (avg0_1/sigma0_1**2 + mean1*n1/sd1_old**2),
        scale=np.sqrt((1/sigma0_1**2 + 1/sd1_old**2)**-1),
        size=1)
    sum_squares1 = np.sum((status_1_measures - avg1_new)**2)
    # invgamma takes arguments(alpha, beta)
    sd1_new = np.sqrt(sp.invgamma.rvs(alpha_1 + n1/2, beta_1 + sum_squares1/2))

    # Constraint that mu1 < mu0 TODO: Review why this is needed?
    if avg1_new > avg0_new:
        avg1_new = avg1_old
        avg0_new = avg0_old
        sd0_new = sd0_old
        sd1_new = sd1_old

    # Sample from hidden variables
    likelihood_ratio = np.zeros(num_samples)
    hidden_variables_new = hidden_variables_old[:]

    # TODO: This section down currently accounts for ~3/4 of the runtime; benchmark why?
    # Calculate likelihood ratios P(Data | New guess) / P(Data | Old guess)
    likelihood_ratio[hidden_variables_old == 0] = (
        sp.norm.pdf(all_data_measures[hidden_variables_old==0], avg1_new, sd1_new)
        / sp.norm.pdf(all_data_measures[hidden_variables_old==0], avg0_new, sd0_new))

    likelihood_ratio[hidden_variables_old == 1] = (
        sp.norm.pdf(all_data_measures[hidden_variables_old==1], avg0_new, sd0_new)
        / sp.norm.pdf(all_data_measures[hidden_variables_old==1], avg1_new, sd1_new))

    # Generate num_sample uniform samples between 0 and 1
    # TODO: What does 'z' represent?
    z = np.random.choice([0, 1], size=(num_samples))

    # If z < likelihood ratio, accept the new state; otherwise, keep the old
    hidden_variables_new[z < likelihood_ratio] = 1 - hidden_variables_old[z<likelihood_ratio]

    # For points that are definitely known, keep the observed values
    hidden_variables_new[np.where(df_data_observed['status'] == 0)] = 0
    hidden_variables_new[np.where(df_data_observed['status'] == 1)] = 1

    parameter_estimates[i, :] = [avg0_new, sd0_new, avg1_new, sd1_new]
    hidden_variables[i, :] = hidden_variables_new

# Remove burned-in samples, where our initial guess was still converging
parameter_estimates = parameter_estimates[burn_in_iterations:, :]
hidden_variables = hidden_variables[burn_in_iterations:, :]

end_time = time.perf_counter()

print(f'DONE! Finished in {end_time - start_time}')
