import numpy as np
import matplotlib.pyplot as plt

def gibbs_sampler_bivariate_normal(mu1, mu2, sigma1, sigma2, rho, num_samples, burn_in):
    """
    Performs Gibbs sampling for a bivariate normal distribution.

    Args:
        mu1, mu2: Means of the two variables.
        sigma1, sigma2: Standard deviations of the two variables.
        rho: Correlation coefficient.
        num_samples: Total number of samples to generate (including burn-in).
        burn_in: Number of initial samples to discard.

    Returns:
        A numpy array of shape (num_samples - burn_in, 2) containing samples.
    """
    samples = np.zeros((num_samples, 2))
    x1, x2 = 0.0, 0.0 # Initial state (can be anything)

    # Pre-calculate conditional parameters (means depend on the other variable)
    cond_var1 = sigma1**2 * (1 - rho**2)
    cond_var2 = sigma2**2 * (1 - rho**2)
    cond_sd1 = np.sqrt(cond_var1)
    cond_sd2 = np.sqrt(cond_var2)

    for i in range(num_samples):
        # Sample x1 given x2
        cond_mean1 = mu1 + (sigma1 / sigma2) * rho * (x2 - mu2)
        x1 = np.random.normal(cond_mean1, cond_sd1)

        # Sample x2 given the *new* x1
        cond_mean2 = mu2 + (sigma2 / sigma1) * rho * (x1 - mu1)
        x2 = np.random.normal(cond_mean2, cond_sd2)

        samples[i, :] = [x1, x2]

    # Discard burn-in samples
    final_samples = samples[burn_in:, :]
    return final_samples

# --- Parameters for the target Bivariate Normal ---
mu1, mu2 = 2, 0
sigma1, sigma2 = 1, 1
rho = 0.8 # Strong positive correlation

# --- MCMC Parameters ---
total_samples = 20000
burn_in_period = 2000

# --- Run the Gibbs Sampler ---
mcmc_samples = gibbs_sampler_bivariate_normal(mu1, mu2, sigma1, sigma2, rho,
                                              total_samples, burn_in_period)

# --- Plotting ---
plt.figure(figsize=(12, 6))

# Plot the samples
plt.subplot(1, 2, 1)
plt.scatter(mcmc_samples[:, 0], mcmc_samples[:, 1], alpha=0.3, s=10, label='MCMC Samples')
plt.title(f'Gibbs Samples from Bivariate Normal (rho={rho})')
plt.xlabel('x1')
plt.ylabel('x2')
plt.axhline(0, color='grey', lw=0.5)
plt.axvline(0, color='grey', lw=0.5)
plt.axis('equal')
plt.legend()

# Plot trace plots to check convergence (should look like white noise around the mean)
plt.subplot(1, 2, 2)
plt.plot(mcmc_samples[:, 0], lw=0.5, label='Trace of x1')
plt.plot(mcmc_samples[:, 1], lw=0.5, label='Trace of x2')
plt.title('Trace Plots')
plt.xlabel('Iteration (post burn-in)')
plt.ylabel('Sampled Value')
plt.legend()

plt.tight_layout()
plt.show()

# Plot histograms of the marginal distributions
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(mcmc_samples[:, 0], bins=50, density=True, alpha=0.7, label='Hist(x1)')
plt.title('Marginal Distribution of x1')
plt.xlabel('x1')
plt.ylabel('Density')
# You could overlay the theoretical N(mu1, sigma1^2) density here

plt.subplot(1, 2, 2)
plt.hist(mcmc_samples[:, 1], bins=50, density=True, alpha=0.7, label='Hist(x2)')
plt.title('Marginal Distribution of x2')
plt.xlabel('x2')
plt.ylabel('Density')
# You could overlay the theoretical N(mu2, sigma2^2) density here

plt.tight_layout()
plt.show()