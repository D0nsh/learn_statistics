import numpy as np
# Removed matplotlib import as plotting is moved to examples

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

# --- Removed parameter definitions, execution, and plotting ---