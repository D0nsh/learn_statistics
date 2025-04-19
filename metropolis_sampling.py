import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

def target_pdf_unnormalized(x):
  """Unnormalized target density function: exp(-x^4)"""
  # Use log-sum-exp trick if needed for numerical stability, not critical here
  return np.exp(-x**4)

def metropolis_sampler(target_func_unnorm, proposal_std, num_samples, burn_in):
    """
    Performs Metropolis sampling for a 1D distribution.

    Args:
        target_func_unnorm: Function that computes the unnormalized target density.
        proposal_std: Standard deviation for the Normal proposal distribution.
        num_samples: Total number of samples to generate.
        burn_in: Number of initial samples to discard.

    Returns:
        A numpy array of samples (post burn-in).
        Acceptance rate.
    """
    samples = np.zeros(num_samples)
    current_x = 0.0 # Initial state
    accepted_count = 0

    for i in range(num_samples):
        # Propose a new state
        proposed_x = np.random.normal(current_x, proposal_std)

        # Calculate acceptance probability
        # Use logs for numerical stability when numbers are very small
        log_target_current = -current_x**4
        log_target_proposed = -proposed_x**4
        log_acceptance_ratio = log_target_proposed - log_target_current

        # alpha = min(1, target(proposed) / target(current))
        # alpha = min(1, exp(log_target_proposed - log_target_current))
        # Check if log(alpha) < log(u) where u ~ Uniform(0,1)
        # is equivalent to alpha < u
        # In logs: log(alpha) = min(0, log_acceptance_ratio)

        if log_acceptance_ratio >= 0: # P(prop)/P(curr) >= 1 -> alpha = 1
             acceptance_prob = 1.0
        else:
             acceptance_prob = np.exp(log_acceptance_ratio) # P(prop)/P(curr)

        # Decide whether to accept
        u = np.random.uniform(0, 1)
        if u < acceptance_prob:
            current_x = proposed_x
            accepted_count += 1
        # else: current_x remains the same

        samples[i] = current_x

    final_samples = samples[burn_in:]
    acceptance_rate = accepted_count / num_samples
    return final_samples, acceptance_rate

# --- MCMC Parameters ---
proposal_sd = 0.5   # Tune this parameter!
total_samples_mh = 50000
burn_in_period_mh = 5000

# --- Run the Metropolis Sampler ---
mcmc_samples_mh, acceptance_rate_mh = metropolis_sampler(target_pdf_unnormalized,
                                                         proposal_sd,
                                                         total_samples_mh,
                                                         burn_in_period_mh)

print(f"Metropolis Acceptance Rate: {acceptance_rate_mh:.3f} (aim for ~0.2-0.5)")

# --- Plotting ---
plt.figure(figsize=(12, 6))

# Plot Histogram of samples vs True PDF (normalized)
plt.subplot(1, 2, 1)
plt.hist(mcmc_samples_mh, bins=50, density=True, alpha=0.7, label='MCMC Samples')

# Calculate the normalizing constant Z = ∫ exp(-x^4) dx
Z, _ = quad(target_pdf_unnormalized, -np.inf, np.inf)
print(f"Normalizing constant Z = {Z:.4f}")

x_vals = np.linspace(min(mcmc_samples_mh), max(mcmc_samples_mh), 200)
true_pdf = target_pdf_unnormalized(x_vals) / Z
plt.plot(x_vals, true_pdf, 'r-', lw=2, label='True PDF (normalized)')
plt.title('Metropolis Samples vs True PDF')
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()

# Plot trace plot
plt.subplot(1, 2, 2)
plt.plot(mcmc_samples_mh, lw=0.5)
plt.title('Trace Plot')
plt.xlabel('Iteration (post burn-in)')
plt.ylabel('Sampled Value')

plt.tight_layout()
plt.show()