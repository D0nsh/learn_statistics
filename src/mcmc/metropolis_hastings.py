import numpy as np
# Removed matplotlib and scipy.integrate imports as they are moved to examples

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

# --- Removed parameter definitions, execution, and plotting ---