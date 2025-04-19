import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.stats import norm # Can be used for proposal, but using np.random.normal

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.mcmc.metropolis_hastings import target_pdf_unnormalized, metropolis_sampler

# --- Example Parameters ---
if __name__ == "__main__":
    # Sampler parameters
    proposal_std = 1.0 # Standard deviation for the Normal proposal distribution
    num_samples = 50000
    burn_in = 5000

    print("Running Metropolis Sampler for target ~ exp(-x^4):")
    print(f"Sampler: proposal_std={proposal_std}, num_samples={num_samples}, burn_in={burn_in}\n")

    # Run the sampler
    samples, acceptance_rate = metropolis_sampler(target_pdf_unnormalized, proposal_std, num_samples, burn_in)

    print(f"Generated {len(samples)} samples (after burn-in).")
    print(f"Acceptance Rate: {acceptance_rate:.4f}")

    # --- Plotting Results ---
    print("\nPlotting results...")

    # 1. Trace plot
    plt.figure(figsize=(10, 4))
    plt.plot(samples)
    plt.title('Trace Plot of Metropolis Samples')
    plt.xlabel('Iteration (Post Burn-in)')
    plt.ylabel('Sample Value')
    plt.grid(True)
    plt.show()

    # 2. Histogram of samples vs target PDF
    plt.figure(figsize=(10, 6))
    # Plot histogram of samples
    plt.hist(samples, bins=50, density=True, alpha=0.7, label='Metropolis Samples')

    # Plot the target PDF (normalized)
    # We need to calculate the normalization constant C = integral(exp(-x^4) dx) from -inf to +inf
    norm_constant, _ = quad(target_pdf_unnormalized, -np.inf, np.inf)
    print(f"Normalization constant (estimated): {norm_constant:.4f}")

    x_range = np.linspace(min(samples), max(samples), 500)
    target_pdf_normalized = target_pdf_unnormalized(x_range) / norm_constant
    plt.plot(x_range, target_pdf_normalized, 'r-', lw=2, label='Target PDF (exp(-x^4)/C)')

    plt.title('Distribution of Metropolis Samples vs Target PDF')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.show()
