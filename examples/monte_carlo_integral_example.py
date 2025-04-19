import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad # For comparison

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.monte_carlo.integral_estimation import g, estimate_integral

# --- Simulation Execution ---
if __name__ == "__main__":
    # Parameters for the integration
    a = 0  # Lower bound
    b = 1  # Upper bound
    num_samples_to_test = [100, 1000, 10000, 100000]

    print(f"Estimating the integral of g(x) = x^2 from {a} to {b}:")

    estimates = []
    for num_samples in num_samples_to_test:
        integral_estimate = estimate_integral(g, a, b, num_samples)
        estimates.append(integral_estimate)
        print(f"  Number of samples: {num_samples:<7} | Estimated Integral: {integral_estimate:.6f}")

    # Calculate the exact integral for comparison
    exact_integral, _ = quad(g, a, b)
    print(f"\nExact Integral (using scipy.integrate.quad): {exact_integral:.6f}")

    # Plotting the convergence (optional)
    plt.figure(figsize=(8, 5))
    plt.plot(num_samples_to_test, estimates, marker='o', linestyle='-', label='MC Estimate')
    plt.axhline(y=exact_integral, color='r', linestyle='--', label=f'Exact Value ({exact_integral:.4f})')
    plt.xscale('log')
    plt.xlabel("Number of Samples")
    plt.ylabel("Integral Estimate")
    plt.title("Monte Carlo Integration Convergence")
    plt.legend()
    plt.grid(True)
    plt.show()
