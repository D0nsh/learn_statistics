import sys
import os
import matplotlib.pyplot as plt # Added for potential future plotting

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.monte_carlo.pi_estimation import estimate_pi

# --- Simulation Execution ---
if __name__ == "__main__":
    num_points_to_test = [100, 1000, 10000, 100000, 1000000]
    print("Estimating Pi using Monte Carlo:")

    for num_points in num_points_to_test:
        pi_estimate = estimate_pi(num_points)
        print(f"  Number of points: {num_points:<10} | Estimated Pi: {pi_estimate:.6f}")

    # Example of how you might add plotting later
    # estimates = [estimate_pi(n) for n in num_points_to_test]
    # plt.figure(figsize=(8, 5))
    # plt.plot(num_points_to_test, estimates, marker='o', linestyle='-')
    # plt.axhline(y=math.pi, color='r', linestyle='--', label='Actual Pi')
    # plt.xscale('log')
    # plt.xlabel("Number of Points")
    # plt.ylabel("Estimated Pi")
    # plt.title("Monte Carlo Pi Estimation Convergence")
    # plt.legend()
    # plt.grid(True)
    # plt.show()
