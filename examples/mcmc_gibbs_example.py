import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # For nicer plotting

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.mcmc.gibbs_sampling import gibbs_sampler_bivariate_normal

# --- Example Parameters ---
if __name__ == "__main__":
    # Parameters for the target bivariate normal distribution
    mu1, mu2 = 0, 0
    sigma1, sigma2 = 1, 1
    rho = 0.8 # Correlation

    # Sampler parameters
    num_samples = 10000
    burn_in = 1000

    print("Running Gibbs Sampler for Bivariate Normal Distribution:")
    print(f"Parameters: mu=({mu1},{mu2}), sigma=({sigma1},{sigma2}), rho={rho}")
    print(f"Sampler: num_samples={num_samples}, burn_in={burn_in}\n")

    # Run the sampler
    samples = gibbs_sampler_bivariate_normal(mu1, mu2, sigma1, sigma2, rho, num_samples, burn_in)

    print(f"Generated {samples.shape[0]} samples (after burn-in). Shape: {samples.shape}")
    print(f"Sample mean: {np.mean(samples, axis=0)}")
    print(f"Sample covariance matrix:\n{np.cov(samples, rowvar=False)}")

    # --- Plotting Results ---
    print("\nPlotting results...")

    # 1. Trace plots (check for convergence)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(samples[:, 0])
    plt.title('Trace Plot for x1')
    plt.xlabel('Iteration')
    plt.ylabel('x1')

    plt.subplot(1, 2, 2)
    plt.plot(samples[:, 1])
    plt.title('Trace Plot for x2')
    plt.xlabel('Iteration')
    plt.ylabel('x2')
    plt.tight_layout()
    plt.show()

    # 2. Scatter plot of samples vs theoretical distribution
    plt.figure(figsize=(7, 7))
    # Use seaborn for a nice joint plot with marginal histograms
    sns.jointplot(x=samples[:, 0], y=samples[:, 1], kind='scatter', s=10, alpha=0.5)
    # Overlay theoretical contours (optional, requires calculating ellipse parameters)
    # from matplotlib.patches import Ellipse
    # cov = np.array([[sigma1**2, rho*sigma1*sigma2], [rho*sigma1*sigma2, sigma2**2]])
    # lambda_, v = np.linalg.eig(cov)
    # lambda_ = np.sqrt(lambda_)
    # for j in range(1, 4):
    #     ell = Ellipse(xy=(mu1, mu2),
    #                   width=lambda_[0]*j*2, height=lambda_[1]*j*2,
    #                   angle=np.rad2deg(np.arccos(v[0, 0])))
    #     ell.set_facecolor('none')
    #     ell.set_edgecolor('red')
    #     plt.gca().add_artist(ell)
    plt.suptitle('Gibbs Samples vs Theoretical Bivariate Normal', y=1.02)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid(True)
    plt.show()
