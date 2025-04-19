import pytest
import numpy as np
import sys
import os

# Add src directory to path to allow importing modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.mcmc.gibbs_sampling import gibbs_sampler_bivariate_normal
from src.mcmc.metropolis_hastings import metropolis_sampler, target_pdf_unnormalized

# --- Tests for Gibbs Sampling ---

def test_gibbs_sampler_output_shape():
    """Test the output shape of the Gibbs sampler."""
    np.random.seed(42)
    mu1, mu2, sigma1, sigma2, rho = 0, 0, 1, 1, 0.8
    num_samples = 1000
    burn_in = 100
    samples = gibbs_sampler_bivariate_normal(mu1, mu2, sigma1, sigma2, rho, num_samples, burn_in)
    assert samples.shape == (num_samples - burn_in, 2)

def test_gibbs_sampler_mean_covariance():
    """Test if sample mean and covariance are close to theoretical values (requires many samples)."""
    np.random.seed(42)
    mu1, mu2, sigma1, sigma2, rho = 0, 0, 1, 1, 0.8
    num_samples = 20000 # More samples for better estimates
    burn_in = 2000
    samples = gibbs_sampler_bivariate_normal(mu1, mu2, sigma1, sigma2, rho, num_samples, burn_in)

    target_mean = np.array([mu1, mu2])
    target_cov = np.array([[sigma1**2, rho*sigma1*sigma2], [rho*sigma1*sigma2, sigma2**2]])

    sample_mean = np.mean(samples, axis=0)
    sample_cov = np.cov(samples, rowvar=False)

    # Use np.allclose for comparing arrays with tolerance
    assert np.allclose(sample_mean, target_mean, atol=0.1)
    assert np.allclose(sample_cov, target_cov, atol=0.15)

# --- Tests for Metropolis-Hastings ---

def test_metropolis_sampler_output_shape():
    """Test the output shape of the Metropolis sampler."""
    np.random.seed(42)
    num_samples = 1000
    burn_in = 100
    samples, rate = metropolis_sampler(target_pdf_unnormalized, 1.0, num_samples, burn_in)
    assert samples.shape == (num_samples - burn_in,)
    assert 0 <= rate <= 1

def test_metropolis_sampler_symmetry():
    """Test if the distribution is roughly symmetric around 0 for exp(-x^4)."""
    np.random.seed(42)
    num_samples = 50000
    burn_in = 5000
    samples, _ = metropolis_sampler(target_pdf_unnormalized, 1.0, num_samples, burn_in)
    sample_mean = np.mean(samples)
    # Mean should be close to 0 for this symmetric distribution
    assert abs(sample_mean) < 0.1

# Add more tests as needed...
