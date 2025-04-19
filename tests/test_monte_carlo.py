import pytest
import numpy as np
import sys
import os

# Add src directory to path to allow importing modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.monte_carlo.pi_estimation import estimate_pi
from src.monte_carlo.integral_estimation import estimate_integral, g
from src.monte_carlo.option_pricing import monte_carlo_option_price, black_scholes_call

# --- Tests for Pi Estimation ---

def test_estimate_pi():
    """Test if pi estimation is roughly correct."""
    # Use a fixed seed for reproducibility in tests
    np.random.seed(42)
    num_points = 100000
    pi_est = estimate_pi(num_points)
    # Check if the estimate is within a reasonable range of pi
    assert abs(pi_est - np.pi) < 0.05

# --- Tests for Integral Estimation ---

def test_estimate_integral():
    """Test if integral estimation is roughly correct."""
    np.random.seed(42)
    a, b = 0, 1
    num_samples = 50000
    integral_est = estimate_integral(g, a, b, num_samples)
    exact_integral = 1/3 # Integral of x^2 from 0 to 1
    assert abs(integral_est - exact_integral) < 0.05

# --- Tests for Option Pricing ---

def test_monte_carlo_option_price():
    """Test if MC option price converges towards Black-Scholes."""
    np.random.seed(42)
    S0 = 100
    K = 105
    T = 1.0
    r = 0.05
    sigma = 0.2
    num_simulations = 20000 # Use fewer simulations for faster tests
    num_steps = 50

    mc_price = monte_carlo_option_price(S0, K, T, r, sigma, num_simulations, num_steps)
    bs_price = black_scholes_call(S0, K, T, r, sigma)

    # Allow a larger tolerance due to MC variance
    assert abs(mc_price - bs_price) < 0.5

# Add more tests as needed...
