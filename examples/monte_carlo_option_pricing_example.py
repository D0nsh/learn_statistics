import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.monte_carlo.option_pricing import monte_carlo_option_price, black_scholes_call

# --- Example Parameters ---
if __name__ == "__main__":
    S0 = 100      # Initial stock price
    K = 105       # Strike price
    T = 1.0       # Time to maturity (in years)
    r = 0.05      # Risk-free interest rate
    sigma = 0.2   # Volatility
    num_simulations_list = [1000, 10000, 50000, 100000] # Number of simulation paths
    num_steps = 100 # Number of time steps in each path

    print("Estimating European Call Option Price using Monte Carlo:")
    print(f"Parameters: S0={S0}, K={K}, T={T}, r={r}, sigma={sigma}, steps={num_steps}\n")

    mc_prices = []
    for num_simulations in num_simulations_list:
        mc_price = monte_carlo_option_price(S0, K, T, r, sigma, num_simulations, num_steps)
        mc_prices.append(mc_price)
        print(f"  Simulations: {num_simulations:<7} | MC Price: {mc_price:.4f}")

    # Calculate the Black-Scholes price for comparison
    bs_price = black_scholes_call(S0, K, T, r, sigma)
    print(f"\nBlack-Scholes Price: {bs_price:.4f}")

    # Plotting the convergence (optional)
    plt.figure(figsize=(8, 5))
    plt.plot(num_simulations_list, mc_prices, marker='o', linestyle='-', label='MC Estimate')
    plt.axhline(y=bs_price, color='r', linestyle='--', label=f'Black-Scholes ({bs_price:.4f})')
    plt.xscale('log')
    plt.xlabel("Number of Simulations")
    plt.ylabel("Option Price Estimate")
    plt.title("Monte Carlo Option Pricing Convergence")
    plt.legend()
    plt.grid(True)
    plt.show()
