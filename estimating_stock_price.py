import numpy as np

def monte_carlo_option_price(S0, K, T, r, sigma, num_simulations, num_steps):
    """Estimates European call option price using Monte Carlo."""
    dt = T / num_steps
    discount_factor = np.exp(-r * T)

    # Pre-generate random numbers for efficiency
    # Z ~ N(0,1) -> shape (num_steps, num_simulations)
    Z = np.random.standard_normal((num_steps, num_simulations))

    # Initialize stock price array: shape (num_steps + 1, num_simulations)
    S = np.zeros((num_steps + 1, num_simulations))
    S[0, :] = S0

    # Simulate price paths
    for t in range(1, num_steps + 1):
        drift = (r - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * Z[t-1, :] # Use pre-generated Z for this step
        S[t, :] = S[t-1, :] * np.exp(drift + diffusion)

    # Get final stock prices (at time T)
    final_prices = S[-1, :] # Last row contains prices at T

    # Calculate payoffs for each simulation
    payoffs = np.maximum(final_prices - K, 0)

    # Calculate the average payoff
    average_payoff = np.mean(payoffs)

    # Discount the average payoff back to present value
    option_price = discount_factor * average_payoff

    return option_price

# --- Example Parameters ---
S0 = 100      # Initial stock price
K = 105       # Strike price
T = 1.0       # Time to expiry (1 year)
r = 0.05      # Risk-free rate (5%)
sigma = 0.2   # Volatility (20%)
N_sim = 100000 # Number of simulations (paths)
M_steps = 100  # Number of time steps

# --- Run the simulation ---
estimated_price = monte_carlo_option_price(S0, K, T, r, sigma, N_sim, M_steps)

# Compare with Black-Scholes analytical solution (optional)
from scipy.stats import norm
def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = (S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    return call_price

bs_price = black_scholes_call(S0, K, T, r, sigma)

print(f"Simulation Parameters:")
print(f"  S0={S0}, K={K}, T={T}, r={r}, sigma={sigma}")
print(f"  Num Simulations={N_sim}, Num Steps={M_steps}")
print(f"Monte Carlo Estimated Call Price: {estimated_price:.4f}")
print(f"Black-Scholes Analytical Price: {bs_price:.4f}")
print(f"Difference: {abs(estimated_price - bs_price):.4f}")