import numpy as np
from scipy.stats import norm # Keep for black_scholes comparison if needed later

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

def black_scholes_call(S, K, T, r, sigma):
    """Calculates European call option price using Black-Scholes formula."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = (S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    return call_price

# --- Removed example parameters and simulation execution ---