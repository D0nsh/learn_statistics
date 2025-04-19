# Monte Carlo Methods

This document provides conceptual explanations for the Monte Carlo methods implemented in `src/monte_carlo`.

## Pi Estimation (`pi_estimation.py`)

**Concept:** The Monte Carlo method can estimate Pi (π) by simulating random points within a square that inscribes a quarter circle.

1.  Imagine a square with corners at (0,0), (1,0), (1,1), and (0,1).
2.  Imagine a quarter circle centered at the origin (0,0) with a radius of 1, fitting perfectly within this square.
3.  The area of the square is 1² = 1.
4.  The area of the full circle is πr² = π(1)² = π. The area of the quarter circle is π/4.
5.  Generate a large number of random points (x, y) where both x and y are between 0 and 1. These points fall uniformly within the square.
6.  Count how many of these points fall *inside* the quarter circle. A point (x, y) is inside the circle if its distance from the origin is less than or equal to the radius (1). This can be checked using the equation: x² + y² ≤ 1.
7.  The ratio of (points inside the circle) / (total points generated) approximates the ratio of the areas: (Area of quarter circle) / (Area of square) = (π/4) / 1 = π/4.
8.  Therefore, we can estimate Pi by calculating: π ≈ 4 * (points inside circle) / (total points).

**Implementation:** The `estimate_pi` function implements this by generating `num_points` random (x, y) pairs and checking the condition x² + y² ≤ 1 to count points inside the circle, finally applying the ratio.

## Integral Estimation (`integral_estimation.py`)

**Concept:** Monte Carlo integration estimates the definite integral of a function `f(x)` from `a` to `b`.

1.  The definite integral ∫[a, b] f(x) dx represents the area under the curve of f(x) between x=a and x=b.
2.  The mean value theorem for integrals states that ∫[a, b] f(x) dx = (b - a) * avg(f(x)), where avg(f(x)) is the average value of the function over the interval [a, b].
3.  Monte Carlo integration approximates this average value by sampling.
4.  Generate a large number (`num_samples`) of random points `x_i` uniformly distributed within the interval [a, b].
5.  Evaluate the function at each random point: `f(x_i)`.
6.  Calculate the sample mean (average) of these function values: `mean_f = (1 / num_samples) * Σ f(x_i)`.
7.  The integral estimate is then: `(b - a) * mean_f`.

**Implementation:** The `estimate_integral` function takes the function `func`, bounds `a` and `b`, and `num_samples`. It generates uniform random numbers in [a, b], evaluates `func` at these points, computes the mean, and multiplies by the interval width (b - a).

## Option Pricing (`option_pricing.py`)

**Concept:** Monte Carlo methods can estimate the price of financial derivatives, like European options, by simulating the potential future paths of the underlying asset (e.g., a stock price).

1.  **Geometric Brownian Motion (GBM):** Stock prices are often modeled as following a random process called GBM. The change in stock price `dS` over a small time step `dt` is modeled as: `dS = S * (μ dt + σ dW)`, where `S` is the stock price, `μ` is the expected return (drift, often the risk-free rate `r` in risk-neutral pricing), `σ` is the volatility, and `dW` is a random term drawn from a normal distribution representing market randomness.
2.  **Simulation:** To estimate the price at a future time `T`, we simulate many possible price paths from the current time (t=0) to `T`. This is done by dividing `T` into many small time steps `dt` and iteratively calculating the price at each step using the discretized GBM formula: `S(t+dt) = S(t) * exp((r - 0.5 * σ²) * dt + σ * sqrt(dt) * Z)`, where `Z` is a random sample from a standard normal distribution.
3.  **Payoff Calculation:** For each simulated path, we determine the stock price at maturity `S(T)`. The payoff of a European call option at maturity is `max(S(T) - K, 0)`, where `K` is the strike price.
4.  **Averaging:** We calculate the payoff for each of the `num_simulations` paths.
5.  **Discounting:** The expected payoff at maturity needs to be discounted back to its present value using the risk-free rate `r`. The discount factor is `exp(-r * T)`.
6.  **Option Price Estimate:** The Monte Carlo estimate of the option price is the average of all simulated payoffs, discounted back to the present: `Price ≈ exp(-r * T) * mean(payoffs)`.

**Implementation:** The `monte_carlo_option_price` function simulates `num_simulations` price paths using the GBM formula over `num_steps`. It calculates the payoff for each path at time `T`, computes the average payoff, and discounts it to get the estimated price. It also includes the `black_scholes_call` function for comparison, which provides an analytical solution for European call options under the same assumptions.
