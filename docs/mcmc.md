# Markov Chain Monte Carlo (MCMC) Methods

This document provides conceptual explanations for the MCMC methods implemented in `src/mcmc`.

MCMC methods are a class of algorithms used to sample from a probability distribution, especially when direct sampling is difficult. They work by constructing a Markov chain whose stationary distribution is the desired target distribution. After a "burn-in" period, the states of the chain can be used as samples from the target distribution.

## Gibbs Sampling (`gibbs_sampling.py`)

**Concept:** Gibbs sampling is an MCMC algorithm particularly useful for sampling from multivariate distributions (distributions with multiple variables) when the *conditional distributions* are known and easy to sample from.

1.  **Target Distribution:** We want to sample from a joint distribution P(x₁, x₂, ..., xₙ). Let **x** = (x₁, ..., xₙ).
2.  **Conditional Distributions:** The core idea relies on being able to sample from the conditional distribution of each variable given the current values of all *other* variables: P(xᵢ | x₁, ..., xᵢ₋₁, xᵢ₊₁, ..., xₙ).
3.  **Algorithm:**
    *   Start with an initial state **x**⁽⁰⁾ = (x₁⁽⁰⁾, ..., xₙ⁽⁰⁾).
    *   For each iteration `t` (from 1 to `num_samples`):
        *   Sample x₁⁽ᵗ⁺¹⁾ from P(x₁ | x₂⁽ᵗ⁾, x₃⁽ᵗ⁾, ..., xₙ⁽ᵗ⁾)
        *   Sample x₂⁽ᵗ⁺¹⁾ from P(x₂ | x₁⁽ᵗ⁺¹⁾, x₃⁽ᵗ⁾, ..., xₙ⁽ᵗ⁾)  *(Note: uses the newly sampled x₁)*
        *   Sample x₃⁽ᵗ⁺¹⁾ from P(x₃ | x₁⁽ᵗ⁺¹⁾, x₂⁽ᵗ⁺¹⁾, ..., xₙ⁽ᵗ⁾)  *(Uses newly sampled x₁ and x₂)*
        *   ... continue for all variables up to xₙ ...
        *   Sample xₙ⁽ᵗ⁺¹⁾ from P(xₙ | x₁⁽ᵗ⁺¹⁾, x₂⁽ᵗ⁺¹⁾, ..., xₙ₋₁⁽ᵗ⁺¹⁾)
    *   The resulting vector **x**⁽ᵗ⁺¹⁾ = (x₁⁽ᵗ⁺¹⁾, ..., xₙ⁽ᵗ⁺¹⁾) is the next state in the Markov chain.
4.  **Burn-in:** The initial samples (e.g., the first `burn_in` samples) are discarded because the chain may not have yet converged to the stationary (target) distribution.
5.  **Samples:** The samples collected after the burn-in period approximate the target joint distribution.

**Implementation:** The `gibbs_sampler_bivariate_normal` function implements this for a 2D (bivariate) normal distribution. For a bivariate normal, the conditional distributions P(x₁|x₂) and P(x₂|x₁) are themselves normal distributions whose parameters depend on the means, standard deviations, and correlation of the joint distribution, and the current value of the *other* variable. The function iteratively samples from these conditional normal distributions.

## Metropolis-Hastings Algorithm (`metropolis_hastings.py`)

**Concept:** The Metropolis-Hastings algorithm is a more general MCMC method for sampling from a distribution P(x) when we can evaluate a function proportional to the probability density, f(x) (where f(x) = C * P(x), and C is an often unknown normalization constant), but direct sampling is hard.

1.  **Target Density:** We want to sample from P(x), but we only know f(x) ∝ P(x).
2.  **Proposal Distribution:** We need a proposal distribution Q(x' | x) which suggests a new candidate state x' given the current state x. A common choice is a symmetric distribution, like a normal distribution centered at the current state: Q(x' | x) = Normal(x, σ²). In this case, the algorithm simplifies to the Metropolis algorithm.
3.  **Algorithm:**
    *   Start with an initial state x⁽⁰⁾.
    *   For each iteration `t` (from 1 to `num_samples`):
        *   **Propose:** Sample a candidate state x' from the proposal distribution Q(x' | x⁽ᵗ⁾).
        *   **Calculate Acceptance Ratio (α):** Compute the ratio of the target density at the proposed and current states: α = f(x') / f(x⁽ᵗ⁾). Since P(x) = f(x)/C, this is equivalent to α = P(x') / P(x⁽ᵗ⁾). If the proposal distribution Q is not symmetric (i.e., Q(x'|x) ≠ Q(x|x')), the ratio needs to be adjusted: α = [P(x') * Q(x⁽ᵗ⁾|x')] / [P(x⁽ᵗ⁾) * Q(x'|x⁽ᵗ⁾)]. (This is the Hastings correction).
        *   **Accept or Reject:** Generate a random number `u` from Uniform(0, 1).
            *   If `u ≤ min(α, 1)`, accept the proposal: set x⁽ᵗ⁺¹⁾ = x'.
            *   If `u > min(α, 1)`, reject the proposal: set x⁽ᵗ⁾ = x⁽ᵗ⁾ (the state remains the same).
4.  **Burn-in:** Discard the initial `burn_in` samples.
5.  **Samples:** The states x⁽ᵗ⁾ collected after burn-in are samples from the target distribution P(x).

**Implementation:** The `metropolis_sampler` function implements the (symmetric) Metropolis algorithm. It takes the unnormalized target density function `target_func_unnorm` (our f(x)), uses a normal proposal distribution (`np.random.normal(current_x, proposal_std)`), calculates the acceptance ratio `f(proposed_x) / f(current_x)`, and performs the accept/reject step. The example uses `f(x) = exp(-x⁴)`. The calculation is often done using logarithms (`log(f(x')) - log(f(x))`) for numerical stability, especially when density values are very small.
