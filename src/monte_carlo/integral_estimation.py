import numpy as np # Using numpy for efficient array operations

def g(x):
    """The function we want to integrate."""
    return x**2

def estimate_integral(func, a, b, num_samples):
    """Estimates the definite integral of func from a to b."""
    # Generate random numbers uniformly distributed between a and b
    random_xs = np.random.uniform(a, b, num_samples)

    # Evaluate the function at these random points
    function_values = func(random_xs)

    # Calculate the sample mean of the function values
    sample_mean = np.mean(function_values)

    # Calculate the integral estimate
    integral_estimate = (b - a) * sample_mean
    return integral_estimate

# --- Removed simulation execution code ---