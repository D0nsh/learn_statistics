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

# --- Run the simulation ---
a = 0
b = 1
num_samples_integral = 10000000
estimated_integral = estimate_integral(g, a, b, num_samples_integral)
analytical_result = 1/3

print(f"Function: x^2")
print(f"Interval: [{a}, {b}]")
print(f"Number of samples: {num_samples_integral}")
print(f"Estimated integral: {estimated_integral}")
print(f"Analytical result: {analytical_result}")
print(f"Error: {abs(estimated_integral - analytical_result)}")