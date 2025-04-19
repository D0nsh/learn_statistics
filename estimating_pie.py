import random
import math

def estimate_pi(num_points):
    """Estimates pi using the Monte Carlo method."""
    points_inside_circle = 0
    total_points = num_points

    for _ in range(num_points):
        # Generate random x, y coordinates between 0 and 1
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)

        # Check if the point is inside the quarter circle (distance from origin <= 1)
        # Using x^2 + y^2 <= 1^2 is equivalent and avoids sqrt
        if x**2 + y**2 <= 1:
            points_inside_circle += 1

    # Calculate the estimate for pi
    pi_estimate = 4 * (points_inside_circle / total_points)
    return pi_estimate

# --- Run the simulation ---
num_samples = 1000000 # Increase for better accuracy
estimated_pi = estimate_pi(num_samples)

print(f"Number of samples: {num_samples}")
print(f"Estimated value of pi: {estimated_pi}")
print(f"Actual value of pi: {math.pi}")
print(f"Error: {abs(estimated_pi - math.pi)}")