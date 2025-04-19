# Learn Statistics with Python

This repository is designed as a learning resource for various statistical concepts, implemented in Python. It provides clear code examples and explanations to help understand fundamental techniques.

## Repository Structure

The repository is organized as follows:

```
├── docs/                 # Conceptual explanations, derivations (to be added)
├── examples/             # Runnable scripts demonstrating the algorithms
│   ├── mcmc_gibbs_example.py
│   ├── mcmc_metropolis_example.py
│   ├── monte_carlo_integral_example.py
│   ├── monte_carlo_option_pricing_example.py
│   └── monte_carlo_pi_example.py
├── src/                  # Core implementation of statistical algorithms
│   ├── __init__.py
│   ├── mcmc/             # Markov Chain Monte Carlo methods
│   │   ├── __init__.py
│   │   ├── gibbs_sampling.py
│   │   └── metropolis_hastings.py
│   └── monte_carlo/      # Monte Carlo methods
│       ├── __init__.py
│       ├── integral_estimation.py
│       ├── option_pricing.py
│       └── pi_estimation.py
├── tests/                # Unit tests for the algorithms (to be added)
├── requirements.txt      # Required Python packages
└── README.md             # This file
```

*   **`src/`**: Contains the core Python modules implementing the statistical algorithms. Each subdirectory focuses on a specific area (e.g., `monte_carlo`, `mcmc`).
*   **`examples/`**: Contains example scripts that import modules from `src/` and demonstrate how to use them. These scripts often include parameter settings and visualizations.
*   **`docs/`**: (Planned) Will contain more detailed explanations, mathematical background, and derivations related to the concepts implemented.
*   **`tests/`**: (Planned) Will contain unit tests to ensure the correctness of the code in `src/`.
*   **`requirements.txt`**: Lists the necessary Python libraries to run the code.
*   **`README.md`**: Provides an overview of the repository.

## Getting Started

### Prerequisites

*   Python 3.x
*   pip (Python package installer)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd learn_statistics
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```
3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

## Running the Examples

Navigate to the `examples/` directory and run any of the example scripts using Python:

```bash
cd examples

# Example: Run the Pi estimation simulation
python monte_carlo_pi_example.py

# Example: Run the Gibbs sampling simulation
python mcmc_gibbs_example.py

# ... and so on for other examples
```

The scripts will print output to the console, and some may generate plots to visualize the results.

## Contributing

This is a learning repository, and contributions are welcome! If you find errors, want to add new examples, improve explanations, or implement other statistical concepts, feel free to:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes.
4.  Add tests for new functionality (if applicable).
5.  Commit your changes (`git commit -am 'Add some feature'`).
6.  Push to the branch (`git push origin feature/your-feature-name`).
7.  Create a new Pull Request.

## Future Plans

*   Add detailed explanations and derivations in the `docs/` folder.
*   Implement unit tests in the `tests/` folder.
*   Add more statistical concepts and algorithms (e.g., Bayesian inference, other sampling methods, hypothesis testing).
*   Potentially add Jupyter Notebook examples for more interactive learning.
