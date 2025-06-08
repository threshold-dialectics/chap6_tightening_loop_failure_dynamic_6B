## Getting Started

Follow these steps to set up your environment and run the simulations.

### Prerequisites

-   Python 3.9 or newer.
-   "pip" for installing packages.
-   "git" for cloning the repository.

### Installation

1.  **Clone the repository:**
    """bash
    git clone https://github.com/threshold-dialectics/book-simulations.git
    cd book-simulations
    """

2.  **Create a virtual environment (recommended):**
    """bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    """

3.  **Install the required packages:**
    """bash
    pip install -r requirements.txt
    """

## How to Use This Repository

This repository is designed to be a hands-on companion to the book.

### Running a Chapter's Experiment

1.  Navigate to the directory for the chapter you are interested in. For example, to run the "Tightening Loop" experiment from Chapter 5:
    """bash
    cd chap5_energetics
    """
2.  Run the main Python script for that experiment. The script names are generally descriptive.
    """bash
    python run_experiment_5C.py
    """
3.  The script will typically run the simulation, perform statistical analysis, print a summary to the console, and generate and save the relevant plots in the "Images/" subdirectory.


## Core Concepts

A brief overview of the key concepts you will see implemented in the code.

### The Three Adaptive Levers

-   **Perception Gain ($\gLever$):** The system's sensitivity or vigilance.
-   **Policy Precision ($\betaLever$):** The system's rigidity or commitment to a specific strategy (exploitation vs. exploration).
-   **Energetic Slack ($\FEcrit$):** The system's buffer of available resources (energy, capital, etc.).

### The Tolerance Sheet

-   **$\ThetaT = C \cdot \gLever^{\wOne} \betaLever^{\wTwo} \FEcrit^{\wThree}$**: A dynamic boundary representing the maximum systemic strain a system can withstand given its current lever configuration. Collapse occurs when strain exceeds tolerance.

### The Core Diagnostics

-   **Speed Index ($\SpeedIndex$):** The joint rate of change of $\betaLever$ and $\FEcrit$. Measures the *magnitude* of structural drift.
-   **Couple Index ($\CoupleIndex$):** The correlation between the velocities of $\betaLever$ and $\FEcrit$. Measures the *coordination* and synchrony of structural drift.

## Contributing

We welcome contributions from the community! Whether it's reporting a bug, suggesting an improvement, or contributing new experimental code, your input is valuable. Please feel free to open an issue or submit a pull request.

When contributing, please try to follow the existing coding style and repository structure. Ensure that any new experimental code is well-documented and includes a way to reproduce the key findings.

## Citation

If you use or refer to this code or the concepts from Threshold Dialectics, please cite the accompanying book:

@book{pond2025threshold,
  author    = {Axel Pond},
  title     = {Threshold Dialectics: Understanding Complex Systems and Enabling Active Robustness},
  year      = {2025},
  isbn      = {978-82-693862-2-6},
  publisher = {Amazon Kindle Direct Publishing},
  url       = {https://www.thresholddialectics.com},
  note      = {Code repository: \url{https://github.com/threshold-dialectics}}
}

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.