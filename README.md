# Vehicle Fueling Fraud Simulation

This repository includes scripts developed to simulate the vehicle refueling process at gas stations and analyze potential fraud in this process.

## Scripts

### 1. `analysis.py`

This script performs data analysis on simulated vehicle refueling fraud using classification techniques.

#### Libraries Used:
- `json`
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`

#### Features:
- Collects data from vehicle refueling simulations.
- Conducts 30 iterations of classification using logistic regression.
- Generates average confusion matrices and average classification reports.

### 2. `simulation.py`

This script simulates the vehicle refueling process at gas stations, including the introduction of potential fraud into the system.

#### Libraries Used:
- `traci`
- `random`
- `json`
- `datetime`
- `matplotlib`
- `numpy`

#### Features:
- Simulates vehicle refueling at specific stations.
- Introduces random errors in the refueling process to simulate fraud.
- Collects refueling data, including fraud information and fuel quantity.
- Generates JSON files containing refueling data for each simulation.

## How to Run:

### Requirements:
- Python 3.x
- Libraries listed in the scripts (`pip install "libraries used"`)
  
### SUMO Installation:
To install SUMO, follow these steps:
1. Download SUMO from the official website [here](https://sumo.dlr.de/docs/Downloads.php).
2. Choose the appropriate version for your operating system and download the installer or binaries.
3. Follow the installation instructions provided on the SUMO website.

### Steps:
1. Clone this repository.
2. Run `analysis.py` to perform the analysis on simulation data.
3. Run `simulation.py` to simulate the refueling process and collect data.
