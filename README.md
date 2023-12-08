# Fuel Tampering Detection

This repository includes scripts developed to simulate the vehicle refueling process at fuel dispensers and analyze potential fraud in this process.

Scripts
1. classification_analysis.py
This script performs data analysis on simulated vehicle refueling fraud using classification techniques.

Libraries Used:
json
pandas
numpy
scikit-learn
matplotlib
seaborn
Features:
Collects data from vehicle refueling simulations.
Conducts 30 iterations of classification using logistic regression.
Generates average confusion matrices and average classification reports.
Plots and saves average confusion matrices and classification reports.
2. simulation_fraud.py
This script simulates the vehicle refueling process at gas stations, including the introduction of potential fraud into the system.

Libraries Used:
traci
random
json
datetime
matplotlib
numpy
Features:
Simulates vehicle refueling at specific stations.
Introduces random errors in the refueling process to simulate fraud.
Collects refueling data, including fraud information and fuel quantity.
Generates JSON files containing refueling data for each simulation.
How to Run:
Requirements:
Python 3.x
Libraries listed in the scripts (pip install -r requirements.txt can be used)
Steps:
Clone this repository.
Run classification_analysis.py to perform the analysis on simulation data.
Run simulation_fraud.py to simulate the refueling process and collect data.
