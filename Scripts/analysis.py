import json
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

#Colecting features for classification models
features = ['refuel_amount_liters','real_refuel_liters','real_fuel_percentage']
target = 'fraud'

#Average of all 30 confusion matrix and report for data analysis
cm_avg = np.zeros((2,2))
report_avg = pd.DataFrame()

#looping for classification each 30 times json files simulated using sumo
for i in range(1, 31):
    #Reading a data file generated from simulation
    data = pd.read_json(f'vehicle_fueling_file_40_station_({i}).json')

    X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.3)

    noise = np.random.normal(0,1,X_train.shape)
    X_train_noise = X_train + noise

    noise_test = np.random.normal(0,1,X_test.shape)
    X_test_noise = X_test + noise_test

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_noise)
    X_test_scaled = scaler.transform(X_test_noise)

    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    cm = confusion_matrix(y_test, y_pred)
    cm_avg += cm

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=1)
    df_report = pd.DataFrame(report).transpose()
    report_avg = report_avg.add(df_report, fill_value=0)

cm_avg /= 30
report_avg /= 30

plt.figure(figsize=(10, 7))
sns.heatmap(cm_avg, annot=True, fmt='.2f')
plt.title(f'Confusion matrix avg 40 station')
plt.savefig('Confusion_matrix_avg_40.station.png')

plt.figure(figsize=(10, 7))
sns.heatmap(report_avg.iloc[:-1, :-1], annot=True)
plt.title(f'Classification Report avg 40 station')
plt.savefig('classification_report_avg_40.station.png')
