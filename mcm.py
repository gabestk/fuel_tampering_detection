import json
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_json('vehicle_fueling_file.json')

features = ['vehicle_random_error','vehicle_factory_error','real_refuel_liters', 'refuel_amount_liters',]
target = 'fraud'

X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.3)

noise = np.random.normal(0,1,X_train.shape)
X_train_noise = X_train + noise

noise_test = np.random.normal(0,1,X_test.shape)
X_test_noise = X_test + noise_test

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d')
plt.title(f'Confusion matrix gauss error fixed fraud 8% ')
plt.savefig('Confusion_matrix_gauss_error_fraud8%.png')

report = classification_report(y_test, y_pred, output_dict=True, zero_division=1)
df_report = pd.DataFrame(report).transpose()

plt.figure(figsize=(10, 7))
sns.heatmap(df_report.iloc[:-1, :-1], annot=True)
plt.title(f'Classification Report gauss error fraud at 8% ')
plt.savefig('classification_report_gauss_error_fraud8%.png')
