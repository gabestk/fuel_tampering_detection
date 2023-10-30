import json
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_json('vehicle_fueling_file_RandError_FrauFix8%.json')

features = ['vehicle_random_error','vehicle_factory_error','real_refuel_liters', 'refuel_amount_liters',]
target = 'fraud'

X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.3)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d')
plt.title(f'Confusion matrix random error fraud fixed 8%')
plt.savefig('Confusion_matrix_random_error_fraud_fixed_8%.png')

report = classification_report(y_test, y_pred, output_dict=True, zero_division=1)
df_report = pd.DataFrame(report).transpose()

plt.figure(figsize=(10, 7))
sns.heatmap(df_report.iloc[:-1, :-1], annot=True)
plt.title(f'Classification Report random error fraud fixed 8%')
plt.savefig('classification_report_random_error_fraud_fixed_8%.png')