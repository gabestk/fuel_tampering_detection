import json
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_json('vehicle_fueling_file.json')

features = ['vehicle_factory_error', 'vehicle_random_error', 'real_refuel_liters', 'refuel_amount_liters', 'expected_refuel_liters', 'expected_fuel_percentage'
            ,'real_fuel_percentage', 'real_fuel_percentage']
target = 'fraud'

X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier()
param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [None, 5, 10], 'min_samples_split': [2, 5], 'min_samples_leaf': [1, 2]}
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)

y_pred = grid_search.predict(X_test_scaled)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d')
plt.title(f'Confusion matrix error fixed 5%  fraud 8% postos 10%')
plt.savefig('Confusion_matrix_error_fixed_5%_fraud_8%_postos_10%.png')

report = classification_report(y_test, y_pred, output_dict=True)
df_report = pd.DataFrame(report).transpose()

plt.figure(figsize=(10, 7))
sns.heatmap(df_report.iloc[:-1, :-1], annot=True)
plt.title(f'Classification Report error fixed 5%  fraud 8% postos 10%')
plt.savefig('classification_report_error_fixed_5%_fraud_8%_postos_10%.png')