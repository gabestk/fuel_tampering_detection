import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

def main():
    data = pd.read_json('vehicle_fueling_file.json')

    features = ['vehicle_factory_error', 'vehicle_random_error', 'real_refuel_liters', 'real_fuel_percentage', 'expected_fuel_percentage', 'expected_refuel_liters', 'refuel_amount_liters']
    target = 'fraud'

    X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2)

    smote = SMOTE()
    X_train, y_train = smote.fit_resample(X_train, y_train)

    X_train_noisy_error_fixed = X_train.copy()
    X_train_noisy_error_fixed['vehicle_factory_error'] = 1
    X_train_noisy_error_fixed['vehicle_random_error'] = 1.5

    X_test_noisy_error_fixed = X_test.copy()
    X_test_noisy_error_fixed['vehicle_factory_error'] = 1
    X_test_noisy_error_fixed['vehicle_random_error'] = 1.5

    scaler = StandardScaler()

    fraud_levels = [0.05, 0.075, 0.10, 0.125, 0.15]

    variant_data = {
        'original': (X_train, X_test),
        'error_fixed': (X_train_noisy_error_fixed, X_test_noisy_error_fixed)
    }

    for fraud_level in fraud_levels:
        X_train_noisy_fraud_level = X_train_noisy_error_fixed.copy()
        X_test_noisy_fraud_level = X_test_noisy_error_fixed.copy()
        X_train_noisy_fraud_level['refuel_amount_liters'] *= (1 - fraud_level)
        X_test_noisy_fraud_level['refuel_amount_liters'] *= (1 - fraud_level)
        variant_data[f'error_fixed_fraud_{int(fraud_level * 100)}'] = (X_train_noisy_fraud_level, X_test_noisy_fraud_level)

    plt.figure(figsize=(10, 10))

    for variant in variant_data.keys():
        X_train_variant, X_test_variant = variant_data[variant]

        if 'fraud' in variant:
            fraud_level = int(variant.split('_')[-1]) / 100
        else:
            fraud_level = 0

        X_train_scaled = scaler.fit_transform(X_train_variant)
        X_test_scaled = scaler.transform(X_test_variant)

        model = RandomForestClassifier()
        param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [None, 5, 10], 'min_samples_split': [2, 5], 'min_samples_leaf': [1, 2], 'bootstrap': [True, False]}
        grid_search = GridSearchCV(model, param_grid, cv=5)
        grid_search.fit(X_train_scaled, y_train)

        y_pred_prob = grid_search.predict_proba(X_test_scaled)[:,1]
        y_pred = grid_search.predict(X_test_scaled)

        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
        
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f'Fraud Level {fraud_level} (AUC = {roc_auc:.2f})')

        # Crie a matriz de confusão
        cm = confusion_matrix(y_test, y_pred)

        # Exiba a matriz de confusão
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d')
        plt.title(f'Confusion Matrix for {variant} (Fraud: {fraud_level * 100}%)')
        plt.savefig(f'confusion_matrix_{variant}_fraud_{int(fraud_level * 100)}.png')

        # Gere e exiba o relatório de classificação
        report = classification_report(y_test, y_pred, output_dict=True)
        df_report = pd.DataFrame(report).transpose()

        plt.figure(figsize=(10, 7))
        sns.heatmap(df_report.iloc[:-1, :-1], annot=True)
        plt.title(f'Classification Report for {variant} (Fraud: {fraud_level * 100}%)')
        plt.savefig(f'classification_report_{variant}_fraud_{int(fraud_level * 100)}.png')
        
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falso Positivo')
    plt.ylabel('Taxa de Verdadeiro Positivo')
    plt.title('Curvas ROC para cada etapa de fraude')
    plt.legend(loc="lower right")
    plt.savefig('Grafico.png')
    
if __name__ == '__main__':
    main()
