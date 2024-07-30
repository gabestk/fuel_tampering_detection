import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def read_json(file_name):
    with open(file_name, 'r') as file:
        return json.load(file)

def calculate_metrics(station_analysis):
    global_metrics = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0, 'precision': 0, 'recall': 0, 'accuracy': 0}

    for station_id, metrics in station_analysis.items():
        TP = metrics['true_positive']
        FP = metrics['false_positive']
        FN = metrics['false_negative']
        TN = metrics['true_negative']
        
        global_metrics['TP'] += TP
        global_metrics['FP'] += FP
        global_metrics['FN'] += FN
        global_metrics['TN'] += TN
        
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
        
        station_analysis[station_id]['precision'] = precision
        station_analysis[station_id]['recall'] = recall
        station_analysis[station_id]['accuracy'] = accuracy
    
    global_metrics['precision'] = global_metrics['TP'] / (global_metrics['TP'] + global_metrics['FP']) if (global_metrics['TP'] + global_metrics['FP']) > 0 else 0
    global_metrics['recall'] = global_metrics['TP'] / (global_metrics['TP'] + global_metrics['FN']) if (global_metrics['TP'] + global_metrics['FN']) > 0 else 0
    global_metrics['accuracy'] = (global_metrics['TP'] + global_metrics['TN']) / (global_metrics['TP'] + global_metrics['TN'] + global_metrics['FP'] + global_metrics['FN']) if (global_metrics['TP'] + global_metrics['TN'] + global_metrics['FP'] + global_metrics['FN']) > 0 else 0
    
    return global_metrics

def analyze_data(data):
    station_analysis = {}
    station_data = {}

    for entry in data:
        station_id = entry['station_id']

        if station_id not in station_analysis:
            station_analysis[station_id] = {
                'true_positive': 0,
                'false_positive': 0,
                'true_negative': 0,
                'false_negative': 0,
                'precision': 0,
                'recall': 0,
                'accuracy': 0
            }

        if entry['uncertainty_bool'] and entry['car_uncertainty_bool']:
            if entry['fraud_test']:
                if entry['fraud']:
                    station_analysis[station_id]['true_positive'] += 1
                else:
                    station_analysis[station_id]['false_positive'] += 1
            else:
                if entry['fraud']:
                    station_analysis[station_id]['false_negative'] += 1
                else:
                    station_analysis[station_id]['true_negative'] += 1

        if station_id not in station_data:
            station_data[station_id] = []
        station_data[station_id].append(entry)

    global_metrics = calculate_metrics(station_analysis)
    return station_analysis, station_data, global_metrics

def write_json(file_name, data):
    with open(file_name, 'w') as file: 
        json.dump(data, file, indent=4)

def plot_confusion_matrix(analysis, title, file_name):
    data = []
    for key, values in analysis.items():
        data.append([
            values['true_positive'],
            values['false_positive'],
            values['false_negative'],
            values['true_negative']
        ])

    df_cm = pd.DataFrame(data, index=analysis.keys(), columns=['TP', 'FP', 'FN', 'TN'])
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="YlGnBu")
    plt.title(title)
    plt.savefig(file_name)
    plt.show()

def plot_global_metrics(global_metrics, title, file_name):
    data = [
        [global_metrics['precision'], global_metrics['recall'], global_metrics['accuracy']]
    ]

    df_metrics = pd.DataFrame(data, index=['Global'], columns=['Precision', 'Recall', 'Accuracy'])
    plt.figure(figsize=(5, 3))
    sns.heatmap(df_metrics, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title(title)
    plt.savefig(file_name)
    plt.show()

def plot_station_metrics(analysis, title, file_name):
    data = []
    for key, values in analysis.items():
        data.append([
            values['precision'],
            values['recall'],
            values['accuracy']
        ])

    df_metrics = pd.DataFrame(data, index=analysis.keys(), columns=['Precision', 'Recall', 'Accuracy'])
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_metrics, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title(title)
    plt.savefig(file_name)
    plt.show()

def main():
    base_input_file = 'postos/refuel_data_{}_4.json'
    output_station_file = 'station_analysis_result.json'

    combined_station_analysis = {}
    combined_global_metrics = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0, 'precision': 0, 'recall': 0, 'accuracy': 0}
    num_files = 20

    for i in range(1, num_files + 1):
        input_file = base_input_file.format(i)
        data = read_json(input_file)
        station_analysis, station_data, global_metrics = analyze_data(data)
        
        for station_id, metrics in station_analysis.items():
            if station_id not in combined_station_analysis:
                combined_station_analysis[station_id] = {
                    'true_positive': 0,
                    'false_positive': 0,
                    'true_negative': 0,
                    'false_negative': 0,
                    'precision': 0,
                    'recall': 0,
                    'accuracy': 0
                }
            combined_station_analysis[station_id]['true_positive'] += metrics['true_positive']
            combined_station_analysis[station_id]['false_positive'] += metrics['false_positive']
            combined_station_analysis[station_id]['true_negative'] += metrics['true_negative']
            combined_station_analysis[station_id]['false_negative'] += metrics['false_negative']
        
        combined_global_metrics['TP'] += global_metrics['TP']
        combined_global_metrics['FP'] += global_metrics['FP']
        combined_global_metrics['FN'] += global_metrics['FN']
        combined_global_metrics['TN'] += global_metrics['TN']

    for station_id, metrics in combined_station_analysis.items():
        combined_station_analysis[station_id]['precision'] = metrics['true_positive'] / (metrics['true_positive'] + metrics['false_positive']) if (metrics['true_positive'] + metrics['false_positive']) > 0 else 0
        combined_station_analysis[station_id]['recall'] = metrics['true_positive'] / (metrics['true_positive'] + metrics['false_negative']) if (metrics['true_positive'] + metrics['false_negative']) > 0 else 0
        combined_station_analysis[station_id]['accuracy'] = (metrics['true_positive'] + metrics['true_negative']) / (metrics['true_positive'] + metrics['true_negative'] + metrics['false_positive'] + metrics['false_negative']) if (metrics['true_positive'] + metrics['true_negative'] + metrics['false_positive'] + metrics['false_negative']) > 0 else 0

    combined_global_metrics['precision'] = combined_global_metrics['TP'] / (combined_global_metrics['TP'] + combined_global_metrics['FP']) if (combined_global_metrics['TP'] + combined_global_metrics['FP']) > 0 else 0
    combined_global_metrics['recall'] = combined_global_metrics['TP'] / (combined_global_metrics['TP'] + combined_global_metrics['FN']) if (combined_global_metrics['TP'] + combined_global_metrics['FN']) > 0 else 0
    combined_global_metrics['accuracy'] = (combined_global_metrics['TP'] + combined_global_metrics['TN']) / (combined_global_metrics['TP'] + combined_global_metrics['TN'] + combined_global_metrics['FP'] + combined_global_metrics['FN']) if (combined_global_metrics['TP'] + combined_global_metrics['TN'] + combined_global_metrics['FP'] + combined_global_metrics['FN']) > 0 else 0

    write_json(output_station_file, combined_station_analysis)

    plot_confusion_matrix(combined_station_analysis, 'Confusion Matrix by Station ID', 'station_confusion_matrix.png')
    plot_global_metrics(combined_global_metrics, 'Global Precision, Recall, and Accuracy', 'global_precision_recall_accuracy_matrix.png')
    plot_station_metrics(combined_station_analysis, 'Precision, Recall, and Accuracy by Station ID', 'station_precision_recall_accuracy_matrix.png')

if __name__ == '__main__':
    main()
