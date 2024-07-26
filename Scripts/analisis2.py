import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def read_json(file_name):
    with open(file_name, 'r') as file:
        return json.load(file)

def analyze_data(data):
    station_analysis = {}
    station_data = {}

    for entry in data:
        station_id = entry['station_id']

        # Initialize analysis dictionary for each station_id
        if station_id not in station_analysis:
            station_analysis[station_id] = {
                'true_positive': 0,
                'false_positive': 0,
                'true_negative': 0,
                'false_negative': 0
            }

        # Analyze data
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

        # Collect data for each station
        if station_id not in station_data:
            station_data[station_id] = []
        station_data[station_id].append(entry)

    return station_analysis, station_data

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

def main():
    input_file = 'refuel_data.json'
    output_station_file = 'station_analysis_result.json'

    data = read_json(input_file)
    station_analysis, station_data = analyze_data(data)
    write_json(output_station_file, station_analysis)

    # Plot confusion matrices
    plot_confusion_matrix(station_analysis, 'Confusion Matrix by Station ID', 'station_confusion_matrix.png')

if __name__ == '__main__':
    main()