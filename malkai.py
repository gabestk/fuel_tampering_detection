import traci
import random
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# Sumo configuration imports
import os
import sys
import optparse
from sumolib import checkBinary  # Checks for the binary in environ vars

# we need to import some python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

def malkai(sorted_data):
    # Write a json file containing the data from all the vehicles
    with open("vehicles_data2.json", "w") as json_file:
        json.dump(sorted_data, json_file, indent=4)

def gabriel(sorted_fuel_data):
    # Write a json file contining the fueling data related from all vehicles
    with open("vehicle_fueling_file.json", "w") as json_file:
        json.dump(sorted_fuel_data, json_file, indent=4, allow_nan=False)


def get_options():
    """Set some SUMO configurations

    """
    opt_parser = optparse.OptionParser()
    opt_parser.add_option("--nogui", action="store_true", default=False, help="run the commandline version of sumo")
    options, args = opt_parser.parse_args()
    print(type(options))
    return options


stations = ['-1049729600', '-183715364#2', '-183715380#2', '-125947335#0', '-125947335#1', '-125814597', '-125947334#2', '-125947334#1', '-125813719',
            '-183715382', '-183715367', '-162154275#0', '-101103849', '-125948402']


def sendFuel(id, fuelStation):
    """Send a vehicle for a fuel station

    Args:
        id (integer): vehicle's id
        fuelStation (string): Fuel station's edge

    Returns:
        string: Fuel station's edge
    """

    if fuelStation is not None:
        traci.vehicle.changeTarget(id, fuelStation)
    else:
        station = random.randint(0, len(stations)-1)
        traci.vehicle.changeTarget(id, stations[station])
        return stations[station]


def checkStation(id, combustivel):
    """Check if the vehicle is currently in the fuel station and need to refuel

    Args:
        id (integer): vehicle's id
        combustivel (_type_): Amount (%) of vehicle's fuel

    Returns:
        boolean: Fueling status of the vehicle
    """
    if traci.vehicle.getRoadID(id) in stations and combustivel < 80:
        return True


houses = ['-968410233', '38333192#0', '-968410233', '38606676', '38655595', '-38655812', '1016398182#0', '-38691697', '-38691677', '125813710#0', '-125948402', '968410229', '38591816', '38698804#1', '968410233',
        '-183715382', '-125813719', '958823492', '968410233', '-968410231', '-968410231', '-968410229', '-90746167#0', '-968410229', '90746172#1', '-38606818', '49713108', '-841705340#0', '968694318', '-38606931']


def rerouting(id):
    """Ensure that the vehicle restart to rerouting after refuel

    Args:
        id (integer): vehicle's id
    """
    edge = traci.vehicle.getRoadID(id)
    route = []
    while len(list(route)) == 0:
        rand_route = random.randint(0, len(houses)-1)
        route = traci.simulation.findRoute(edge, houses[rand_route]).edges

    if len(list(traci.simulation.findRoute(edge, houses[rand_route]).edges)) == 0:
        print(edge, houses[rand_route])

    traci.vehicle.setRoute(id, list(route))
    
def run():
    """TraCI control loop"""
    try:
        # Definition of some useful variables

        # Vehicles data list
        # vehicle_data_list = []
        # Fuel data list
        fraud_percentage = 0
        fraud = 0        
        fuel_data_list = []
        # Fuel density for consumption calculation in liters
        fuel_density_mg_l = 740000
        # Simulation start date
        data_atual = datetime.now()
        data_atual = datetime.now()
        # Vehicle fuel dictionary
        vehicle_info = {}
        # Fuel tank capacity
        tanque = 50

        # Simulation Loop
        step = 0
        while traci.simulation.getMinExpectedNumber() > 0:

            # Advance a step in the simulation loop
            traci.simulationStep()

            # Obtain the time, in seconds, for each step
            delta_t = traci.simulation.getDeltaT()

            # Runs through all running vehicles
            for vehicle_id in traci.vehicle.getIDList():
                # This is deprecated, but i still mantain just for security.
                # Assure that only the data from the vehicle firsts 30 vehicles
                # will be colected.
                if vehicle_id == 100:
                    break

                # If not exist, create a new dictionary to keep consumption
                # steps from the each vehicle
                if vehicle_id not in vehicle_info:
                    vehicle_info[vehicle_id] = {
                        "consumption_steps": [],
                        "refuel_info": [],
                        "fuel_percentage": 100,
                        "expected_fuel_percentage": 100,
                        "expected_fuel_remaining": tanque,
                        "real_fuel_remaining": tanque,
                        "route_num": 0,
                        "isGoingToRefuel": False,
                        "fueling": False,
                        "fuel_station": "",
                        "vehicle_factory_error": random.uniform(-2,2)
                    }
                
                """
                # Add a delta time step to the inicial data
                data_incrementada = data_atual + \
                    timedelta(seconds=step*delta_t)

                # Format the data (2023-07-21 14:39:05.278504)
                data_incrementada_formatada = data_incrementada.strftime("%Y-%m-%d %H:%M:%S.%f")
                """
                
                # Get the consumption step
                fuel_consumption_mg_s = traci.vehicle.getFuelConsumption(vehicle_id)

                # Get the amount of fuel consumption
                fuel_consumption_mg = fuel_consumption_mg_s * delta_t

                # Convert fuel consumption to liters
                fuel_consumption_liters = fuel_consumption_mg / fuel_density_mg_l
                
                fuel_consumption_percentage = (fuel_consumption_liters/tanque)*100

                # Add to consumption dictionary
                vehicle_info[vehicle_id]["consumption_steps"].append(fuel_consumption_liters)
                vehicle_info[vehicle_id]["expected_fuel_remaining"] -= fuel_consumption_liters
                vehicle_info[vehicle_id]["real_fuel_remaining"] -= fuel_consumption_liters
                vehicle_info[vehicle_id]["fuel_percentage"] -= fuel_consumption_percentage
                vehicle_info[vehicle_id]["expected_fuel_percentage"] -= fuel_consumption_percentage
                """
                # Get the geographic coordinates of the vehicle
                pos_x, pos_y = traci.vehicle.getPosition(vehicle_id)

                # Convert the coord to lat and long
                lat, lon = traci.simulation.convertGeo(pos_x, pos_y)
                """
                
                # Get the amount of fuel in the tank
                combus_percentage = vehicle_info[vehicle_id]["fuel_percentage"]
                expected_percentage = vehicle_info[vehicle_id]["expected_fuel_percentage"]
                
                goingToFueling = vehicle_info[vehicle_id]["isGoingToRefuel"]
                # If the vehicle needs to refuel, sent to a fuel station
                if combus_percentage < 80 and goingToFueling is False:
                    vehicle_info[vehicle_id]["isGoingToRefuel"] = True
                    fuelStation = sendFuel(vehicle_id, None)
                    vehicle_info[vehicle_id]["fuel_station"] = fuelStation

                if goingToFueling:
                    sendFuel(vehicle_id, vehicle_info[vehicle_id]["fuel_station"])

                # Checks if the vehicle is currently in a fuel station and need to refuel
                if checkStation(vehicle_id, combus_percentage):
                    vehicle_info[vehicle_id]["isGoingToRefuel"] = False
                    vehicle_info[vehicle_id]["fueling"] = True
                    vehicle_info[vehicle_id]["consumption_steps"].append(-(sum(vehicle_info[vehicle_id]["consumption_steps"])))
                    rerouting(vehicle_id)
                    vehicle_info[vehicle_id]["route_num"] += 1
                    # Refuel a vehicle
                    if vehicle_info[vehicle_id]["fueling"] == True:
                        fraud_bool = False
                        fraud_percentage = 0
                        fraud = 0
                        vehicle_info[vehicle_id]["fueling"] = False
                        vehicle_factory_error = vehicle_info[vehicle_id]["vehicle_factory_error"] # Get the factory error from vehicle's fuel tank
                        vehicle_random_error = random.uniform(-3,3) # Get a random error to simulate the moment of fueling, such as expansion, fuel moving, etc
                        random_refuel = random.randint(5,10) # Get a random refuel amount
                        match vehicle_info[vehicle_id]["fuel_station"]:
                            case '-1049729600':
                                fraud_percentage = random.randint(5,15) # Get a random percentage fraud
                                fraud = (fraud_percentage/100) * random_refuel # Calculate the fraud value based on the percentage
                                fraud_bool = True
                            case '-183715364#2':
                                fraud_percentage = random.randint(5,15) # Get a random percentage fraud
                                fraud = (fraud_percentage/100) * random_refuel # Calculate the fraud value based on the percentage   
                                fraud_bool = True                
                            case '-183715380#2':
                                fraud_percentage = random.randint(5,15) # Get a random percentage fraud
                                fraud = (fraud_percentage/100) * random_refuel # Calculate the fraud value based on the percentage  
                                fraud_bool = True                        
                            case '-125947335#0':
                                fraud_percentage = random.randint(5,15) # Get a random percentage fraud
                                fraud = (fraud_percentage/100) * random_refuel # Calculate the fraud value based on the percentage  
                                fraud_bool = True                         
                            case '-125947335#1':
                                fraud_percentage = random.randint(5,15) # Get a random percentage fraud
                                fraud = (fraud_percentage/100) * random_refuel # Calculate the fraud value based on the percentage   
                                fraud_bool = True                                                       
                        real_refuel = vehicle_factory_error + vehicle_random_error + random_refuel - fraud # Add amount of real fuel
                        expected_refuel = vehicle_factory_error + vehicle_random_error + random_refuel # Add amount of expected refuel
                        expected_percentage += (expected_refuel/tanque)*100 # Get a expected percentage of fuel tank
                        combus_percentage += (real_refuel/tanque)*100 # Get a real percentage of fuel tank
                        vehicle_info[vehicle_id]["expected_fuel_remaining"] += expected_refuel
                        vehicle_info[vehicle_id]["real_fuel_remaining"] += real_refuel
                        vehicle_info[vehicle_id]["fuel_percentage"] = combus_percentage
                        vehicle_info[vehicle_id]["expected_fuel_percentage"] = expected_percentage
                        # Vehicle refuel information tuple
                        refuel_info = {
                            "vehicle_id": int(vehicle_id),
                            "station_id": vehicle_info[vehicle_id]["fuel_station"],
                            "vehicle_factory_error": round(vehicle_factory_error,2),
                            "vehicle_random_error": round(vehicle_random_error,2),
                            "fraud_percentage": round(fraud_percentage,2),
                            "fraud_liters": round(fraud,2),
                            "refuel_amount_liters": random_refuel,
                            "expected_refuel_liters": round(expected_refuel,2),
                            "real_refuel_liters": round(real_refuel,2),
                            "expected_fuel_percentage": round(expected_percentage,2),
                            "real_fuel_percentage":round(combus_percentage,2),
                            "fraud": fraud_bool
                        }
                        vehicle_info[vehicle_id]["refuel_info"].append(refuel_info)

                """
                id_veh = vehicle_id + "_" + \
                    str(vehicle_info[vehicle_id]["route_num"])
                
                # Dict for vehicle data
                veh_data = {
                    "uservehicle": {
                        "id": id_veh,
                        "vehicle_data": {
                            "Tuple": [
                                {
                                    "pos": {
                                        "long": lon,
                                        "lat": lat
                                    }
                                },
                                {
                                    "time": data_incrementada_formatada
                                },
                                {
                                    "Combustivel": "{:f}%".format(((tanque - sum(vehicle_info[vehicle_id]["consumption_steps"])) / tanque) * 100)
                                }
                            ]
                        }
                    }
                }
                """
                
                # Append the data from the vehicle to a list
                # vehicle_data_list.append(veh_data)

            # Ensure that the vehicle continues rerouting after refueling
            # rerouting()

            step += 1

            # Condition to stop the simulation loop
            if step > 100000:
                break
    
        for vehicle_id, info in vehicle_info.items():
            fuel_data_list.extend(info["refuel_info"])

        sorted_fuel_data = sorted(fuel_data_list, key=lambda x: (x["vehicle_id"]))
        
        # Function that writes a json file containing the data from all the vehicle
        #malkai(sorted_data)
        gabriel(sorted_fuel_data)

        # Suponha que 'data' é um DataFrame do pandas contendo seus dados
        data = pd.read_json('vehicle_fueling_file.json')

        # Seleciona os recursos que serão usados para a previsão
        features = ['vehicle_factory_error', 'vehicle_random_error', 'real_refuel_liters', 'real_fuel_percentage']

        # Suponha que 'is_fraud' é a coluna que indica se houve fraude ou não
        target = 'fraud'

        # Divide os dados em conjuntos de treinamento e teste
        X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2)

        # Adiciona ruído aos dados de treinamento e teste
        X_train_noisy = X_train.copy()
        X_train_noisy['vehicle_factory_error'] += np.random.normal(0, 0.1, X_train['vehicle_factory_error'].shape)
        X_train_noisy['vehicle_random_error'] += np.random.normal(0, 0.1, X_train['vehicle_random_error'].shape)

        X_test_noisy = X_test.copy()
        X_test_noisy['vehicle_factory_error'] += np.random.normal(0, 0.1, X_test['vehicle_factory_error'].shape)
        X_test_noisy['vehicle_random_error'] += np.random.normal(0, 0.1, X_test['vehicle_random_error'].shape)
        
        # Normaliza os dados
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_noisy)
        X_test_scaled = scaler.transform(X_test_noisy)

        # Cria e treina o modelo com validação cruzada e ajuste de hiperparâmetros
        model = LogisticRegression()
        param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
        grid_search = GridSearchCV(model, param_grid, cv=5)
        grid_search.fit(X_train_scaled, y_train)

        # Faz previsões no conjunto de teste
        y_pred = grid_search.predict(X_test_scaled)
        
        report = classification_report(y_test, y_pred, output_dict=True)
        df_report = pd.DataFrame(report).transpose()

        # Cria a matriz de confusão
        cm = confusion_matrix(y_test, y_pred)

        # Exibe a matriz de confusão
        np.set_printoptions(suppress=True)
        sns.heatmap(cm, annot=True, fmt='d')
        plt.savefig('confusion_matrix.png')
        plt.show()
        sns.heatmap(df_report.iloc[:-1, :-1], annot=True)
        plt.savefig('classification_report.png')
        plt.show()
        

    finally:

        # Close the simulation
        traci.close()
        sys.stdout.flush()


if __name__ == "__main__":
    """Main entry point
    """
    options = get_options()

    # check binary
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    # Start traci connection and set the parameters
    traci.start([sumoBinary, "-c", "osm.sumocfg", "--tripinfo-output",
                "tripinfo.xml", "--max-num-vehicles", "100"])

    # Start the simulation loop
    run()
