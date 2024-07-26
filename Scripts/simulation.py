import traci
import random
import json
import matplotlib.pyplot as plt
import numpy as np
import math

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

def write_json(sorted_fuel_data, filename):
    # Write a json file containing the fueling data related to all vehicles
    with open(filename, "w") as json_file:
        json.dump(sorted_fuel_data, json_file, indent=4, allow_nan=False)

def get_options():
    """Set some SUMO configurations
    """
    opt_parser = optparse.OptionParser()
    opt_parser.add_option("--nogui", action="store_true", default=True, help="run the commandline version of sumo")
    options, args = opt_parser.parse_args()
    return options

def get_distance(pos1, pos2):
    """Calcula a distância euclidiana entre duas posições."""
    return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

def get_edge_position(edge_id):
    """Obtém a posição da primeira lane de uma edge."""
    #lane_id = traci.edge.getLaneNumber(edge_id)
    
    lane_shape = traci.lane.getShape(f"{edge_id}_0")

    return lane_shape[0]  # Retorna a posição inicial da lane

def get_closest_edge(vehicle_id, edge_list):
    # Obter a posição atual do veículo
    vehicle_position = traci.vehicle.getPosition(vehicle_id)

    # Inicializar variáveis para armazenar a menor distância e a edge mais próxima
    min_distance = float('inf')
    closest_edge = None

    # Iterar sobre todas as edges na lista
    for edge_id in edge_list:
        # Obter a posição da edge
        edge_position = get_edge_position(edge_id)

        # Calcular a distância até a posição do veículo
        distance = get_distance(vehicle_position, edge_position)

        # Atualizar a menor distância e a edge mais próxima se necessário
        if distance < min_distance:
            min_distance = distance
            closest_edge = edge_id

    return closest_edge


stations = ['-101103849', '-1049729600','-125947335#0', '-125947335#1', '-125814597', '-125947334#2', '-125947334#1', '-125813719', '-125948402', '-125948399']

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
        
        #station = random.randint(0, len(stations)-1)
        closest_edge = get_closest_edge(id, stations)
        
        #route = len(traci.simulation.findRoute(veh_edge, stations[i]).edges)

        traci.vehicle.changeTarget(id, closest_edge)
        return closest_edge

def checkStation(id, combustivel):
    """Check if the vehicle is currently in the fuel station and needs to refuel

    Args:
        id (integer): vehicle's id
        combustivel (_type_): Amount (%) of vehicle's fuel

    Returns:
        boolean: Fueling status of the vehicle
    """
    if traci.vehicle.getRoadID(id) in stations and combustivel < 90:
        return True

houses = ['-968410233', '38333192#0', '-968410233', '38606676', '38655595', '-38655812', '1016398182#0', '-38691697', '-38691677', '125813710#0','968410229', '38591816', '38698804#1', '968410233',
        '-183715382', '958823492', '968410233', '-968410231', '-968410231', '-968410229', '-90746167#0', '-968410229', '90746172#1', '-38606818', '49713108', '-841705340#0', '968694318', '-38606931']

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

def calcular_incerteza_erro_padrao(erros):
    n = len(erros)
    media = sum(erros) / n
    variancia = sum((erro - media) ** 2 for erro in erros) / (n - 1)  # Use n-1 para amostra, n para população
    desvio_padrao = math.sqrt(variancia)
    return desvio_padrao

def run(sim_n, pump):
    """TraCI control loop"""
    try:
        # Definition of some useful variables
        
        fraud = 0  
        error_percentage = 0
        error_difference = 0

        # Fuel data list      
        fuel_data_list = []
        station_data_list= []

        # Fuel density for consumption calculation in liters
        fuel_density_mg_l = 740000

        # Vehicle fuel dictionary
        vehicle_info = {}
        station_info = {}

        # Fuel tank capacity
        tanque = 50
        tanque_truck = 495
        tanque_moto = 9
        tanque_bus = 350
        tanque_lv = 40
        tanque_hv = 60

        # Simulation Loop
        step = 0
        while traci.simulation.getMinExpectedNumber() > 0:
            # Advance a step in the simulation loop
            traci.simulationStep()

            # Obtain the time, in seconds, for each step
            delta_t = traci.simulation.getDeltaT()

            # Runs through all running vehicles
            for vehicle_id in traci.vehicle.getIDList():
                if vehicle_id == 100:
                    break
                
                if 0 <= int(vehicle_id) <= 49:
                    tanque = tanque
                    error_percentage = 5
                elif 50 <= int(vehicle_id) <= 69:
                    tanque = tanque_moto
                    error_percentage = 2
                elif 70 <= int(vehicle_id) <= 79:
                    tanque = tanque_hv
                    error_percentage = 7
                elif 80 <= int(vehicle_id) <= 82:
                    tanque = tanque_truck
                    error_percentage = 23
                elif 83 <= int(vehicle_id) <= 84:
                    tanque = tanque_bus
                    error_percentage = 18
                else:
                    tanque = tanque_lv
                    error_percentage = 10
                    
                #error = (error_percentage / 100) * tanque
                
                vehicle_factory_error = np.random.normal(0, (error_percentage/3)/2)
                vehicle_random_error = np.random.normal(vehicle_factory_error, (error_percentage/3)/2)
                
                # If not exist, create a new dictionary to keep consumption
                # steps from each vehicle
                if vehicle_id not in vehicle_info:
                    vehicle_info[vehicle_id] = {
                        "consumption_steps": [],
                        "refuel_info": [],
                        "refuel_error": [],
                        "uncertainty": [],
                        "uncertainty_bool": False,
                        "fuel_percentage": 100,
                        "expected_fuel_percentage": 100,
                        "expected_fuel_remaining": tanque,
                        "real_fuel_remaining": tanque,
                        "route_num": 0,
                        "isGoingToRefuel": False,
                        "fueling": False,
                        "fuel_station": "",
                        #"vehicle_factory_error": random.uniform(-error_percentage/2, error_percentage/2),
                        #"vehicle_random_error": random.uniform(-error_percentage/2, error_percentage/2),
                        "vehicle_factory_error": vehicle_factory_error,
                        "vehicle_random_error": vehicle_random_error,
                        "tank_capacity": tanque,
                    }

                # Get the consumption step
                fuel_consumption_mg_s = traci.vehicle.getFuelConsumption(vehicle_id)

                # Get the amount of fuel consumption
                fuel_consumption_mg = fuel_consumption_mg_s * delta_t

                # Convert fuel consumption to liters
                fuel_consumption_liters = fuel_consumption_mg / fuel_density_mg_l
                
                fuel_consumption_percentage = (fuel_consumption_liters / tanque) * 100

                # Add to consumption dictionary
                vehicle_info[vehicle_id]["consumption_steps"].append(fuel_consumption_liters)
                vehicle_info[vehicle_id]["expected_fuel_remaining"] -= fuel_consumption_liters
                vehicle_info[vehicle_id]["real_fuel_remaining"] -= fuel_consumption_liters
                vehicle_info[vehicle_id]["fuel_percentage"] -= fuel_consumption_percentage
                vehicle_info[vehicle_id]["expected_fuel_percentage"] -= fuel_consumption_percentage

                # Get the amount of fuel in the tank
                combus_percentage = vehicle_info[vehicle_id]["fuel_percentage"]

                expected_percentage = vehicle_info[vehicle_id]["expected_fuel_percentage"]
                
                goingToFueling = vehicle_info[vehicle_id]["isGoingToRefuel"]
                # If the vehicle needs to refuel, sent to a fuel station
                if combus_percentage < 90 and goingToFueling is False:
                    vehicle_info[vehicle_id]["isGoingToRefuel"] = True
                    fuelStation = sendFuel(vehicle_id, None)
                    vehicle_info[vehicle_id]["fuel_station"] = fuelStation
                    
                station_id = vehicle_info[vehicle_id]["fuel_station"]
                
                if goingToFueling:
                    sendFuel(vehicle_id, station_id)

                # Checks if the vehicle is currently in a fuel station and needs to refuel
                if checkStation(vehicle_id, combus_percentage):
                    vehicle_info[vehicle_id]["isGoingToRefuel"] = False
                    vehicle_info[vehicle_id]["fueling"] = True
                    sum_consumption = (sum(vehicle_info[vehicle_id]["consumption_steps"]))
                    vehicle_info[vehicle_id]["consumption_steps"].append(-sum_consumption)
                    rerouting(vehicle_id)
                    vehicle_info[vehicle_id]["route_num"] += 1
                    # Refuel a vehicle
                    if vehicle_info[vehicle_id]["fueling"] == True:
                        fraud_bool = False
                        fraud_percentage = 0
                        fraud = 0
                        vehicle_info[vehicle_id]["fueling"] = False
                        vehicle_factory_error = vehicle_info[vehicle_id]["vehicle_factory_error"] # Get the factory error from vehicle's fuel tank
                        #vehicle_random_error = random.uniform(-error_percentage/2, error_percentage/2) # Get the random error to simulate the moment of fueling, such as expansion, fuel moving, etc
                        vehicle_random_error = np.random.normal(vehicle_factory_error, (error_percentage/3)/2)
                        random_refuel = random.uniform(1, 3) / 100 * tanque
                        if pump == 1:
                            match vehicle_info[vehicle_id]["fuel_station"]: 
                                case '-1049729600':
                                    fraud_percentage = 8 # Get a random percentage fraud
                                    fraud = (fraud_percentage/100) * random_refuel # Calculate the fraud value based on the percentage
                                    fraud_bool = True       
                                case '-125948402':
                                    fraud_percentage = 8 # Get a random percentage fraud
                                    fraud = (fraud_percentage/100) * random_refuel # Calculate the fraud value based on the percentage
                                    fraud_bool = True
                                case '-125947335#0':
                                    fraud_percentage = 8 # Get a random percentage fraud
                                    fraud = (fraud_percentage/100) * random_refuel # Calculate the fraud value based on the percentage
                                    fraud_bool = True
                        elif pump == 2:
                            match vehicle_info[vehicle_id]["fuel_station"]: 
                                case '-1049729600':
                                    fraud_percentage = 8 # Get a random percentage fraud
                                    fraud = (fraud_percentage/100) * random_refuel # Calculate the fraud value based on the percentage
                                    fraud_bool = True       
                                case '-125948402':
                                    fraud_percentage = 8 # Get a random percentage fraud
                                    fraud = (fraud_percentage/100) * random_refuel # Calculate the fraud value based on the percentage
                                    fraud_bool = True
                                case '-125947335#0':
                                    fraud_percentage = 8 # Get a random percentage fraud
                                    fraud = (fraud_percentage/100) * random_refuel # Calculate the fraud value based on the percentage
                                    fraud_bool = True       
                                case '-125947335#1':
                                    fraud_percentage = 8 # Get a random percentage fraud
                                    fraud = (fraud_percentage/100) * random_refuel # Calculate the fraud value based on the percentage
                                    fraud_bool = True
                        elif pump == 3:
                            match vehicle_info[vehicle_id]["fuel_station"]: 
                                case '-1049729600':
                                    fraud_percentage = 8 # Get a random percentage fraud
                                    fraud = (fraud_percentage/100) * random_refuel # Calculate the fraud value based on the percentage
                                    fraud_bool = True       
                                case '-125948402':
                                    fraud_percentage = 8 # Get a random percentage fraud
                                    fraud = (fraud_percentage/100) * random_refuel # Calculate the fraud value based on the percentage
                                    fraud_bool = True
                                case '-125947335#0':
                                    fraud_percentage = 8 # Get a random percentage fraud
                                    fraud = (fraud_percentage/100) * random_refuel # Calculate the fraud value based on the percentage
                                    fraud_bool = True       
                                case '-125947335#1':
                                    fraud_percentage = 8 # Get a random percentage fraud
                                    fraud = (fraud_percentage/100) * random_refuel # Calculate the fraud value based on the percentage
                                    fraud_bool = True
                                case '-125814597':
                                    fraud_percentage = 8 # Get a random percentage fraud
                                    fraud = (fraud_percentage/100) * random_refuel # Calculate the fraud value based on the percentage
                                    fraud_bool = True
                        elif pump == 4:
                            match vehicle_info[vehicle_id]["fuel_station"]: 
                                case '-1049729600':
                                    fraud_percentage = 8 # Get a random percentage fraud
                                    fraud = (fraud_percentage/100) * random_refuel # Calculate the fraud value based on the percentage
                                    fraud_bool = True       
                                case '-125948402':
                                    fraud_percentage = 8 # Get a random percentage fraud
                                    fraud = (fraud_percentage/100) * random_refuel # Calculate the fraud value based on the percentage
                                    fraud_bool = True
                                case '-125947335#0':
                                    fraud_percentage = 8 # Get a random percentage fraud
                                    fraud = (fraud_percentage/100) * random_refuel # Calculate the fraud value based on the percentage
                                    fraud_bool = True       
                                case '-125814597':
                                    fraud_percentage = 8 # Get a random percentage fraud
                                    fraud = (fraud_percentage/100) * random_refuel # Calculate the fraud value based on the percentage
                                    fraud_bool = True
                                case '-125947335#1':
                                    fraud_percentage = 8 # Get a random percentage fraud
                                    fraud = (fraud_percentage/100) * random_refuel # Calculate the fraud value based on the percentage
                                    fraud_bool = True       
                                case '-125947334#2':
                                    fraud_percentage = 8 # Get a random percentage fraud
                                    fraud = (fraud_percentage/100) * random_refuel # Calculate the fraud value based on the percentage
                                    fraud_bool = True                                                    
                        real_refuel = ((vehicle_random_error/100) * random_refuel) + random_refuel - fraud # Add amount of real fuel
                        expected_refuel = random_refuel # Add amount of expected refuel
                        expected_percentage += (expected_refuel / tanque) * 100 # Get an expected percentage of fuel tank
                        combus_percentage += (real_refuel / tanque) * 100 # Get a real percentage of fuel tank
                        #vehicle_info[vehicle_id]["expected_fuel_remaining"] += expected_refuel
                        #vehicle_info[vehicle_id]["real_fuel_remaining"] += real_refuel
                        vehicle_info[vehicle_id]["fuel_percentage"] = combus_percentage
                        vehicle_info[vehicle_id]["expected_fuel_percentage"] = expected_percentage
                        refuel_error = (1 - (real_refuel/ expected_refuel)) * 100
                        
                        erros = vehicle_info[vehicle_id]["refuel_error"]    
                        uncertainty = vehicle_info[vehicle_id]["uncertainty"]
                        error_mean = 0
                        station_mean_error = 0
                        fraud_test = False
                        real_refuel2 = 0
                        janela = []
                        if len(erros) >= 2:
                            incerteza = calcular_incerteza_erro_padrao(erros)
                            uncertainty.append(incerteza)
                        if len(uncertainty) >= 10:
                            #for j in range(len(uncertainty) - 4):
                            janela = uncertainty[len(uncertainty) - 10 : len(uncertainty)]
                            if max(janela) - min(janela) < 0.5:
                                vehicle_info[vehicle_id]["uncertainty_bool"] = True
                                """if refuel_error > (sum(janela) / 10):
                                    fraud_test = True"""
                                len_error = len(vehicle_info[vehicle_id]["refuel_error"])
                                error_mean = sum(vehicle_info[vehicle_id]["refuel_error"])/len_error                   
                                real_refuel2 = (1 - (error_mean/100)) * real_refuel
                            else:
                                vehicle_info[vehicle_id]["uncertainty_bool"] = False

                        difference = real_refuel - expected_refuel
                                
                        #if int(vehicle_id) < 69:       
                        station_fraud = False     
                        if station_id not in station_info:
                            station_info[station_id] = {
                                "cont": 0,
                                "cont_bool": False,
                                "station_error": [],           
                                "uncertainty": [],
                                "janela": 0,
                                "mean_error": 0                                  
                            }                          
                                        
                        erros = station_info[station_id]["station_error"]

                        uncertainty = station_info[station_id]["uncertainty"]
                        station_fraud_test = False
                        station_uncertainty_bool = False
                        refuel_error = (1 - (real_refuel/ expected_refuel)) * 100
                        station_janela = []
                        if len(erros) >= 2:
                            station_mean_error = sum(erros)/len(erros)    
                            incerteza = calcular_incerteza_erro_padrao(erros)
                            uncertainty.append(incerteza)
                        if len(uncertainty) >= 10:
                            #for j in range(len(uncertainty) - 4):
                            station_janela = uncertainty[len(uncertainty) - 10 : len(uncertainty)]
                            if max(station_janela) - min(station_janela) < 1:
                                station_uncertainty_bool = True
                                if (vehicle_info[vehicle_id]["uncertainty_bool"] == True):
                                    station_info[station_id]["janela"] = sum(station_janela) / 10
                                    station_info[station_id]["mean_error"] = station_mean_error
                                    if (station_mean_error > 3):                        
                                        station_fraud_test = True
                                        if station_info[station_id]["cont_bool"] == False:
                                            station_info[station_id]["cont_bool"] = True
                                            station_info[station_id]["cont"] = len(station_info[station_id]["station_error"])
                                    else:
                                        station_fraud_test = False
                                            
                            else:
                                station_uncertainty_bool = False   
                            
                        # Vehicle refuel information tuple  
                        refuel_info = {
                            "vehicle_id": int(vehicle_id),
                            "station_id": station_id,
                            "fraud_test": fraud_test,
                            "uncertainty_bool": vehicle_info[vehicle_id]["uncertainty_bool"],
                            "uncertainty_value": janela,
                            "error_mean": error_mean,
                            "fraud": fraud_bool,
                            "vehicle_factory_error": round(vehicle_factory_error, 2),
                            "vehicle_random_error": round(vehicle_random_error, 2),
                            "fraud_percentage": round(fraud_percentage, 2),
                            "fraud_liters": round(fraud, 2),
                            "refuel_amount_liters": random_refuel,
                            "expected_refuel_liters": round(expected_refuel, 2),
                            "real_refuel_liters": round(real_refuel, 2),
                            "expected_fuel_percentage": round(expected_percentage, 2),
                            "real_fuel_percentage": round(combus_percentage, 2), 
                            "error_difference": refuel_error,
                            "tank_capacity": tanque,
                            "vehicle_type": traci.vehicle.getTypeID(vehicle_id),
                            "error_percentage": error_percentage,
                        }
                        
                        station_refuel = {
                            "vehicle_id": vehicle_id,
                            "station_id": station_id,
                            "fraud_test": station_fraud_test,
                            "fraud": fraud_bool,
                            "uncertainty_bool": station_uncertainty_bool,
                            "car_uncertainty_bool": vehicle_info[vehicle_id]["uncertainty_bool"],
                            "refuel_error": refuel_error,
                            "expected_refuel": expected_refuel,
                            "real_refuel": real_refuel,
                            "real_refuel2": real_refuel2,
                            "difference": difference,
                            "error_mean": station_mean_error,
                            "factory_error": vehicle_factory_error,
                            "random_error": vehicle_random_error
                        }
                        
                        vehicle_info[vehicle_id]["refuel_info"].append(refuel_info)                      
                        vehicle_info[vehicle_id]["refuel_error"].append(refuel_error)
                        #if int(vehicle_id) < 69:                                                                           
                        station_info[station_id]["station_error"].append(refuel_error) 
                        #write_json_to_file(station_refuel)
                        station_data_list.append(station_refuel)

            step += 1

            # Condition to stop the simulation loop
            if step > 86400:                    
                break
        
        # Update simulation refuel event dictionaries
        for vehicle_id, info in vehicle_info.items():
            fuel_data_list.extend(info["refuel_info"])

        sorted_fuel_data = sorted(fuel_data_list, key=lambda x: (x["vehicle_id"]))
        
        fuel_data_filename = f"vehicle({sim_n}).json"

        write_json(sorted_fuel_data, fuel_data_filename)

        write_json(station_data_list, f"refuel_data_{sim_n}_{pump}.json")
                
    finally:
        # Close the simulation
        traci.close()
        sys.stdout.flush()

# Loop for data collection iterations
for i in range(31, 151):
    if __name__ == "__main__":
        """Main entry point
        """
        options = get_options()

        # Check binary
        if options.nogui:
            sumoBinary = checkBinary('sumo')
        else:
            sumoBinary = checkBinary('sumo-gui')

        # Start traci connection and set the parameters
        traci.start([sumoBinary, "-c", "Sumo/osm.sumocfg", "--tripinfo-output",
                    "Sumo/tripinfo.xml", "--max-num-vehicles", "100", "--quit-on-end"])
        
        if i <= 30:
            pump = 1
        elif i <= 60: 
            pump = 2
        elif i <= 90: 
            pump = 3
        elif i <= 120: 
            pump = 4

        # Start the simulation loop
        run(i, pump)