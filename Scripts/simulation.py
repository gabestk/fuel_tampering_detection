import traci
import random
import json
import matplotlib.pyplot as plt
import numpy as np

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

def fuel_data(sorted_fuel_data, filename):
    # Write a json file contining the fueling data related from all vehicles
    with open(filename, "w") as json_file:
        json.dump(sorted_fuel_data, json_file, indent=4, allow_nan=False)


def get_options():
    """Set some SUMO configurations

    """
    opt_parser = optparse.OptionParser()
    opt_parser.add_option("--nogui", action="store_true", default=True, help="run the commandline version of sumo")
    options, args = opt_parser.parse_args()
    print(type(options))
    return options


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
    if traci.vehicle.getRoadID(id) in stations and combustivel < 50:
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
    
def generate_random_error():
    """
    generate random error in normal distribution
    """
    mu = 0 
    sigma = 2.6 
    return np.random.normal(mu, sigma)

def run(sim_n):
    """TraCI control loop"""
    try:
        # Definition of some useful variables
        fraud_percentage = 0
        fraud = 0  
        # Fuel data list      
        fuel_data_list = []
        # Fuel density for consumption calculation in liters
        fuel_density_mg_l = 740000
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
                        "vehicle_factory_error": 1.5
                    }               
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

                # Get the amount of fuel in the tank
                combus_percentage = vehicle_info[vehicle_id]["fuel_percentage"]
                expected_percentage = vehicle_info[vehicle_id]["expected_fuel_percentage"]
                
                goingToFueling = vehicle_info[vehicle_id]["isGoingToRefuel"]
                # If the vehicle needs to refuel, sent to a fuel station
                if combus_percentage < 50 and goingToFueling is False:
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
                        vehicle_random_error = 1 # Get a random error to simulate the moment of fueling, such as expansion, fuel moving, etc
                        random_refuel = random.randint(5,10) # Get a random refuel amount
                        match vehicle_info[vehicle_id]["fuel_station"]: 
                            case '-101103849':
                                fraud_percentage = 8 # Get a random percentage fraud
                                fraud = (fraud_percentage/100) * random_refuel # Calculate the fraud value based on the percentage
                                fraud_bool = True       
                            case '-1049729600':
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

            step += 1

            # Condition to stop the simulation loop
            if step > 86400:                    
                break
        
        #Update simulation refuel event dicionaries
        for vehicle_id, info in vehicle_info.items():
            fuel_data_list.extend(info["refuel_info"])

        sorted_fuel_data = sorted(fuel_data_list, key=lambda x: (x["vehicle_id"]))
        
        fuel_data_filename = f"vehicle_fueling_file_50_station_({sim_n}).json"
        fuel_data(sorted_fuel_data, fuel_data_filename)
        
                
    finally:

        # Close the simulation
        traci.close()
        sys.stdout.flush()
# Loop for 30 data collection iterations
for i in range(1, 31):
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
        traci.start([sumoBinary, "-c", "Sumo\osm.sumocfg", "--tripinfo-output",
                    "Sumo/tripinfo.xml", "--max-num-vehicles", "100", "--quit-on-end", "--start"])

        # Start the simulation loop
        run(i)
