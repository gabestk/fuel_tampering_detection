import traci
import random
import json
from datetime import datetime, timedelta

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
        json.dump(sorted_fuel_data, json_file, indent=4)


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
    if traci.vehicle.getRoadID(id) in stations and combustivel < 60:
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
    """TraCI control loop
    """
    try:
        # Definition of some useful variables

        # Vehicles data list
        # vehicle_data_list = []
        # Fuel data list
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
        refuel_fraud = 0
        refuel_no_fraud = 0

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
                        "fuel_remaining_no_fraud": tanque,
                        "fuel_remaining_fraud": tanque,
                        "route_num": 0,
                        "isGoingToRefuel": False,
                        "fueling": False,
                        "fuel_station": "",
                        "vehicle_factory_error": random.uniform(-2,2),
                        "refuel_count": 0
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
                vehicle_info[vehicle_id]["fuel_remaining_no_fraud"] -= fuel_consumption_liters
                vehicle_info[vehicle_id]["fuel_remaining_fraud"] -= fuel_consumption_liters
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
                #combus_liters_no_fraud = vehicle_info[vehicle_id]["fuel_remaining_no_fraud"]
                #combus_liters_fraud = vehicle_info[vehicle_id]["fuel_remaining_fraud"]
                #print(combus_percentage, expected_percentage, combus_liters_no_fraud, combus_liters_fraud)
                
                goingToFueling = vehicle_info[vehicle_id]["isGoingToRefuel"]
                
                # If the vehicle needs to refuel, sent to a fuel station
                if combus_percentage < 60 and goingToFueling is False:
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
                        vehicle_info[vehicle_id]["fueling"] = False
                        vehicle_factory_error = vehicle_info[vehicle_id]["vehicle_factory_error"] # Get the factory error from vehicle's fuel tank
                        vehicle_random_error = random.uniform(-3,3) # Get a random error to simulate the moment of fueling, such as expansion, fuel moving, etc.
                        random_refuel = random.uniform(5,15) # Get a random refuel amount
                        fraud = random.uniform(0,10) # Get a random fraud from this refuel
                        refuel_no_fraud = vehicle_factory_error + vehicle_random_error + random_refuel # Add amount of fuel without fraud
                        refuel_fraud = vehicle_factory_error + vehicle_random_error + random_refuel - fraud # Add amount of fuel with fraud
                        combus_percentage += (refuel_fraud/tanque)*100 # Get a real percentage of fuel tank
                        expected_percentage += (refuel_no_fraud/tanque)*100 # Get a expected percentage of fuel tank
                        vehicle_info[vehicle_id]["fuel_remaining_no_fraud"] += refuel_no_fraud
                        vehicle_info[vehicle_id]["fuel_remaining_fraud"] += refuel_fraud
                        vehicle_info[vehicle_id]["fuel_percentage"] = combus_percentage
                        vehicle_info[vehicle_id]["expected_fuel_percentage"] = expected_percentage
                        #combus_liters_no_fraud = vehicle_info[vehicle_id]["fuel_remaining_no_fraud"]
                        #combus_liters_fraud = vehicle_info[vehicle_id]["fuel_remaining_fraud"]
                        vehicle_info[vehicle_id]["refuel_count"] += 1
                        # Vehicle refuel information tuple
                        refuel_info = {
                            "vehicle_id": int(vehicle_id),
                            "station_id": vehicle_info[vehicle_id]["fuel_station"],
                            "fraud": fraud,
                            "vehicle_factory_error": vehicle_factory_error,
                            "vehicle_random_error": vehicle_random_error,
                            "refuel_amount": random_refuel,
                            "refuel_without_fraud": refuel_no_fraud,
                            "refuel_with_fraud": refuel_fraud,
                            "expected_fuel_percentage": expected_percentage,
                            "measured_fuel_percentage": combus_percentage
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
            if step > 35000:
                break
    
        for vehicle_id, info in vehicle_info.items():
            fuel_data_list.extend(info["refuel_info"])
            
        for vehicle_id, info in vehicle_info.items():
            print(f"Vehicle {vehicle_id} refueled {info['refuel_count']} times.")
            
        sorted_fuel_data = sorted(fuel_data_list, key=lambda x: (x["vehicle_id"]))
        
        # Function that writes a json file containing the data from all the vehicle
        #malkai(sorted_data)
        gabriel(sorted_fuel_data)
        

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
