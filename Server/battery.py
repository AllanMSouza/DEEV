import random

def get_energy_by_completion_time(comp, comm, avg_joules):
    # Understanding Operational 5G: A First Measurement Study on Its Coverage, Performance and Energy Consumption
    # 55% of 4W (avg 5g watts in the energy consumption graphs)
    # 1W = 1J/s
    # from: https://www.stouchlighting.com/blog/electricity-and-energy-terms-in-led-lighting-j-kw-kwh-lm/w
    AVG_5G_JOULES = 2.22
    comm_joules = AVG_5G_JOULES * comm

    comp_joules = avg_joules * comp

    return comp_joules + comm_joules
def update_utility(energy, utility):
    utility = utility - energy
    return utility
def init_battery_level(max_battery):
    perc = random.randint(30, 100)
    utility = battery = max_battery * (perc / 100)

    return battery, utility
def idle_power_deduction(battery, elapsed_time):
    # from: A Novel Non-invasive Method to Measure Power Consumption on Smartphones (39.27 mA = 0.15 W)
    # from: What can Android mobile app developers do about the energy consumption of machine learning (0.1 W)
    IDLE_CONSUMPTION = 0.1
    consumption = IDLE_CONSUMPTION * elapsed_time

    battery -= consumption

    return battery
def get_devices_battery_profiles(num_clients):
    # Common full mAh batteries
    # Values from commom smartphones specifications
    batteries_mah = random.choices([3500, 4000, 4500, 5000], k=num_clients)

    # Converting to joules
    # https://www.axconnectorlubricant.com/rce/battery-electronics-101.html#faq6
    # V = 3.85, commom value from specifications
    batteries_joules = [mAh * 3.85 * 3.6 for mAh in batteries_mah]

    # from: A Novel Non-invasive Method to Measure Power Consumption on Smartphones (39.27 mA = 0.15 W)
    # from: What can Android mobile app developers do about the energy consumption of machine learning (0.1 W)
    IDLE_CONSUMPTION = 0.1

    # Machine Learning at Facebook: Understanding Inference at the Edge
    # Energy Consumption of Batch and Online Data Stream Learning Models for Smartphone-based Human Activity Recognition
    min_battery = 2.0
    max_battery = 5.0
    battery_interval = max_battery - min_battery

    percentuals = [random.random() for _ in range(num_clients)]
    # 1W = 1J/s
    # from: https://www.stouchlighting.com/blog/electricity-and-energy-terms-in-led-lighting-j-kw-kwh-lm/w
    avg_joules = [IDLE_CONSUMPTION + min_battery + percentual * battery_interval for percentual in percentuals]


    return list(zip(batteries_joules, avg_joules))