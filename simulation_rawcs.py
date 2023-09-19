from Client.rawcs_client import FedCli
from Server.rawcs_server import Rawcs_sp

from Client.data_loader import get_dataset

import pickle
import flwr as fl
import os

try:
	os.remove('./results/history_simulation.pickle')
except FileNotFoundError:
	pass

n_clients = 24

STRATEGY='rawcs-sp'
NUM_CLIENTS=24
NUM_CLASSES=6
FIT_FRACTION=0.125
EVAL_FRACTION=1.0
MIN_FIT=2
MIN_EVAL=2
MIN_AVAIL=2
LEARNING_RATE=0.001
RESULTS_DIR='logs/MotionSense/'
SIM_ID='MotionSense'
TRANS_THRESH=0.2
DEVICES_PROFILES='Server/devices_profiles/profiles_sim_MotionSense_seed_1.json'
NETWORK_PROFILES='Server/devices_profiles/sim_1_num_clients_24_num_rounds_250.pkl'
SIM_IDX=1
DATASET_NAME='MotionSense'
DATASET_PATH='/home/gabrieltalasso/DEEV/Client/data/motion_sense'
NUM_ROUNDS=10
D_TEMP_SET_SIZE=48
EXPLORATION_FACTOR=0.9
STEP_WINDOW=20
PACER_STEP=2
PENALTY=2.0
CUT_OFF=0.95
BLACKLIST_NUM=1000
UTILITY_FACTOR=0.25
BATTERY_WEIGHT=0.33
CPU_COST_WEIGHT=0.33
LINK_PROB_WEIGHT=0.33
TARGET_ACCURACY=1.0
LINK_QUALITY_LOWER_LIM=0.2

input_shape = None
samples_per_client = []
for cid in range(NUM_CLIENTS):
    x_train, y_train, x_test, y_test = get_dataset(DATASET_NAME, DATASET_PATH, cid)
    input_shape = x_train.shape
    samples_per_client.append(len(x_train))

def funcao_cliente(cid):
	return FedCli(cid = int(cid))

history = fl.simulation.start_simulation(client_fn=funcao_cliente, 
								num_clients=n_clients, 
								strategy=Rawcs_sp(
                                                num_clients = NUM_CLIENTS,
                                                num_classes = NUM_CLASSES,
                                                fit_fraction = FIT_FRACTION,
                                                eval_fraction = EVAL_FRACTION,
                                                min_fit = MIN_FIT,
                                                min_eval = MIN_EVAL,
                                                min_avail = MIN_AVAIL,
                                                learning_rate = LEARNING_RATE,
                                                results_dir = RESULTS_DIR,
                                                sim_id = SIM_ID,
                                                transmission_threshold = TRANS_THRESH,
                                                devices_profile = DEVICES_PROFILES,
                                                network_profiles = NETWORK_PROFILES,
                                                sim_idx = SIM_IDX,
                                                #dataset_name = DATASET_NAME,
                                                #dataset_path = DATASET_PATH,
                                                #d_temp_set_size = D_TEMP_SET_SIZE,
                                                #exploration_factor = EXPLORATION_FACTOR,
                                                #step_window = STEP_WINDOW,
                                                #pacer_step = PACER_STEP,
                                                #penalty = PENALTY,
                                                #cut_off = CUT_OFF,
                                                #blacklist_num = BLACKLIST_NUM,
                                                #utility_factor = UTILITY_FACTOR,
                                                input_shape = input_shape,
                                                battery_weight = BATTERY_WEIGHT,
                                                cpu_cost_weight = CPU_COST_WEIGHT,
                                                link_prob_weight = LINK_PROB_WEIGHT,
                                                target_accuracy = TARGET_ACCURACY,
                                                link_quality_lower_lim = LINK_QUALITY_LOWER_LIM),
								config=fl.server.ServerConfig(NUM_ROUNDS))
