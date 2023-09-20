from Client.client import FedClient
from Server.server import FedServer
import pickle
import flwr as fl
import os
import numpy as np

from Client.data_loader import get_dataset
from rawcs_utils import get_rawcs_params
try:
	os.remove('./results/history_simulation.pickle')
except FileNotFoundError:
	pass

n_clients = 24
n_rounds = 50

sol_name = 'teste2'
agg_method = 'Rawcs'
perc_clients = 0.1
dec = 0.5

def funcao_cliente(cid):
	return FedClient(cid = int(cid), n_clients=n_clients, epochs=1, 
				 model_name            = 'DNN', 
				 client_selection      = False, 
				 solution_name = sol_name,
				 aggregation_method    = agg_method,
				 dataset               = 'MotionSense',
				 perc_of_clients       = perc_clients,
				 decay                 = dec,
				 transmittion_threshold = 0.2)

rawcs_params = get_rawcs_params()

history = fl.simulation.start_simulation(client_fn=funcao_cliente, 
								num_clients=n_clients, 
								strategy=FedServer(aggregation_method=agg_method,
			   										fraction_fit = 1, 
													num_clients = n_clients, 
					                                decay=dec, perc_of_clients=perc_clients, 
													dataset='MotionSense', 
													solution_name = sol_name,
													model_name='DNN', 
													rawcs_params=rawcs_params),
								config=fl.server.ServerConfig(n_rounds))
