import flwr as fl
import tensorflow
import random
import time
import numpy as np
import tensorflow as tf
import os
import time
import sys

from Client.dataset_utils import ManageDatasets
from Client.model_definition import ModelCreation


import warnings
warnings.simplefilter("ignore")

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

class FedClient(fl.client.NumPyClient):

	def __init__(self, cid=0, n_clients=None, epochs=1, 
				 model_name            = 'None', 
				 client_selection      = False, 
				 solution_name         = 'None', 
				 aggregation_method    = 'None',
				 dataset               = '',
				 perc_of_clients       = 0,
				 decay                 = 0,
				 transmittion_threshold = 0.2):

		self.cid          = int(cid)
		self.n_clients    = n_clients
		self.model_name   = model_name
		self.local_epochs = epochs
		self.non_iid      = False

		self.model        = None
		self.x_train      = None
		self.x_test       = None
		self.y_train      = None
		self.y_test       = None

		#resources
		self.battery               = random.randint(10, 100)
		self.cpu_cost              = 0 
		self.transmittion_prob     = 1
		self.transmittion_threshold = transmittion_threshold

		#logs
		self.solution_name      = solution_name
		self.aggregation_method = aggregation_method
		self.dataset            = dataset

		self.client_selection = client_selection
		self.perc_of_clients  = perc_of_clients
		self.decay            = decay

		#params
		if self.aggregation_method == 'POC':
			self.solution_name = f"{solution_name}-{aggregation_method}-{self.perc_of_clients}"

		elif self.aggregation_method == 'DEEV': 
			self.solution_name = f"{solution_name}-{aggregation_method}-{self.decay}"

		else:
			self.solution_name = f"{solution_name}-{aggregation_method}"

		self.x_train, self.y_train, self.x_test, self.y_test = self.load_data(self.dataset, n_clients=self.n_clients)
		self.model                                           = self.create_model()

	def load_data(self, dataset_name, n_clients):
		return ManageDatasets(self.cid).select_dataset(dataset_name, n_clients, self.non_iid)

	def create_model(self):
		input_shape = self.x_train.shape

		if self.model_name == 'LR':
			return ModelCreation().create_LogisticRegression(input_shape, 6)

		elif self.model_name == 'DNN':
			return ModelCreation().create_DNN(input_shape, 8)

		elif self.model_name == 'CNN':
			return ModelCreation().create_CNN(input_shape, 6)
		

	def get_parameters(self, config):
		return self.model.get_weights()


	def fit(self, parameters, config):
		selected_clients   = []
		trained_parameters = []
		selected           = 0
		has_battery        = False
		total_time         = -1

		if config['selected_clients'] != '':
			selected_clients = [int (cid_selected) for cid_selected in config['selected_clients'].split(' ')]
		
		start_time = time.process_time()
		if self.cid in selected_clients or self.client_selection == False or int(config['round']) == 1:
			
			#check if client has some battery available for training
			if self.battery >= 0.05:
				self.model.set_weights(parameters)
				has_battery        = True
				selected           = 1
				history            = self.model.fit(self.x_train, self.y_train, verbose=1, epochs=self.local_epochs)
				trained_parameters = self.model.get_weights()
		
				total_time         = time.process_time() - start_time
				size_of_parameters = sum(map(sys.getsizeof, trained_parameters))
				avg_loss_train     = np.mean(history.history['loss'])
				avg_acc_train      = np.mean(history.history['accuracy'])

				filename = f"logs/{self.dataset}/{self.solution_name}/{self.model_name}/train_client.csv"
				os.makedirs(os.path.dirname(filename), exist_ok=True)

				self.battery  = self.battery - (total_time * 0.05)
				self.cpu_cost = total_time

			#fit_response = {'cid': self.cid, 'transmittion_prob' : self.transmittion_prob,'cpu_cost': total_time}

			#check transmission probability
			last_prob              = self.transmittion_prob
			self.transmittion_prob = random.uniform(0, 1)

			if last_prob >= self.transmittion_threshold and has_battery:
				with open(filename, 'a') as log_train_file:
					log_train_file.write(f"{config['round']}, {self.cid}, {selected}, {total_time}, {size_of_parameters}, {avg_loss_train}, {avg_acc_train}\n")
					
				return trained_parameters, len(self.x_train), {'cid': self.cid, 'transmittion_prob' : self.transmittion_prob, 'cpu_cost': total_time}

			#transmission or train failled
			else:
				filename = f"logs/{self.dataset}/{self.solution_name}/{self.model_name}/failures.csv"
				os.makedirs(os.path.dirname(filename), exist_ok=True)

				with open(filename, 'a') as log_failure_file:
					log_failure_file.write(f"{config['round']}, {self.cid}, {last_prob}, {self.battery}\n")

				return parameters, len(self.x_train), {'cid': self.cid, 'transmittion_prob' : self.transmittion_prob, 'cpu_cost': total_time}
		else:
			return parameters, len(self.x_train), {'cid': self.cid, 'transmittion_prob' : self.transmittion_prob, 'cpu_cost': total_time}				



	def evaluate(self, parameters, config):
		
		self.model.set_weights(parameters)
		loss, accuracy     = self.model.evaluate(self.x_test, self.y_test, verbose=0)
		size_of_parameters = sum(map(sys.getsizeof, parameters))

		filename = f"logs/{self.dataset}/{self.solution_name}/{self.model_name}/evaluate_client.csv"
		os.makedirs(os.path.dirname(filename), exist_ok=True)

		with open(filename, 'a') as log_evaluate_file:
			log_evaluate_file.write(f"{config['round']}, {self.cid}, {size_of_parameters}, {loss}, {accuracy}\n")

		evaluation_response = {
			"cid"               : self.cid,
			"accuracy"          : float(accuracy),
			"transmittion_prob" : self.transmittion_prob,
			"cpu_cost"          : self.cpu_cost,
			"battery"           : self.battery
		}

		return loss, len(self.x_test), evaluation_response


def main():
	
	client =  FedClient(
					cid                    = int(os.environ['CLIENT_ID']), 
					n_clients              = None, 
					model_name             = os.environ['MODEL'], 
					client_selection       = not os.environ['CLIENT_SELECTION'] == 'False', 
					epochs                 = int(os.environ['LOCAL_EPOCHS']), 
					solution_name          = os.environ['SOLUTION_NAME'],
					aggregation_method     = os.environ['ALGORITHM'],
					dataset                = os.environ['DATASET'],
					perc_of_clients        = float(os.environ['POC']),
					decay                  = float(os.environ['DECAY']),
					transmittion_threshold = float(os.environ['TRANSMISSION_THRESHOLD'])
					)

	fl.client.start_numpy_client(server_address=os.environ['SERVER_IP'], client=client)


if __name__ == '__main__':
	main()