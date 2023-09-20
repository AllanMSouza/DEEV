import os
import random
import time

import flwr as fl

from Client.data_loader import get_dataset
from Client.model_builder import create_DNN
from Client.dataset_utils import ManageDatasets
from Client. model_definition import ModelCreation


class FedCli(fl.client.NumPyClient):
    def __init__(self, cid, n_clients, model_name, dataset_name, epochs):
        self.cid = cid
        self.epochs = epochs

        self.dataset = dataset_name
        self.n_clients = n_clients

        self.non_iid = True

        self.model_name = model_name



        self.x_train, self.y_train, self.x_test, self.y_test = self.load_data(self.dataset, n_clients=self.n_clients)
        self.model = self.create_model()

    def load_data(self, dataset_name, n_clients):
        return ManageDatasets(self.cid).select_dataset(dataset_name, n_clients, self.non_iid)

    def create_model(self):
        input_shape = self.x_train.shape

        if self.model_name == 'LR':
            return ModelCreation().create_LogisticRegression(input_shape, 6)

        elif self.model_name == 'DNN':
            return ModelCreation().create_DNN(input_shape, 6)

        elif self.model_name == 'CNN':
            return ModelCreation().create_CNN(input_shape, 6)

    def fit(self, parameters, config):
        selected_clients = []

        if config['selected_clients'] != '':
            selected_clients = [int(cid_selected) for cid_selected in config['selected_clients'].split(' ')]

        if self.cid in selected_clients:
            self.model.set_weights(parameters)
            oort_statistical_utility = 0

            if config['strategy'] == "power_of_choice" and int(config['round']) % 2 != 0:
                loss, accuracy = self.model.evaluate(self.x_train, self.y_train, verbose=0)
            else:
                history = self.model.fit(self.x_train, self.y_train, verbose=0, epochs=self.epochs)
                loss = sum(history.history['loss']) / len(history.history['loss'])
                accuracy = sum(history.history['accuracy']) / len(history.history['accuracy'])

                if config['strategy'] == "eafl":
                    oort_statistical_utility = ((sum([loss ** 2 for loss in history.history['loss']]) / len(
                        history.history['loss'])) ** 0.5) / len(history.history['loss'])
                    
            filename = f"logs/{self.dataset}/{'RAWCS'}/{self.model_name}/train_client.csv"
            os.makedirs(os.path.dirname(filename), exist_ok=True)

            with open(filename, 'a') as log_train_file:
                log_train_file.write(f"{config['round']}, {self.cid}, {1}, {'-'}, {'-'}, {loss}, {accuracy}\n")


            trained_parameters = self.model.get_weights()

            return trained_parameters, len(self.x_train), {"loss": float(loss),
                                                           "accuracy": float(accuracy),
                                                           "cid": int(self.cid),
                                                           "oort_stat_util": float(oort_statistical_utility)
                                                           }
        else:
            return parameters, 0, {"loss": float(0.0),
                                   "accuracy": float(0.0),
                                   "cid": int(self.cid),
                                   "oort_stat_util": float(0.0)
                                   }

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)

        filename = f"logs/{self.dataset}/{'RAWCS'}/{self.model_name}/evaluate_client.csv"
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, 'a') as log_test_file:
            log_test_file.write(f"{config['round']}, {self.cid}, {'-'},  {loss}, {accuracy}\n")

        return loss, len(self.x_test), {"loss": loss, "accuracy": accuracy, "cid": int(self.cid)}


def main():
    cid = int(os.environ['CLIENT_ID'])
    num_classes = int(os.environ['NUM_CLASSES'])
    epochs = int(os.environ['EPOCHS'])
    dataset_name = os.environ['DATASET_NAME']
    dataset_path = os.environ['DATASET_PATH']
    time2start_min = int(os.environ['TIME2STARTMIN'])
    time2start_max = int(os.environ['TIME2STARTMAX'])
    server_addr = os.environ['SERVER_ADDR']

    x_train, y_train, x_test, y_test = get_dataset(dataset_name, dataset_path, cid)
    model = ModelCreation().create_DNN(x_train.shape, num_classes)
    client = FedCli(cid, model, x_train, y_train, x_test, y_test, epochs)

    time.sleep(random.uniform(time2start_min, time2start_max))

    fl.client.start_numpy_client(server_address=server_addr, client=client)


if __name__ == '__main__':
    main()