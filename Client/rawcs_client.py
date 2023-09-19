import os
import random
import time

import flwr as fl

from Client.data_loader import get_dataset
from Client.model_builder import create_DNN


class FedCli(fl.client.NumPyClient):
    def __init__(self, cid, net, x_train, y_train, x_test, y_test, epochs):
        self.cid = cid
        self.net = net
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.epochs = epochs

    def fit(self, parameters, config):
        selected_clients = []

        if config['selected_clients'] != '':
            selected_clients = [int(cid_selected) for cid_selected in config['selected_clients'].split(' ')]

        if self.cid in selected_clients:
            self.net.set_weights(parameters)
            oort_statistical_utility = 0

            if config['strategy'] == "power_of_choice" and int(config['round']) % 2 != 0:
                loss, accuracy = self.net.evaluate(self.x_train, self.y_train, verbose=0)
            else:
                history = self.net.fit(self.x_train, self.y_train, verbose=0, epochs=self.epochs)
                loss = sum(history.history['loss']) / len(history.history['loss'])
                accuracy = sum(history.history['accuracy']) / len(history.history['accuracy'])

                if config['strategy'] == "eafl":
                    oort_statistical_utility = ((sum([loss ** 2 for loss in history.history['loss']]) / len(
                        history.history['loss'])) ** 0.5) / len(history.history['loss'])


            trained_parameters = self.net.get_weights()

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
        self.net.set_weights(parameters)
        loss, accuracy = self.net.evaluate(self.x_test, self.y_test, verbose=0)

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
    net = create_DNN(x_train.shape, num_classes)
    client = FedCli(cid, net, x_train, y_train, x_test, y_test, epochs)

    time.sleep(random.uniform(time2start_min, time2start_max))

    fl.client.start_numpy_client(server_address=server_addr, client=client)


if __name__ == '__main__':
    main()