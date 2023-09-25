import numpy as np
import random
import threading
import numpy as np
import math
from abc import ABC, abstractmethod
from logging import INFO
from typing import Dict, List, Optional

from flwr.common.logger import log


from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion

import json
import os
import pickle
import random
from typing import Optional, List, Tuple, Union, Dict

import flwr as fl
import numpy as np
from flwr.common import Parameters, ndarrays_to_parameters, FitIns, FitRes, Scalar, parameters_to_ndarrays, EvaluateIns, \
    EvaluateRes
from flwr.server import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg

from Server.battery import get_energy_by_completion_time, idle_power_deduction
from Server.model_builder import create_DNN

from Client.model_definition import ModelCreation

def sample(
        clients,
        num_clients: int,
        criterion: Optional[Criterion] = None,
        selection = None,
        acc = None,
        decay_factor = None,
        server_round = None,
        POC_perc_of_clients = 0.5,
        rawcs_params = {}, 
        Rawcs_Manager = None
    ) -> List[ClientProxy]:
        
        # Sample clients which meet the criterion
        available_cids = list(clients)
        if criterion is not None:
            available_cids = [
                cid for cid in available_cids if criterion.select(clients[cid])
            ]

        if num_clients > len(available_cids):
            log(
                INFO,
                "Sampling failed: number of available clients"
                " (%s) is less than number of requested clients (%s).",
                len(available_cids),
                num_clients,
            )
            return []
        
        sampled_cids = available_cids.copy()
        
        if selection == 'DEEV' and server_round>1:
            selected_clients = []

            for idx_accuracy in range(len(acc)):
                if acc[idx_accuracy] < np.mean(np.array(acc)):
                    selected_clients.append(available_cids[idx_accuracy])
            
            sampled_cids = selected_clients.copy()

            if decay_factor > 0:
                the_chosen_ones  = len(selected_clients) * (1 - decay_factor)**int(server_round)
                selected_clients = selected_clients[ : math.ceil(the_chosen_ones)]
                sampled_cids = selected_clients.copy()


        if selection == 'POC' and server_round>1:
            selected_clients = []
            clients2select        = max(int(float(len(acc)) * float(POC_perc_of_clients)), 1)
            sorted_acc = [str(x) for _,x in sorted(zip(acc,available_cids))]
            for c in sorted_acc[:clients2select]:
                selected_clients.append(c)
                sampled_cids = selected_clients.copy()

        if selection == 'Rawcs':
            if server_round == 1:
                Rawcs_Manager = ManageRawcs(**rawcs_params)
            selected_cids = Rawcs_Manager.sample_fit()
            selected_cids = Rawcs_Manager.filter_clients_to_train_by_predicted_behavior(selected_cids, server_round)
            for j in range(len(selected_cids)):
                selected_cids[j] = str(selected_cids[j])
            sampled_cids = selected_cids.copy()

            return Rawcs_Manager, [clients[cid] for cid in sampled_cids]

        if selection == 'All':
            sampled_cids = random.sample(available_cids, num_clients)  


        return [clients[cid] for cid in sampled_cids]

class ManageRawcs():
    def __init__(self, iterable=(), **kwargs):
        self.__dict__.update(iterable, **kwargs)
    

    def sample_fit(self):
        network_profiles = None

        with open(self.network_profiles, 'rb') as file:
            network_profiles = pickle.load(file)

        clients_training_time = []

        self.clients_info = {}

        with open(self.devices_profile, 'r') as file:
            json_dict = json.load(file)
            for key in json_dict.keys():
                self.clients_info[int(key)] = json_dict[key]
                self.clients_info[int(key)]['perc_budget_10'] = False
                self.clients_info[int(key)]['perc_budget_20'] = False
                self.clients_info[int(key)]['perc_budget_30'] = False
                self.clients_info[int(key)]['perc_budget_40'] = False
                self.clients_info[int(key)]['perc_budget_50'] = False
                self.clients_info[int(key)]['perc_budget_60'] = False
                self.clients_info[int(key)]['perc_budget_70'] = False
                self.clients_info[int(key)]['perc_budget_80'] = False
                self.clients_info[int(key)]['perc_budget_90'] = False
                self.clients_info[int(key)]['perc_budget_100'] = False
                self.clients_info[int(key)]['initial_battery'] = self.clients_info[int(key)]['battery']
                if self.clients_info[int(key)]['total_train_latency'] > self.max_training_latency:
                    self.max_training_latency = self.clients_info[int(key)]['total_train_latency']
                clients_training_time.append(self.clients_info[int(key)]['total_train_latency'])
                self.clients_info[int(key)]['network_profile'] = network_profiles[int(key)]

        self.comp_latency_lim = np.percentile(clients_training_time, self.time_percentile)

        self.limit_relationship_max_latency = self.comp_latency_lim / self.max_training_latency
        selected_cids = []
        clients_with_resources = []


        for client in self.clients_info:
            if self.has_client_resources(client):
                clients_with_resources.append((client, self.clients_info[client]['accuracy']))

                client_cost = self.get_cost(self.battery_weight, self.cpu_cost_weight, self.link_prob_weight,
                                            self.clients_info[client]['battery'] /
                                            self.clients_info[client]['max_battery'],
                                            self.clients_info[client][
                                                'total_train_latency'] / self.max_training_latency,
                                            self.clients_info[client]['trans_prob'],
                                            self.target_accuracy,
                                            self.clients_info[client]['accuracy'])

                client_benefit = self.get_benefit()

                if random.random() <= (1 - client_cost / client_benefit):
                    selected_cids.append(client)

        if len(selected_cids) == 0 and len(clients_with_resources) != 0:
            clients_with_resources.sort(key=lambda client: client[1])
            selected_cids = [client[0] for client in clients_with_resources[:round(len(clients_with_resources)
                                                                                   * self.fit_fraction)]]
        if len(selected_cids) == 0:
            clients_with_battery = []

            for client in self.clients_info:
                if self.clients_info[client]['battery'] - \
                        self.clients_info[client]['delta_train_battery'] >= self.clients_info[client]['min_battery']:
                    clients_with_battery.append((client, self.clients_info[client]['accuracy']))

            clients_with_battery.sort(key=lambda client: client[1])

            selected_cids = [client[0] for client in
                             clients_with_battery[:round(len(clients_with_battery) * self.fit_fraction)]]

        return selected_cids

    def sample_eval(self, client_manager: ClientManager, num_clients):
        return client_manager.sample(num_clients)

    def filter_clients_to_train_by_predicted_behavior(self, selected_cids, server_round):
        # 1 - Atualiza tempo máximo de processamento
        # 2 - Atualiza bateria consumida pelos clientes em treinamento
        # 3 - Atualiza métricas: energia consumida, desperdício de energia, clientes dreanados, latência total
        # 4 - Identificar clientes que falharão a transmissão devido a alguma instabilidade
        # 5 - Atualizar métrica de consumo desperdiçado por falha do cliente
        # 6 - Atualiza lista de clientes que não completaram o treino por falta de bateria ou instabilidade da rede
        total_train_latency_round = 0.0
        total_energy_consumed = 0.0
        total_wasted_energy = 0.0
        round_depleted_battery_by_train = 0
        round_depleted_battery_total = 0
        round_transpassed_min_battery_level = 0
        max_latency = 0.0
        filtered_by_transmisssion = 0
        clients_to_not_train = []

        for cid in selected_cids:
            comp_latency = self.clients_info[cid]['train_latency']
            comm_latency = self.clients_info[cid]['comm_latency']
            avg_joules = self.clients_info[cid]["avg_joules"]

            client_latency = self.clients_info[cid]["total_train_latency"]
            client_consumed_energy = get_energy_by_completion_time(comp_latency, comm_latency, avg_joules)
            new_battery_value = self.clients_info[cid]['battery'] - client_consumed_energy

            filename = f"logs/clients_infos.csv"
            os.makedirs(os.path.dirname(filename), exist_ok=True)

            with open(filename, 'a') as batt_eval:
                batt_eval.write(f"{server_round}, {cid}, {new_battery_value}\n")

            if self.clients_info[cid]['battery'] >= self.clients_info[cid]['min_battery'] and new_battery_value < \
                    self.clients_info[cid]['min_battery']:
                round_transpassed_min_battery_level += 1

            if new_battery_value < 0:
                total_energy_consumed += self.clients_info[cid]['battery']
                total_wasted_energy += self.clients_info[cid]['battery']
                self.clients_info[cid]['battery'] = 0
                round_depleted_battery_by_train += 1
                round_depleted_battery_total += 1
                clients_to_not_train.append(cid)
            else:
                total_energy_consumed += client_consumed_energy
                self.clients_info[cid]['battery'] = new_battery_value

                if self.clients_info[cid]['network_profile'][server_round-1] < self.transmission_threshold:
                    clients_to_not_train.append(cid)
                    total_wasted_energy += client_consumed_energy
                    filtered_by_transmisssion += 1
                else:
                    total_train_latency_round += client_latency

            if client_latency > max_latency and cid not in clients_to_not_train:
                max_latency = client_latency
        # 7 - Remove de clientes selecionados os que foram drenados pelo treinamento
        filtered_selected_cids = list(set(selected_cids).difference(clients_to_not_train))
        # 8 - Calcular consumo em estado de espera
        # 9 - Atualizar bateria de cada cliente
        # 10 - Atualizar clientes que foram drenados sem que seja pelo treino
        for cid in self.clients_info:
            old_battery_level = self.clients_info[cid]['battery']

            if old_battery_level > 0:
                if cid not in filtered_selected_cids:
                    new_battery_level = idle_power_deduction(old_battery_level, max_latency)
                else:
                    idle_time = max_latency - (self.clients_info[cid]['total_train_latency'])
                    new_battery_level = idle_power_deduction(old_battery_level, idle_time)

                if self.clients_info[cid]['battery'] >= self.clients_info[cid]['min_battery'] and new_battery_level < \
                        self.clients_info[cid]['min_battery']:
                    round_transpassed_min_battery_level += 1

                if new_battery_level <= 0:
                    self.clients_info[cid]['battery'] = 0
                    round_depleted_battery_total += 1
                else:
                    self.clients_info[cid]['battery'] = new_battery_level

        perc_budget_10, perc_budget_100, perc_budget_20, perc_budget_30, perc_budget_40, perc_budget_50, \
            perc_budget_60, perc_budget_70, perc_budget_80, perc_budget_90 = self.transpassed_budget()

        #filename = self.results_dir + self.__repr__() + "_" + self.sim_id + "_" + str(
        #    self.sim_idx) + f"_system_metrics_frac_{self.fit_fraction}.csv"

        # os.makedirs(os.path.dirname(filename), exist_ok=True)

        # with open(filename, 'a') as log:
        #     log.write(f"{server_round},{total_train_latency_round},{total_energy_consumed},{total_wasted_energy},"
        #               f"{len(selected_cids)},{round_depleted_battery_by_train},{round_depleted_battery_total},"
        #               f"{filtered_by_transmisssion},{len(filtered_selected_cids)},"
        #               f"{round_transpassed_min_battery_level},{perc_budget_10},{perc_budget_20},{perc_budget_30},"
        #               f"{perc_budget_40},{perc_budget_50},{perc_budget_60},{perc_budget_70},{perc_budget_80},"
        #               f"{perc_budget_90},{perc_budget_100}\n"
        #               )
        return filtered_selected_cids

    def update_sample(self, client_manager, selected_cids):
        selected_clients = []

        for cid in selected_cids:
            selected_clients.append(client_manager.clients[str(cid)])

        return selected_clients

    def has_client_resources(self, client_id: int):
        if (self.clients_info[client_id]['battery'] - self.clients_info[client_id]['delta_train_battery']) \
                >= self.clients_info[client_id]['min_battery'] and self.clients_info[client_id]['trans_prob'] \
                >= self.link_quality_lower_lim:
            return True

        return False

    def get_cost(self, battery_w: float, cpu_w: float, link_w: float, battery: float, cpu_relation: float,
                 link_qlt: float, target_accuracy: float, client_accuracy: float):

        if cpu_relation > self.limit_relationship_max_latency:
            penalty_factor = np.e ** (abs(cpu_relation - self.limit_relationship_max_latency)) - 1
        else:
            penalty_factor = 1

        return (((battery_w) * (1 - battery)) + ((cpu_w) * (cpu_relation) * penalty_factor) + (
                    (link_w) * (1 - link_qlt))) ** (target_accuracy - client_accuracy)

    def get_benefit(self):
        return self.target_accuracy

    def transpassed_budget(self):
        perc_budget_10 = 0
        perc_budget_20 = 0
        perc_budget_30 = 0
        perc_budget_40 = 0
        perc_budget_50 = 0
        perc_budget_60 = 0
        perc_budget_70 = 0
        perc_budget_80 = 0
        perc_budget_90 = 0
        perc_budget_100 = 0
        for cid in self.clients_info:
            depletion = 1 - self.clients_info[cid]['battery'] / self.clients_info[cid]['initial_battery']
            if not self.clients_info[cid]['perc_budget_10'] and depletion > 0.1:
                self.clients_info[cid]['perc_budget_10'] = True
            if not self.clients_info[cid]['perc_budget_20'] and depletion > 0.2:
                self.clients_info[cid]['perc_budget_20'] = True
            if not self.clients_info[cid]['perc_budget_30'] and depletion > 0.3:
                self.clients_info[cid]['perc_budget_30'] = True
            if not self.clients_info[cid]['perc_budget_40'] and depletion > 0.4:
                self.clients_info[cid]['perc_budget_40'] = True
            if not self.clients_info[cid]['perc_budget_50'] and depletion > 0.5:
                self.clients_info[cid]['perc_budget_50'] = True
            if not self.clients_info[cid]['perc_budget_60'] and depletion > 0.6:
                self.clients_info[cid]['perc_budget_60'] = True
            if not self.clients_info[cid]['perc_budget_70'] and depletion > 0.7:
                self.clients_info[cid]['perc_budget_70'] = True
            if not self.clients_info[cid]['perc_budget_80'] and depletion > 0.8:
                self.clients_info[cid]['perc_budget_80'] = True
            if not self.clients_info[cid]['perc_budget_90'] and depletion > 0.9:
                self.clients_info[cid]['perc_budget_90'] = True
            if not self.clients_info[cid]['perc_budget_100'] and depletion == 1.0:
                self.clients_info[cid]['perc_budget_100'] = True

            if self.clients_info[cid]['perc_budget_10']:
                perc_budget_10 += 1
            if self.clients_info[cid]['perc_budget_20']:
                perc_budget_20 += 1
            if self.clients_info[cid]['perc_budget_30']:
                perc_budget_30 += 1
            if self.clients_info[cid]['perc_budget_40']:
                perc_budget_40 += 1
            if self.clients_info[cid]['perc_budget_50']:
                perc_budget_50 += 1
            if self.clients_info[cid]['perc_budget_60']:
                perc_budget_60 += 1
            if self.clients_info[cid]['perc_budget_70']:
                perc_budget_70 += 1
            if self.clients_info[cid]['perc_budget_80']:
                perc_budget_80 += 1
            if self.clients_info[cid]['perc_budget_90']:
                perc_budget_90 += 1
            if self.clients_info[cid]['perc_budget_100']:
                perc_budget_100 += 1
        return perc_budget_10, perc_budget_100, perc_budget_20, perc_budget_30, perc_budget_40, perc_budget_50, \
            perc_budget_60, perc_budget_70, perc_budget_80, perc_budget_90