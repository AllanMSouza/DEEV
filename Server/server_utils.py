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

def sample(
        clients,
        num_clients: int,
        min_num_clients: Optional[int] = None,
        criterion: Optional[Criterion] = None,
        CL = True,
        selection = None,
        acc = None,
        decay_factor = None,
        server_round = None,
        idx = None,
        cluster_round = 0,
        POC_perc_of_clients = 0.5
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

        if selection == 'All':
            sampled_cids = random.sample(available_cids, num_clients)  


        return [clients[cid] for cid in sampled_cids]