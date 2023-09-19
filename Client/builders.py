from eafl import Eafl
from fedavg import FedAvg
from power_of_choice import Poc
from rawcs_md import Rawcs_md
from rawcs_mp import Rawcs_mp
from rawcs_sd import Rawcs_sd
from rawcs_sp import Rawcs_sp


def get_strategy(strategy: str, num_clients: int, num_classes: int, fit_fraction: float, eval_fraction: float,
                 min_fit: int, min_eval: int, min_avail: int, learning_rate: float, results_dir: str, sim_id: str,
                 transmission_threshold: float, devices_profile: str, network_profiles: str, sim_idx: int, input_shape,
                 samples_per_client: list, d_temp_set_size: int, exploration_factor: float, step_window: int,
                 pacer_step: int, penalty: float, cut_off: float, blacklist_num: int, utility_factor: float,
                 battery_weight: float, cpu_cost_weight: float, link_prob_weight: float, target_accuracy: float,
                 link_quality_lower_lim: float):
    to_return = None

    if strategy == "fedavg":
        to_return = FedAvg(num_clients, num_classes, fit_fraction, eval_fraction, min_fit, min_eval, min_avail,
                           learning_rate, results_dir, sim_id, transmission_threshold, devices_profile, network_profiles,
                           sim_idx, input_shape)
    elif strategy == "poc":
        to_return = Poc(num_clients, num_classes, fit_fraction, eval_fraction, min_fit, min_eval, min_avail,
                        learning_rate, results_dir, sim_id, transmission_threshold, devices_profile, network_profiles,
                        sim_idx, input_shape, samples_per_client, d_temp_set_size)
    elif strategy == 'eafl':
        to_return = Eafl(num_clients, num_classes, fit_fraction, eval_fraction, min_fit, min_eval, min_avail,
                         learning_rate, results_dir, sim_id, transmission_threshold, devices_profile, network_profiles,
                         sim_idx, input_shape, exploration_factor, step_window, pacer_step, penalty, cut_off,
                         blacklist_num, utility_factor)
    elif strategy == "rawcs-sp":
        to_return = Rawcs_sp(num_clients, num_classes, fit_fraction, eval_fraction, min_fit, min_eval, min_avail,
                             learning_rate, results_dir, sim_id, transmission_threshold, devices_profile,
                             network_profiles, sim_idx, input_shape, battery_weight, cpu_cost_weight, link_prob_weight,
                             target_accuracy, link_quality_lower_lim)
    elif strategy == "rawcs-mp":
        to_return = Rawcs_mp(num_clients, num_classes, fit_fraction, eval_fraction, min_fit, min_eval, min_avail,
                             learning_rate, results_dir, sim_id, transmission_threshold, devices_profile,
                             network_profiles, sim_idx, input_shape, battery_weight, cpu_cost_weight, link_prob_weight,
                             target_accuracy, link_quality_lower_lim)
    elif strategy == "rawcs-sd":
        to_return = Rawcs_sd(num_clients, num_classes, fit_fraction, eval_fraction, min_fit, min_eval, min_avail,
                             learning_rate, results_dir, sim_id, transmission_threshold, devices_profile,
                             network_profiles, sim_idx, input_shape, battery_weight, cpu_cost_weight, link_prob_weight,
                             target_accuracy, link_quality_lower_lim)
    elif strategy == "rawcs-md":
        to_return = Rawcs_md(num_clients, num_classes, fit_fraction, eval_fraction, min_fit, min_eval, min_avail,
                             learning_rate, results_dir, sim_id, transmission_threshold, devices_profile,
                             network_profiles, sim_idx, input_shape, battery_weight, cpu_cost_weight, link_prob_weight,
                             target_accuracy, link_quality_lower_lim)

    return to_return
