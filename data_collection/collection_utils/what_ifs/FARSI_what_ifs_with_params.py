#Copyright (c) Facebook, Inc. and its affiliates.
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.

import sys
import os
sys.path.append(os.path.abspath('./../'))
import home_settings
from top.main_FARSI import run_FARSI
from settings import config
import os
import itertools
# main function
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
from visualization_utils.vis_hardware import *
import numpy as np
from specs.LW_cl import *
from specs.database_input import  *
import math
import matplotlib.colors as colors
#import pandas
import matplotlib.colors as mcolors
import pandas as pd
import argparse, sys
from FARSI_what_ifs import *

#  selecting the database based on the simulation method (power or performance)
if config.simulation_method == "power_knobs":
    from specs import database_input_powerKnobs as database_input
elif config.simulation_method == "performance":
    from specs import database_input
else:
    raise NameError("Simulation method unavailable")



def run_with_params(workloads, SA_depth, freq_range, base_budget_scaling, study_type, workload_folder):
    config.SA_depth = SA_depth
    # set the number of workers to be used (parallelism applied)
    current_process_id = 0
    total_process_cnt = 1
    system_workers = (current_process_id, total_process_cnt)

    # set the study type
    #study_type = "cost_PPA"


    workloads_first_letter  = '_'.join(sorted([el[0] for el in workloads]))
    budget_values = "lat_"+str(base_budget_scaling["latency"])+"__pow_"+str(base_budget_scaling["power"]) + "__area_"+str(base_budget_scaling["area"])

    # set result folder
    result_folder = os.path.join(workload_folder,
                                 date_time + "____"+ budget_values +"___workloads_"+workloads_first_letter)
    # set the IP spawning params
    ip_loop_unrolling = {"incr": 2, "max_spawn_ip": 17, "spawn_mode": "geometric"}
    #ip_freq_range = {"incr":3, "upper_bound":8}
    #mem_freq_range = {"incr":3, "upper_bound":6}
    #ic_freq_range = {"incr":4, "upper_bound":6}
    ip_freq_range = freq_range
    mem_freq_range = freq_range
    ic_freq_range = freq_range
    tech_node_SF = {"perf":1, "energy":{"non_gpp":.064, "gpp":1}, "area":{"non_mem":.0374 , "mem":.079, "gpp":1}}   # technology node scaling factor
    db_population_misc_knobs = {"ip_freq_correction_ratio": 1, "gpp_freq_correction_ratio": 1,
                                "ip_spawn": {"ip_loop_unrolling": ip_loop_unrolling, "ip_freq_range": ip_freq_range},
                                "mem_spawn": {"mem_freq_range":mem_freq_range},
                                "ic_spawn": {"ic_freq_range":ic_freq_range},
                                "tech_node_SF":tech_node_SF,
                                "base_budget_scaling":base_budget_scaling}

    # set software hardware database population
    # for SLAM
    #sw_hw_database_population = {"db_mode": "hardcoded", "hw_graph_mode": "generated_from_scratch",
    #                             "workloads": workloads, "misc_knobs": db_population_misc_knobs}
    # for paper workloads
    sw_hw_database_population = {"db_mode": "parse", "hw_graph_mode": "generated_from_scratch",
                                 "workloads": workloads, "misc_knobs": db_population_misc_knobs}
    # for check pointed
    #sw_hw_database_population = {"db_mode": "parse", "hw_graph_mode": "generated_from_check_point",
    #                             "workloads": workloads, "misc_knobs": db_population_misc_knobs}


    # depending on the study/substudy type, invoke the appropriate function
    if study_type == "simple_run":
        simple_run(result_folder, sw_hw_database_population, system_workers)
    elif study_type == "cost_PPA" and study_subtype == "run":
        input_error_output_cost_sensitivity_study(result_folder, sw_hw_database_population, system_workers, False, False)
    elif study_type == "input_error_output_cost_sensitivity" and study_subtype == "run":
        input_error_output_cost_sensitivity_study(result_folder, sw_hw_database_population, system_workers,  True, False)
    elif study_type == "input_error_input_cost_sensitivity" and study_subtype == "run":
        input_error_output_cost_sensitivity_study(result_folder, sw_hw_database_population, system_workers,True, True)
    elif study_type == "cost_PPA" and study_subtype == "plot_3d_distance":
        result_folder = "05-28_18-46_40"  # edge detection
        result_folder = "05-28_18-47_33" # hpvm cava
        result_folder = "05-28_18-47_03"
        result_folder = "05-31_16-24_49" # hpvm cava (2, tighter constraints)
        result_dir_addr= os.path.join(config.home_dir, 'data_collection/data/', study_type, result_folder,
                                                   "result_summary")
        full_file_addr = os.path.join(result_dir_addr,
                                      config.FARSI_cost_correlation_study_prefix + "_0_1.csv")
        plot_3d_dist(result_dir_addr, full_file_addr, workloads)

if __name__ == "__main__":

    study_type = "simple_run"
    #study_subtype = "plot_3d_distance"
    study_subtype = "run"
    assert study_type in ["cost_PPA", "simple_run", "input_error_output_cost_sensitivity", "input_error_input_cost_sensitivity"]
    assert study_subtype in ["run", "plot_3d_distance"]
    SA_depth = [10]
    freq_range = [1, 4, 6, 8]

    # fast run
    #workloads = [{"edge_detection"}]

    # each workload in isolation
    #workloads =[{"audio_decoder"}, {"edge_detection"}, {"hpvm_cava"}]

    # all workloads together
    workloads =[{"audio_decoder", "edge_detection", "hpvm_cava"}]

    # entire workload set
    #workloads = [{"hpvm_cava"}, {"audio_decoder"}, {"edge_detection"}, {"edge_detection", "audio_decoder"}, {"hpvm_cava", "audio_decoder"}, {"hpvm_cava", "edge_detection"} , {"audio_decoder", "edge_detection", "hpvm_cava"}]

    latency_scaling_range  = [.8, 1, 1.2]
    power_scaling_range  = [.8,1,1.2]
    area_scaling_range  = [.8,1,1.2]

    result_home_dir_default = os.path.join(os.getcwd(), "data_collection/data/" + study_type)
    result_folder = os.path.join(config.home_dir, "data_collection/data/" + study_type)
    date_time = datetime.now().strftime('%m-%d_%H-%M_%S')
    run_folder = os.path.join(result_folder, date_time)
    os.mkdir(run_folder)

    for w in workloads:
        workloads_first_letter = '_'.join(sorted([el[0] for el in w]))
        workload_folder = os.path.join(run_folder, workloads_first_letter)
        os.mkdir(workload_folder)
        for d in SA_depth:
            for latency_scaling,power_scaling, area_scaling in itertools.product(latency_scaling_range, power_scaling_range, area_scaling_range):
                base_budget_scaling = {"latency": latency_scaling, "power": power_scaling, "area": area_scaling}
                run_with_params(w, d, freq_range, base_budget_scaling, study_type, workload_folder)
