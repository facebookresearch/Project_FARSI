#Copyright (c) Facebook, Inc. and its affiliates.
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.

import sys
import os
sys.path.append(os.path.abspath('./../'))
import home_settings
from top.main_FARSI import run_FARSI_only_simulation
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

#  selecting the database based on the simulation method (power or performance)
if config.simulation_method == "power_knobs":
    from specs import database_input_powerKnobs as database_input
elif config.simulation_method == "performance":
    from specs import database_input
else:
    raise NameError("Simulation method unavailable")


# ------------------------------
# Functionality:
#     write the results into a file
# Variables:
#      sim_dp: design point simulation
#      result_dir: result directory
#      unique_number: a number to differentiate between designs
#      file_name: output file name
# ------------------------------
def write_results(sim_dp, case_study, result_dir, unique_number, file_name):
    result_dir_specific = os.path.join(result_dir, "result_summary")
    if not os.path.isdir(os.path.join(result_dir, "result_summary")):
        os.makedirs(result_dir_specific)

    output_file_minimal = os.path.join(result_dir_specific, file_name+ ".csv")

    # minimal output
    if os.path.exists(output_file_minimal):
        output_fh_minimal = open(output_file_minimal, "a")
    else:
        output_fh_minimal = open(output_file_minimal, "w")
        for metric in config.all_metrics:
            output_fh_minimal.write(metric + ",")
            if metric in sim_dp.database.db_input.get_budget_dict("glass").keys():
                output_fh_minimal.write(metric+"_budget" + ",")
        output_fh_minimal.write("sampling_mode,")
        output_fh_minimal.write("sampling_reduction" +",")
        for metric, accuracy_percentage in sim_dp.database.hw_sampling["accuracy_percentage"]["ip"].items():
            output_fh_minimal.write(metric+"_accuracy" + ",")  # for now only write the latency accuracy as the other
        for block_type, porting_effort in sim_dp.database.db_input.porting_effort.items():
            output_fh_minimal.write(block_type+"_effort" + ",")  # for now only write the latency accuracy as the other

        output_fh_minimal.write("output_design_status"+ ",")  # for now only write the latency accuracy as the other
        output_fh_minimal.write("case_study"+ ",")  # for now only write the latency accuracy as the other
        output_fh_minimal.write("unique_number" + ",")  # for now only write the latency accuracy as the other

    output_fh_minimal.write("\n")
    for metric in config.all_metrics:
        output_fh_minimal.write(str(sim_dp.dp_stats.get_system_complex_metric(metric)) + ",")
        if metric in sim_dp.database.db_input.get_budget_dict("glass").keys():
            output_fh_minimal.write(str(sim_dp.database.db_input.get_budget_dict("glass")[metric]) + ",")
    output_fh_minimal.write(sim_dp.database.hw_sampling["mode"] + ",")
    output_fh_minimal.write(sim_dp.database.hw_sampling["reduction"] + ",")
    for metric, accuracy_percentage in sim_dp.database.hw_sampling["accuracy_percentage"]["ip"].items():
        output_fh_minimal.write(str(accuracy_percentage) + ",")  # for now only write the latency accuracy as the other
    for block_type, porting_effort in sim_dp.database.db_input.porting_effort.items():
        output_fh_minimal.write(str(porting_effort)+ ",")  # for now only write the latency accuracy as the other

    if sim_dp.dp_stats.fits_budget(1):
        output_fh_minimal.write("budget_met"+ ",")  # for now only write the latency accuracy as the other
    else:
        output_fh_minimal.write("budget_not_met" + ",")  # for now only write the latency accuracy as the other
    output_fh_minimal.write(case_study + ",")  # for now only write the latency accuracy as the other
    output_fh_minimal.write(str(unique_number)+ ",")  # for now only write the latency accuracy as the other

    output_fh_minimal.close()


def copy_DSE_data(result_dir):
    #result_dir_specific = os.path.join(result_dir, "result_summary")
    os.system("cp " + config.latest_visualization+"/*" + " " + result_dir_specific)

if __name__ == "__main__":
    case_study = "simple_sim_run"
    current_process_id = 0
    total_process_cnt = 1
    #starting_exploration_mode = config.exploration_mode
    print('case study:' + case_study)

    # -------------------------------------------
    # set result folder
    # -------------------------------------------
    result_home_dir_default = os.path.join(os.getcwd(), "data_collection/data/" + case_study)
    result_home_dir = os.path.join(config.home_dir, "data_collection/data/" + case_study)
    date_time = datetime.now().strftime('%m-%d_%H-%M_%S')
    result_folder = os.path.join(result_home_dir,
                                 date_time)

    # -------------------------------------------
    # set parameters
    # -------------------------------------------
    experiment_repetition_cnt = 1
    reduction = "most_likely"
    db_population_misc_knobs = {"ip_freq_correction_ratio":1, "gpp_freq_correction_ratio":1}
    #workloads = {"audio_decoder", "edge_detection"}
    workloads = {"audio_decoder"}
    #workloads = {"edge_detection"}
    #workloads = {"hpvm_cava"}
    #workloads = {"SOC_example"}
    sw_hw_database_population = {"db_mode": "parse", "hw_graph_mode": "parse",
                                 "workloads": workloads, "misc_knobs":db_population_misc_knobs}

    # -------------------------------------------
    #  distribute the work
    # -------------------------------------------
    work_per_process = math.ceil(experiment_repetition_cnt / total_process_cnt)
    run_ctr = 0
    # -------------------------------------------
    # run the combination and collect the data
    # -------------------------------------------
    # -------------------------------------------
    # collect the exact hw sampling
    # -------------------------------------------
    accuracy_percentage = {}
    accuracy_percentage["sram"] = accuracy_percentage["dram"] = accuracy_percentage["ic"] = accuracy_percentage["gpp"] = accuracy_percentage[
        "ip"] = \
        {"latency": 1,
         "energy": 1,
         "area": 1,
         "one_over_area": 1}
    hw_sampling = {"mode": "exact", "population_size": 1, "reduction": reduction,
                   "accuracy_percentage": accuracy_percentage}
    db_input = database_input_class(sw_hw_database_population)
    print("hw_sampling:" + str(hw_sampling))
    print("budget set to:" + str(db_input.get_budget_dict("glass")))
    unique_suffix = str(total_process_cnt) + "_" + str(current_process_id) + "_" + str(run_ctr)
    dse_hndlr = run_FARSI_only_simulation(result_folder, unique_suffix, db_input, hw_sampling, sw_hw_database_population["hw_graph_mode"])
    run_ctr += 1

    result_dir_specific = os.path.join(result_folder, "result_summary")
    write_results(dse_hndlr.dse.so_far_best_sim_dp, case_study, result_dir_specific, unique_suffix,
                  config.FARSI_simple_run_prefix + "_" + str(current_process_id) + "_" + str(total_process_cnt))

    # write the results in the specific folder
    result_folder_modified = result_folder + "/runs/" + str(run_ctr) + "/"
    os.system("mkdir -p " + result_folder_modified)
    copy_DSE_data(result_folder_modified)
    write_results(dse_hndlr.dse.so_far_best_sim_dp, case_study, result_folder_modified, unique_suffix,
                  config.FARSI_simple_run_prefix + "_" + str(current_process_id) + "_" + str(total_process_cnt))

    copy_DSE_data(result_folder)

