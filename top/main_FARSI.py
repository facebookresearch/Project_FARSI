#Copyright (c) Facebook, Inc. and its affiliates.
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.

from SIM_utils.SIM import *
from DSE_utils.design_space_exploration_handler import  *
from specs.database_input import *


# Run an instance of FARSI, the exploration framework
# Variables:
#   unique_number: a unique number used for check_point
#   db_input: input database to use for design generation
#   hw_sampling: hardware sampling mode. This specifies: 1.what percentage of
#   error does the data base has. What should be the population size for each design
#   and the statistical reduction mode (avg, max, min)
#   starting_exploration_mode: whether to start from scratch or from an existing check pointed design
def run_FARSI(result_folder, unique_number, db_input, hw_sampling, starting_exploration_mode ="generated_from_scratch"):
    if config.use_cacti:
        print("*****************************")
        print("***** YOU ASKED TO USE CACTI FOR POWER/AREA MODELING OF MEMORY SUBSYSTEM. MAKE SURE YOU HAVE CACTI INSTALLED ****")
        print("*****************************")

    with warnings.catch_warnings():
        warnings.simplefilter(config.warning_mode)
        assert starting_exploration_mode in ["generated_from_scratch", "generated_from_check_point", "parse"]

        config.dse_type = "hill_climbing"
        if not (config.dse_type == "hill_climbing"):
            print("this main is only suitable the hill_climbing search")
            exit(0)

        # init relevant configs and objects
        num_SOCs = 1  # how many SOCs to spread the design across
        so_far_best_ex_dp = None
        boost_SOC = False  # specify whether to stick with the old SOC type or boost it

        # set up the design handler and the design explorer
        dse_handler = DSEHandler(result_folder)
        # First copies the DataBase information (SoC, Blocks(modules), tasks, mappings, scheduling)
        # then chooses among DSE algorithms (hill climbing) and initializes it
        dse_handler.setup_an_explorer(db_input, hw_sampling)

        # iterate until you find a design meeting the constraints or terminate if none found
        while True:
            # Use the check pointed design, parsed design or generate an simple design point
            dse_handler.prepare_for_exploration(so_far_best_ex_dp, boost_SOC, starting_exploration_mode)
            # does the simulation for design points (performance, energy, and area core calculations)
            dse_handler.explore()
            dse_handler.check_point_best_design(unique_number)  # check point
            return dse_handler


# Run FARSI only to simulate one design (parsed, generated or from check point)
# Variables:
#   unique_number: a unique number used for check_point
#   db_input: input database to use for design generation
#   hw_sampling: hardware sampling mode. This specifies: 1.what percentage of
#   error does the data base has. What should be the population size for each design
#   and the statistical reduction mode (avg, max, min)
#   starting_exploration_mode: whether to start from scratch or from an existing check pointed design
def run_FARSI_only_simulation(result_folder, unique_number, db_input, hw_sampling, starting_exploration_mode ="from_scratch"):
    if config.use_cacti:
        print("*****************************")
        print("***** YOU ASKED TO USE CACTI FOR POWER/AREA MODELING OF MEMORY SUBSYSTEM. MAKE SURE YOU HAVE CACTI INSTALLED ****")
        print("*****************************")

    with warnings.catch_warnings():
        warnings.simplefilter(config.warning_mode)

        config.dse_type = "hill_climbing"
        if not (config.dse_type == "hill_climbing"):
            print("this main is only suitable the hill_climbing search")
            exit(0)

        # init relevant configs and objects
        num_SOCs = 1  # how many SOCs to spread the design across
        so_far_best_ex_dp = None
        boost_SOC = False  # specify whether to stick with the old SOC type or boost it

        # set up the design handler and the design explorer
        dse_handler = DSEHandler(result_folder)
        # First copies the DataBase information (SoC, Blocks(modules), tasks, mappings, scheduling)
        # then chooses among DSE algorithms (hill climbing) and initializes it
        dse_handler.setup_an_explorer(db_input, hw_sampling)

        # Use the check pointed design, parsed design or generate an simple design point
        dse_handler.prepare_for_exploration(so_far_best_ex_dp, boost_SOC, starting_exploration_mode)
        # does the simulation for design points (performance, energy, and area core calculations)
        dse_handler.explore_one_design()
        dse_handler.check_point_best_design(unique_number)  # check point
        return dse_handler


# main function. If this file is run in isolation,
# the the main function is called (which itself calls runFARSI mentioned above)
if __name__ == "__main__":

    run_ctr = 0
    case_study = "simple_run"
    result_home_dir = os.path.join(os.getcwd(), "data_collection/data/" + case_study)
    home_dir = os.getcwd()
    date_time = datetime.now().strftime('%m-%d_%H-%M_%S')
    result_folder = os.path.join(result_home_dir,
                                 date_time)

    current_process_id = 0
    total_process_cnt = 0
    starting_exploration_mode = config.exploration_mode
    print('case study:' + case_study)

    # -------------------------------------------
    # set parameters
    # -------------------------------------------
    experiment_repetition_cnt = 1
    reduction = "most_likely"
    workloads = {"SLAM"}
    sw_hw_database_population = {"db_mode": "hardcoded", "hw_graph_mode": "generated_from_scratch",
                                 "workloads": workloads}

    accuracy_percentage = {}
    accuracy_percentage["mem"] = accuracy_percentage["ic"] = accuracy_percentage["gpp"] = accuracy_percentage["ip"] = \
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
    dse_hndlr = run_FARSI(result_folder, unique_suffix, db_input, hw_sampling, sw_hw_database_population["hw_graph_mode"])


    exploration_start_time = time.time()
    db_input = database_input_class(config.budgets_dict, config.other_values_dict)
    hw_sampling = {"mode": "exact", "population_size": 1}
    dse_handler = run_FARSI(db_input, hw_sampling)
    if config.REPORT: dse_handler.dse.report(exploration_start_time); dse_handler.dse.plot_data()
