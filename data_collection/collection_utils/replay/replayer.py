#Copyright (c) Facebook, Inc. and its affiliates.
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.

import pickle
from DSE_utils import hill_climbing
from specs.data_base import *
from design_utils.design import *
from visualization_utils import vis_sim
from visualization_utils import vis_hardware, vis_stats,plot
import importlib

if config.simulation_method == "power_knobs":
    from specs import database_input_powerKnobs as database_input
elif config.simulation_method == "performance":
    from specs import database_input
else:
    raise NameError("Simulation method unavailable")

# This class allows for replaying the designs already generated by FARSI.
# This helps for debugging to have a closer look at the simulation step by step, for a design
# with a large error.
class Replayer:
    def __init__(self):

        # TODO: needs to be update to work
        print("needs to be update to work")
        exit(0)

        self.database = DataBase(database_input.tasksL, database_input.blocksL,
                         database_input.pe_mapsL, database_input.pe_schedulesL, database_input.SOCsL)   # hw/sw database
        self.dse = hill_climbing.HillClimbing(self.database)  # design space exploration to use
        self.home_dir = config.home_dir   # home directory of repo
        self.data_collection_top_folder = self.home_dir + "/data_collection/data_already_collected"  # directory with designs in it.
        self.replay_top_folder = self.data_collection_top_folder + "/replay/"   # directory that replayer dumps its results into.
        self.pickled_file_name = "ex_dp_pickled.txt"   # generated designs are stored in pickle format, that are then read by replayer
        self.latest_ex_dp = None  # design to look at (replay)
        self.latest_sim_dp = None   # latest simulated design
        self.latest_stat_result = None   # latest stats associated with the design
        self.latest_des_folder_addr = None  # latest design folder
        self.name_ctr = 0   # name ctr (id for designs)

    # ------------------------------
    # Functionality:
    #       loading a design.
    # Variables:
    #       des_folder_name: folder where the design resides in
    # ------------------------------
    def load_design(self, des_folder_name):
        # des_folder_name is with respect to data_collection
        self.latest_des_folder_addr = self.data_collection_top_folder + "/" + des_folder_name
        self.latest_top_replay_folder_addr = self.replay_top_folder + "/" + "/".join(des_folder_name.split("/")[:-1])
        pickled_file_addr = self.latest_des_folder_addr + "/" + "ex_dp_pickled.txt"
        return pickle.load(open(pickled_file_addr, "rb"))

    # ------------------------------
    # Functionality:
    #       replay the design.
    # Variables:
    #       des_folder_name: folder where the design resides in
    # ------------------------------
    def replay(self, des_folder_name):
        self.latest_ex_dp = self.load_design(des_folder_name)
        self.latest_ex_dp = copy.deepcopy(self.latest_ex_dp)  # need to do this to clear some memories
        self.latest_sim_dp = self.dse.eval_design(self.latest_ex_dp, self.database)
        self.latest_stat_result = self.latest_sim_dp.dp_stats

    # ------------------------------
    # Functionality:
    #       generate PA digestible files for the replayed design.
    # ------------------------------
    def gen_pa(self):
        import_ver = importlib.import_module("data_collection.FB_private.verification_utils.PA_generation.PA_generators")
        # get PA genera
        pa_ver_obj = import_ver.PAVerGen()
        # make all the combinations
        knobs_list, knob_order = pa_ver_obj.gen_all_PA_knob_combos(import_ver.PA_knobs_to_explore)
        os.makedirs(self.latest_top_replay_folder_addr, exist_ok=True)

        # go through all the combinations and generate the corresponding the PA design.
        for knobs in knobs_list:
            self.latest_ex_dp.reset_PA_knobs()
            self.latest_ex_dp.update_PA_knobs(knobs, knob_order)
            PA_result_folder = os.path.join(self.latest_top_replay_folder_addr, str(self.latest_ex_dp.id))
            os.makedirs(PA_result_folder, exist_ok =True)
            # visualize the hardware
            vis_hardware.vis_hardware(self.latest_ex_dp, "block_extra", PA_result_folder, "system_image_block_extra.pdf")
            vis_hardware.vis_hardware(self.latest_ex_dp, "block_task", PA_result_folder, "system_image_block_task.pdf")
            self.latest_stat_result.dump_stats(PA_result_folder)
            if config.VIS_SIM_PER_GEN: vis_sim.plot_sim_data(self.latest_stat_result, self.latest_ex_dp, PA_result_folder)
            # generate PA
            pa_obj = import_ver.PAGen(database_input.proj_name, self.latest_ex_dp, PA_result_folder, config.sw_model)
            pa_obj.gen_all()
            self.latest_ex_dp.dump_props(PA_result_folder)

            # pickle the result
            ex_dp_pickled_file = open(os.path.join(PA_result_folder, "ex_dp_pickled.txt"), "wb")
            pickle.dump(self.latest_ex_dp, ex_dp_pickled_file)
            ex_dp_pickled_file.close()

            sim_dp_pickled_file = open(os.path.join(PA_result_folder, "sim_dp_pickled.txt"), "wb")
            pickle.dump(self.latest_sim_dp, sim_dp_pickled_file)
            sim_dp_pickled_file.close()
            self.name_ctr += 1


