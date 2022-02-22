#Copyright (c) Facebook, Inc. and its affiliates.
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.
from zipfile import ZipFile
from os.path import basename
from design_utils.design import  *
from DSE_utils import hill_climbing
from specs.data_base import *
from visualization_utils import vis_hardware, vis_stats,plot
import csv
import dill
import pickle
import matplotlib.pyplot as plt
if config.simulation_method == "power_knobs":
    from specs import database_input_powerKnobs as database_input
elif config.simulation_method == "performance":
    from specs import database_input
else:
    raise NameError("Simulation method unavailable")
import random
import pickle

# class used for deign handling.
# This class uses an exploration algorithm (such as hill climbing to explore the design space)
# specify the exploration algorithm in the config file.
class DSEHandler:
    def __init__(self, result_dir=os.getcwd()):
        self.check_pointed_best_sim_dps = []  # list of check pointed simulated designs
        self.check_pointed_best_ex_dps = []   # list of check pointed example designs
        self.dse = None                       # design space exploration algorithm
        self.database = None                  # data base (contains hw and sw database for mapping/allocation of hw/sw)
        self.IP_library = []
        self.result_dir = result_dir
        self.check_point_folder_name = "check_points"
        self.check_point_ctr = 0
        return None

    # ---------------
    # Functionality:
    #       set up an exploration dse.
    #       specify the explorer type in the config file
    # ---------------
    def explore_exhaustively(self, db_input, hw_sampling, system_workers):
        mapping_process_id = system_workers[1]
        FARSI_gen_process_id = system_workers[3]

        if config.dse_type == "exhaustive":
            self.database = DataBase(db_input, hw_sampling)
            self.dse = hill_climbing.HillClimbing(self.database, self.result_dir)

            # generate light systems
            start = time.time()
            all_light_systems = self.dse.dh.light_system_gen_exhaustively(system_workers, self.database)
            print("light system generation time: " + str(time.time() - start))
            print("----- all light system generated for process: " + str(mapping_process_id) + "_" + str(FARSI_gen_process_id))

            # generate FARSI systems
            start = time.time()
            all_exs = self.dse.dh.FARSI_system_gen_exhaustively(all_light_systems, system_workers)
            print("FARSI system generation time: " + str(time.time() - start))
            print("----- all FARSI system generated for process: " + str(mapping_process_id) + "_" + str(FARSI_gen_process_id))

            # simulate them
            start = time.time()
            all_sims = []
            for ex_dp in all_exs:
                sim_dp = self.dse.eval_design(ex_dp, self.database)
                if config.RUN_VERIFICATION_PER_GEN or config.RUN_VERIFICATION_PER_NEW_CONFIG  or config.RUN_VERIFICATION_PER_IMPROVMENT:
                    self.dse.gen_verification_data(sim_dp, ex_dp)
                all_sims.append(sim_dp)

            print("simulation time: " + str(time.time() - start))
            print("----- all FARSI system simulated process: " + str(mapping_process_id) + "_" + str(FARSI_gen_process_id))

            # collect data
            latency = [sim.get_dp_stats().get_system_complex_metric("latency") for sim in all_sims]
            power = [sim.get_dp_stats().get_system_complex_metric("power") for sim in all_sims]
            area = [sim.get_dp_stats().get_system_complex_metric("area") for sim in all_sims]
            energy = [sim.get_dp_stats().get_system_complex_metric("energy") for sim in all_sims]
            x = range(0, len(latency))

            # write into a file
            base_dir = os.getcwd()
            result_dir = os.path.join(base_dir, config.exhaustive_result_dir )
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)

            result_in_list_file_addr = os.path.join(result_dir,
                                                     config.exhaustive_output_file_prefix +
                                                     str(mapping_process_id) +"_" + str(FARSI_gen_process_id) + '.txt')
            with open(result_in_list_file_addr, 'w') as f:
                f.write('itr_count: ')
                for listitem in x:
                    f.write('%s,' % str(listitem))

            result_in_pdf_file_addr = os.path.join(result_dir,
                                                   'exhaustive_for_pid' + str(mapping_process_id) + "_" + str(
                                                       FARSI_gen_process_id) + '.txt')
            with open(result_in_pdf_file_addr,
                      'a') as f:
                f.write('\n')
                f.write('latency: ')
                for listitem in latency:
                    f.write('%s,' % str(listitem))

                f.write('\n')
                f.write('power: ')
                for listitem in power:
                    f.write('%s,' % str(listitem))

                f.write('\n')
                f.write('energy: ')
                for listitem in energy:
                    f.write('%s,' % str(listitem))

                f.write('\n')
                f.write('area: ')
                for listitem in area:
                    f.write('%s,' % str(listitem))

            # plot
            for metric in ["latency", "area", "power", "energy"]:
                fig, ax = plt.subplots()
                if metric == "latency":
                    y = [max(list(el.values()))  for el in vars()[metric]]
                else:
                    y = vars()[metric]
                ax.scatter(x, y, marker="x")
                ax.set_xlabel("iteration count")
                ax.set_ylabel(metric)
                fig.savefig("exhaustive_"+metric+"_for_pid_"+ str(mapping_process_id) +"_" + str(FARSI_gen_process_id) +".pdf")

            print("done")
        else:
            print("can not explore exhaustively with dse_type:" + config.dse_type)
            exit(0)

    # ---------------
    # Functionality:
    #       set up an exploration dse.
    #       specify the explorer type in the config file
    # ---------------
    def setup_an_explorer(self, db_input, hw_sampling):
        # body
        if config.dse_type == "hill_climbing" or config.dse_type == "moos":
            exploration_start_time = time.time()  # time hooks (for data collection)
            self.database = DataBase(db_input, hw_sampling)   # initialize the database
            # initializes the design space exploration of certain type
            self.dse = hill_climbing.HillClimbing(self.database, self.result_dir)
        elif config.dse_type == "exhaustive":
            print("this main is not suitable for exhaustive search")
            exit(0)
            #TODO: fix the following. The following commented code is there
            #                         to guide writing the code
            """
            self.database = DataBase(database_input.tasksL, database_input.blocksL,
                                     database_input.pe_mapsL, database_input.pe_schedulesL, database_input.SOCsL)

            # change this later
            self.dse = hill_climbing.HillClimbing(self.database)
            all_exs = self.dse.dh.gen_des_exhaustively()
            all_sims = []
            for ex_dp in all_exs:
                all_sims.append(self.dse.eval_design(ex_dp, self.database))

            latency = [sim.get_dp_stats().get_system_complex_metric("latency") for sim in all_sims]
            plot.scatter_plot(range(0, len(latency)), latency, latency, self.database)
            self.dse = Exhaustive(self.database)
            """

    # populate the IP library from an external source
    # mode: {"python", "csv"}
    def populate_IP_library(self, mode="python"):
        if (mode == "python"):
            for task_name,blocksL in self.database.get_mappable_blocksL_to_tasks().items():
                for task in self.database.get_tasks():
                    if task.name == task_name:
                        for blockL_ in blocksL:
                            IP_library_element = IPLibraryElement()
                            IP_library_element.set_task(task)
                            IP_library_element.set_blockL(blockL_)
                            IP_library_element.generate()
                            self.IP_library.append(IP_library_element)

        # latency
        for metric in ["latency", "energy", "area", "power"]:
            IP_library_dict = defaultdict(dict)
            for IP_library_element in self.IP_library:
                IP_library_dict[IP_library_element.blockL.block_instance_name][IP_library_element.get_task().name] = \
                    IP_library_element.get_PPAC()[metric]

            all_task_names = [task.name for task in self.database.get_tasks()]
            IP_library_dict_ordered = defaultdict(dict)
            for IP_library_key in IP_library_dict.keys():
                for task_name in all_task_names:
                    if task_name in IP_library_dict[IP_library_key].keys(): # check if exists
                        IP_library_dict_ordered[IP_library_key][task_name] = IP_library_dict[IP_library_key][task_name]
                    else:
                        IP_library_dict_ordered[IP_library_key][task_name] = "NA"

            # writing into a file
            fields = ["tasks"] + all_task_names
            with open("IP_library_"+metric+".csv", "w") as f:
                w = csv.DictWriter(f, fields)
                w.writeheader()
                for k in IP_library_dict_ordered:
                    w.writerow({field: IP_library_dict_ordered[k].get(field) or k for field in fields})

    # ---------------
    # Functionality:
    #       prepare the exploration by either generating an initial design (mode ="from scratch")
    #                            or using a check-pointed design
    # Variables:
    #      init_des_point: design point to boost trap the exploration with
    #      boost_SOC: choose a better SOC (This is for multiple SOC design. Not activated yet)
    #      mode: whether to bootstrap exploration from scratch or from an already existing design.
    # ---------------
    def prepare_for_exploration(self, boost_SOC, starting_exploration_mode="from_scratch"):
        # either generate an initial design point(dh.gen_init_des()) or use a check_pointed one
        self.dse.gen_init_ex_dp(starting_exploration_mode)
        self.dse.dh.boos_SOC = boost_SOC

    # ---------------
    # Functionality:
    #       explore the design space
    # ---------------
    def explore(self):
        exploration_start_time = time.time()  # time hook (data collection)
        self.dse.explore_ds()


    # ---------------
    # Functionality:
    #       explore the one design. Basically simulate the design and profile
    # ---------------
    def explore_one_design(self):
        exploration_start_time = time.time()  # time hook (data collection)
        self.dse.explore_one_design()


    # ---------------
    # Functionality:
    #       check point the best design. Check pointing allows to iteratively improve the design by
    #       using the best of the last iteration design.
    # ---------------
    def check_point_best_design(self, unique_number):
        # deactivate check point to prevent running out of memory
        if not config.check_pointing_allowed:
            return

        #  pickle the results for (out of run) verifications.
        # make a directory according to the data/time
        date_time = datetime.now().strftime('%m-%d_%H-%M_%S')
        result_folder = os.path.join(self.result_dir, self.check_point_folder_name)
                                     #date_time + "_" + str(unique_number))
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)
        # pickle the results in it
        if "ex" in config.check_point_list:
            zip_file_name = 'ex_dp_pickled.zip'
            zip_file_addr = os.path.join(result_folder, zip_file_name)
            pickle_file_name = "ex_dp_pickled"+".txt"
            pickle_file_addr = os.path.join(result_folder,pickle_file_name)
            ex_dp_pickled_file = open(pickle_file_addr, "wb")
            dill.dump(self.dse.so_far_best_ex_dp, ex_dp_pickled_file)
            ex_dp_pickled_file.close()

            # remove the old zip file
            if os.path.isfile(zip_file_addr):
                os.remove(zip_file_addr)

            zipObj = ZipFile(zip_file_addr, 'w')
            # Add multiple files to the zip
            zipObj.write(pickle_file_addr, basename(pickle_file_addr))
            # close the Zip File
            zipObj.close()

            # remove the pickle file
            os.remove(pickle_file_addr)

        if "db" in config.check_point_list:
            #database_pickled_file = open(os.path.join(result_folder, "database_pickled"+".txt"), "wb")
            #dill.dump(self.database, database_pickled_file)
            #database_pickled_file.close()
            zip_file_name = 'database_pickled.zip'
            zip_file_addr = os.path.join(result_folder, zip_file_name)
            pickle_file_name = "database_pickled"+".txt"
            pickle_file_addr = os.path.join(result_folder,pickle_file_name)
            database_pickled_file = open(pickle_file_addr, "wb")
            dill.dump(self.database, database_pickled_file)
            #dill.dump(self.dse.so_far_best_ex_dp, ex_dp_pickled_file)
            database_pickled_file.close()

            # remove the old zip file
            if os.path.isfile(zip_file_addr):
                os.remove(zip_file_addr)

            zipObj = ZipFile(zip_file_addr, 'w')
            # Add multiple files to the zip
            zipObj.write(pickle_file_addr, basename(pickle_file_addr))
            # close the Zip File
            zipObj.close()

            # remove the pickle file
            os.remove(pickle_file_addr)

        if "sim" in config.check_point_list:
            sim_dp_pickled_file = open(os.path.join(result_folder, "sim_dp_pickled"+".txt"), "wb")
            dill.dump(self.dse.so_far_best_sim_dp, sim_dp_pickled_file)
            sim_dp_pickled_file.close()
            vis_hardware.vis_hardware(self.dse.so_far_best_ex_dp, config.hw_graphing_mode, result_folder)

        for key, val in self.dse.so_far_best_sim_dp.dp_stats.SOC_metric_dict["latency"]["glass"][0].items():
            print("lat is {} for {}".format(val, key))
            burst_size = config.default_burst_size
            queue_size = config.default_data_queue_size
            print("burst size is {}".format(burst_size))
            print("queue size is {}".format(queue_size))

        #self.dse.write_data_log(list(self.dse.get_log_data()), self.dse.reason_to_terminate, "", result_folder, self.check_point_ctr,
        #              config.FARSI_simple_run_prefix)
        self.check_point_ctr +=1
