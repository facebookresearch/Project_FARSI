#Copyright (c) Facebook, Inc. and its affiliates.
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.

from design_utils.components.hardware import *
from design_utils.components.workload import *
from design_utils.components.mapping import *
from design_utils.components.scheduling import *
from SIM_utils.SIM import *
from design_utils.design import *
from design_utils.des_handler import *
from design_utils.components.krnel import *
from typing import Dict, Tuple, List
from settings import config
from visualization_utils import vis_hardware, vis_stats, plot
from visualization_utils import vis_sim
#from data_collection.FB_private.verification_utils.common import *
import dill
import pickle
import importlib
import difflib
# ------------------------------
# This class is responsible for design space exploration using our proprietary hill-climbing algorithm.
# Our Algorithm currently uses swap (improving the current design) and  duplicate (relaxing the contention on the
# current bottleneck) as two main exploration move.
# ------------------------------
class HillClimbing:
    def __init__(self, database, result_dir):

        # parameters (to configure)
        self.result_dir = result_dir
        self.fitted_budget_ctr = 0  # counting the number of times that we were able to find a design to fit the budget. Used to terminate the search
        self.name_ctr = 0
        self.DES_STAG_THRESHOLD = config.DES_STAG_THRESHOLD   # Acceptable iterations count without improvement before termination.
        self.TOTAL_RUN_THRESHOLD = config.TOTAL_RUN_THRESHOLD  # Total  number of iterations to terminate with.
        self.neigh_gen_mode = config.neigh_gen_mode   # Neighbouring design pts generation mode ("all" "random_one").
        self.num_neighs_to_try = config.num_neighs_to_try  # How many neighs to try around a current design point.
        self.neigh_sel_mode = config.neigh_sel_mode  # Neighbouring design selection mode (best, sometimes, best ...)
        self.dp_rank_obj = config.dp_rank_obj  # Design point ranking object function(best, sometimes, best ...)
        self.num_clusters = config.num_clusters  # How many clusters to create everytime we split.
        self.budget_coeff = config.max_budget_coeff
        self.move_profile = []
        # variables (to initialize)
        self.area_explored = []  # List containing area associated with the all explored designs areas.
        self.latency_explored = []  # List containing latency associated with all explored designs latency.
        self.power_explored = []    # List containing Power associated with all explored designs latency.
        self.design_itr = []  # Design iteration counter. Simply indexing (numbering) the designs explored.
        self.space_distance = 2
        self.database = database  # hw/sw database to use for exploration.
        self.dh = DesignHandler(self.database)  # design handler for design modification.
        self.so_far_best_sim_dp = None     # best design found so far through out all iterations.
                                           # For iteratively improvements.
        self.cur_best_ex_dp, self.cur_best_sim_dp = None, None   # current iteration's best design.
        self.last_des_trail = None # last design (trail)
        self.last_move = None # last move applied
        self.init_ex_dp = None  # Initial exploration design point. (Staring design point for the whole algorithm)
        self.coeff_slice_size = int(self.TOTAL_RUN_THRESHOLD/config.max_budget_coeff)
        #self.hot_krnl_pos = 0  # position of the kernel among the kernel list. Used to found and improve the
                               # corresponding occupying block.

        # Counters: to determine which control path the exploration should take (e.g., terminate, pick another block instead
        # of hotblock, ...).
        self.des_stag_ctr = 0  # Iteration count since seen last design improvement.
        self.total_itr_ctr = 0  # Total iteration count (for termination purposes).

        self.vis_move_trail_ctr = 0
        # Sanity checks (preventing bad configuration setup)
        if self.neigh_gen_mode not in ["all", "some"]: raise ValueError()
        # TODO: sel_cri needs to be fixed to include combinations of the objective functions
        if self.dp_rank_obj not in ["all", "latency", "throughput", "power", "design_cost"]: raise ValueError()
        if self.neigh_sel_mode not in ["best", "best_sometime"]: raise ValueError()
        self.des_trail_list = []
        self.krnel_rnk_to_consider = 0   # this rank determines which kernel (among the sorted kernels to consider).
                                         # we use this counter to avoid getting stuck
        self.krnel_stagnation_ctr = 0    # if the same kernel is selected across iterations and no improvement observed,
                                         # count up

        self.recently_seen_design_ctr = 0
        self.recently_cached_designs = []
        self.cleanup_ctr = 0  # use this to invoke the cleaner once we pass a threshold

        self.SA_current_breadth = -1  # which breadth is current move on
        self.SA_current_depth = -1  # which depth is current move on
        self.check_point_folder = config.check_point_folder

        self.seen_SOC_design_codes = []  # config code of all the designs seen so far (this is mainly for debugging, concretely
                                     # simulation validation

        self.cached_SOC_sim = {} # cache of designs simulated already. index is a unique code base on allocation and mapping

    def set_check_point_folder(self, check_point_folder):
        self.check_point_folder = check_point_folder

    # retrieving the pickled check pointed file
    def get_pickeld_file(self, file_addr):
        with open(file_addr, 'rb') as f:  # will close() when we leave this block
            obj = pickle.load(f)
        return obj

    # ------------------------------
    # Functionality
    #       generate initial design point to start the exploration from.
    #       If mode is from_scratch, the default behavior is to pick the cheapest design.
    #       If mode is check_pointed, we start from a previously check pointed design.
    #       If mode is hardcode, we pick a design that is hardcoded.
    # Variables
    #       init_des_point: initial design point
    #       mode: starting point mode (from scratch or from check point)
    # ------------------------------
    def gen_init_ex_dp(self, init_des_point="None", mode="generated_from_scratch"):
        if mode == "generated_from_scratch":  # start from the simplest design possible
            self.init_ex_dp = self.dh.gen_init_des()
        elif mode == "generated_from_check_point":
            pickled_file_addr = self.check_point_folder + "/" + "ex_dp_pickled.txt"
            database_file_addr = self.check_point_folder + "/" + "database_pickled.txt"
            self.database = self.get_pickeld_file(database_file_addr)
            self.init_ex_dp = self.get_pickeld_file(pickled_file_addr)
        elif mode == "hardcoded":
            self.init_ex_dp = self.dh.gen_specific_hardcoded_ex_dp(self.dh.database)
        elif mode == "parse":
            self.init_ex_dp = self.dh.gen_specific_parsed_ex_dp(self.dh.database)
        else: raise Exception("mode:" + mode + " is not supported")

    # ------------------------------
    # Functionality:
    #       Generate one neighbouring design based on the moves available.
    #       To do this, we first specify a move and then apply it.
    #       A move is specified by a metric, direction kernel, block, and transformation.
    #       look for move definition in the move class
    # Variables
    #       des_tup: design tuple. Contains a design tuple (ex_dp, sim_dp). ex_dp: design to find neighbours for.
    #                                                                       sim_dp: simulated ex_dp.
    # ------------------------------
    def gen_one_neigh(self, des_tup):
        ex_dp, sim_dp = des_tup

        # Copy to avoid modifying the current designs.
        #new_ex_dp_pre_mod = copy.deepcopy(ex_dp)  # getting a copy before modifying
        #new_sim_dp_pre_mod = copy.deepcopy(sim_dp) # getting a copy before modifying
        new_ex_dp = copy.deepcopy(ex_dp)
        #new_sim_dp = copy.deepcopy(sim_dp)
        new_des_tup = (new_ex_dp, sim_dp)

        # ------------------------
        # select (generate) a move
        # ------------------------
        # It's important that we do analysis of move selection on the copy (and not the original) because
        # 1. we'd like to keep original for further modifications
        # 2. for block identification/comparison of the move and the copied design
        safety_chk_passed = False
        # iterate and continuously generate moves, until one passes some sanity check
        while not safety_chk_passed:
            move_to_try = self.sel_moves(new_des_tup, "dist_rank")
            safety_chk_passed = move_to_try.safety_check(new_ex_dp)

        #move_to_try.print_info()

        # ------------------------
        # apply the move
        # ------------------------
        # while conduction various validity/sanity checks
        try:
            self.dh.unload_read_mem(new_des_tup[0])    # unload read memories
            move_to_try.validity_check()  # call after unload rad mems, because we need to check the scenarios where
                                          # task is unloaded from the mem, but was decided to be migrated/swapped
            new_ex_dp_res, succeeded = self.dh.apply_move(new_des_tup, move_to_try)
            move_to_try.set_before_after_designs(new_des_tup[0], new_ex_dp_res)
            new_ex_dp_res.sanity_check()  # sanity check
            move_to_try.sanity_check()
            self.dh.load_tasks_to_read_mem_and_ic(new_ex_dp_res)  # loading the tasks on to memory and ic
            new_ex_dp_res.sanity_check()
        except Exception as e:
            # if the error is already something that we are familiar with
            # react appropriately, otherwise, simply raise it.
            if e.__class__.__name__ in errors_names:
                print("Error: " + e.__class__.__name__)
                # TODOs
                # for now, just return the previous design, but this needs to be fixed immediately
                new_ex_dp_res = ex_dp
                #raise e
            elif e.__class__.__name__ in exception_names:
                print("Exception: " + e.__class__.__name__)
                new_ex_dp_res = ex_dp
            else:
                raise e

        return new_ex_dp_res, move_to_try

    # ------------------------------
    # Functionality:
    #   Select a item given a probability distribution.
    #   The input provides a list of values and their probabilities/fitness,...
    #   and this function randomly picks a value based on the fitness/probability, ...
    #   Used for random but prioritized selections (of for example blocks, or kernels)
    # input: item_prob_dict {} (item, probability)
    # ------------------------------
    def pick_from_prob_dict(self, item_prob_dict):
        # now encode this priorities into a encoded histogram (for example with performance
        # encoded as 1 and power as 2 and ...) with frequencies
        if config.DEBUG_FIX: random.seed(0)
        else: time.sleep(.00001), random.seed(datetime.now().microsecond)

        item_encoding = np.arange(0, len(item_prob_dict.keys()))  # encoding the metrics from 1 to ... clusters
        rand_var_dis = list(item_prob_dict.values())  # distribution

        encoded_metric = np.random.choice(item_encoding, p=rand_var_dis)  # cluster (metric) selected
        selected_item = list(item_prob_dict.keys())[encoded_metric]
        return selected_item

    # ------------------------------
    # Functionality:
    #       return all the hardware blocks that have same hardware characteristics (e.g, same type and same work-rate and mappability)
    # ------------------------------
    def find_matching_blocks(self, blocks):
        matched_idx = []
        matching_blocks = []
        for idx, _ in enumerate(blocks):
            if idx in matched_idx:
                continue
            matched_idx.append(idx)
            for idx_2 in range(idx+1, len(blocks)):
                if blocks[idx].get_generic_instance_name() == blocks[idx_2].get_generic_instance_name():  # for PEs
                    matching_blocks.append((blocks[idx], blocks[idx_2]))
                    matched_idx.append(idx_2)
                elif blocks[idx].type in ["mem", "ic"] and blocks[idx_2].type in ["mem", "ic"]:  # for mem and ic
                    if blocks[idx].type == blocks[idx_2].type:
                        matching_blocks.append((blocks[idx], blocks[idx_2]))
                        matched_idx.append(idx_2)
        return matching_blocks

    # check if there is another task (on the block that can run in parallel with the task of interest
    def check_if_task_can_run_with_any_other_task_in_parallel(self, sim_dp, task, block):
        tasks_of_block = [task_ for task_ in block.get_tasks_of_block() if (not ("souurce" in task_.name) or not ("siink" in task_.name))]
        for task_ in tasks_of_block:
            if sim_dp.get_dp_rep().get_hardware_graph().get_task_graph().tasks_can_run_in_parallel(task_, task):
                return True
        return False


    # ------------------------------
    # Functionality:
    #       check if there are any tasks across two blocks that can be run in parallel
    #       this is used for cleaning up, if there is not opportunities for parallelization
    # Variables:
    #        sim_dp: design
    # ------------------------------
    def check_if_any_tasks_on_two_blocks_parallel(self, sim_dp, block_1, block_2):
        tasks_of_block_1 = [task for task in block_1.get_tasks_of_block() if (not("souurce" in task.name) or not("siink" in task.name))]
        tasks_of_block_2 = [task for task in block_2.get_tasks_of_block() if (not("souurce" in task.name) or not("siink" in task.name))]

        for idx_1, _ in enumerate(tasks_of_block_1):
            for idx_2, _ in enumerate(tasks_of_block_2):
                if sim_dp.get_dp_rep().get_hardware_graph().get_task_graph().tasks_can_run_in_parallel(tasks_of_block_2[idx_2], tasks_of_block_1[idx_1]):
                    return True

        return False

    # ------------------------------
    # Functionality:
    #       Return all the blocks that are unnecessarily parallelized, i.e., there are no
    #       tasks across them that can run in parallel.
    #       this is used for cleaning up, if there is not opportunities for parallelization
    # Variables:
    #        sim_dp: design
    #        matching_blocks_list: list of hardware blocks with equivalent characteristics (e.g., two A53, or two
    #        identical acclerators)
    # ------------------------------
    def find_blocks_with_all_serial_tasks(self, sim_dp, matching_blocks_list):
        matching_blocks_list_filtered = []
        for matching_blocks in matching_blocks_list:
            if self.check_if_any_tasks_on_two_blocks_parallel(sim_dp, matching_blocks[0], matching_blocks[1]):
                continue
            matching_blocks_list_filtered.append((matching_blocks[0], matching_blocks[1]))

        return matching_blocks_list_filtered

    # ------------------------------
    # Functionality:
    #      search through all the blocks and return a pair of blocks that cleanup can  apply to
    # Variables:
    #        sim_dp: design
    # ------------------------------
    def pick_block_pair_to_clean_up(self, sim_dp, block_pairs):
        if len(block_pairs) == 0:
            return block_pairs

        cleanup_ease_list = []
        block_pairs_sorted = []  # sorting the pairs elements (within each pair) based on the number of tasks on each
        for blck_1, blck_2 in block_pairs:
            if blck_2.type == "ic":  # for now ignore ics
                continue
            elif blck_2.type == "mem":
                if self.database.check_superiority(blck_1, blck_2):
                    block_pairs_sorted.append((blck_1, blck_2))
                else:
                    block_pairs_sorted.append((blck_2, blck_1))
            else:
                if len(blck_1.get_tasks_of_block()) < len(blck_2.get_tasks_of_block()):
                    block_pairs_sorted.append((blck_1, blck_2))
                else:
                    block_pairs_sorted.append((blck_2, blck_1))

            distance = len(sim_dp.get_dp_rep().get_hardware_graph().get_path_between_two_vertecies(blck_1, blck_2))
            num_tasks_to_move = min(len(blck_1.get_tasks_of_block()), len(blck_2.get_tasks_of_block()))

            cleanup_ease_list.append(distance + num_tasks_to_move)

        # when we need to clean up the ics, ignore for now
        if len(cleanup_ease_list) == 0:
            return []

        picked_easiest = False
        min_ease = 100000000
        for idx, ease in enumerate(cleanup_ease_list):
            if ease < min_ease:
                picked_easiest = True
                easiest_pair = block_pairs_sorted[idx]
                min_ease = ease
        return easiest_pair

    # ------------------------------
    # Functionality:
    #      used to determine if two different task can use the same accelerators.
    # ------------------------------
    def are_same_ip_tasks(self, task_1, task_2):
        return (task_1.name, task_2.name) in self.database.db_input.misc_data["same_ip_tasks_list"] or (task_2.name, task_1.name) in self.database.db_input.misc_data["same_ip_tasks_list"]

    # ------------------------------
    # Functionality:
    #     find all the tasks that can run on the same ip (accelerator)
    # Variables:
    #      des_tup: (design, simulated design)
    # ------------------------------
    def find_task_with_similar_mappable_ips(self, des_tup):
        ex_dp, sim_dp = des_tup
        #krnls = sim_dp.get_dp_stats().get_kernels()
        blcks = ex_dp.get_blocks()
        pe_blocks = [blck for blck in blcks if blck.type=="pe"]
        tasks_sub_ip_type = [] # (task, sub_ip)
        matches = []
        for blck in pe_blocks:
            tasks_sub_ip_type.extend(zip(blck.get_tasks_of_block(), [blck.subtype]*len(blck.get_tasks_of_block())))

        for task, sub_ip_type in tasks_sub_ip_type:
            check_for_similarity = False
            if sub_ip_type == "ip":
                check_for_similarity = True
            if not check_for_similarity:
                continue

            for task_2, sub_ip_type_2 in tasks_sub_ip_type:
                if task_2.name == task.name :
                    continue
                if self.are_same_ip_tasks(task, task_2):
                    for blck in pe_blocks:
                        if task_2 in blck.get_tasks_of_block():
                            block_to_migrate_from = blck
                        if task in blck.get_tasks_of_block():
                            block_to_migrate_to = blck

                    if not (block_to_migrate_to == block_to_migrate_from):
                        matches.append((task_2, block_to_migrate_from, task, block_to_migrate_to))

        if len(matches) == 0:
            return None, None, None, None
        else:
           return random.choice(matches)

    # ------------------------------
    # Functionality:
    #     pick a block pair to apply cleaning to
    # Variables:
    #      des_tup: (design, simulated design)
    # ------------------------------
    def gen_block_match_cleanup_move(self, des_tup):
        ex_dp, sim_dp = des_tup
        krnls = sim_dp.get_dp_stats().get_kernels()
        blcks = ex_dp.get_blocks()

        # move tasks to already generated IPs
        # clean up the matching blocks
        matching_blocks_list = self.find_matching_blocks(blcks)
        matching_blocks_lists_filtered = self.find_blocks_with_all_serial_tasks(sim_dp, matching_blocks_list)
        return self.pick_block_pair_to_clean_up(sim_dp, matching_blocks_lists_filtered)

    # ------------------------------
    # Functionality:
    #     is current iteration a clean up iteration (ie., should be used for clean up)
    # ------------------------------
    def is_cleanup_iter(self):
        return (self.cleanup_ctr % (config.cleaning_threshold)) > (config.cleaning_threshold- config.cleaning_consecutive_iterations)

    # ------------------------------
    # Functionality:
    #    pick which transformation to apply
    # Variables:
    #      hot_blck_synced: the block bottleneck
    #      selected_metric: metric to focus on
    #      selected_krnl: the kernel to focus on
    # ------------------------------
    def select_transformation(self, ex_dp, sim_dp, hot_blck_synced, selected_metric, selected_krnl, selected_dir):
        imm_block = self.dh.get_immediate_block(hot_blck_synced, selected_metric, selected_dir,  hot_blck_synced.get_tasks_of_block())
        task_synced = [task__ for task__ in hot_blck_synced.get_tasks_of_block() if task__.name == selected_krnl.get_task_name()][0]
        feasible_transformations = set(config.metric_trans_dict[selected_metric])

        # find the block that is at least as good as the block (for migration)
        # if can't find any, we return the same block
        equal_imm_block_present_for_migration = self.dh.get_equal_immediate_block_present(ex_dp, hot_blck_synced,
                                                                selected_metric, selected_dir, [task_synced])

        # filter transformations accordingly
        if equal_imm_block_present_for_migration == hot_blck_synced:
            # if can't find a block that is at least as good as the current block, can't migrate
            feasible_transformations =  set(list(feasible_transformations - set(['migrate'])))
        #else:
        #   print("now find now ok")
        if len(hot_blck_synced.get_tasks_of_block()) == 1:  # can't split an accelerator
            feasible_transformations =  set(list(feasible_transformations - set(['split'])))
        if imm_block.get_generic_instance_name() == hot_blck_synced.get_generic_instance_name():
            # if can't swap improve, get rid of swap
            feasible_transformations = set(list(feasible_transformations - set(['swap'])))
        if hot_blck_synced.type in ["ic", "mem"]  and selected_metric == "latency" and selected_dir  == -1:
            # if no other task to run in parallel with, don't split
            if not (self.check_if_task_can_run_with_any_other_task_in_parallel(sim_dp, task_synced, hot_blck_synced)):
                feasible_transformations = set(list(feasible_transformations - set(['split', 'migrate'])))
            # delete the following line right after debugging
            #feasible_transformations = set(list(feasible_transformations - set(['split', 'migrate'])))
        if hot_blck_synced.type in ["ic"]:
            # we don't cover migrate for ICs at the moment
            # TODO: add this feature later
            feasible_transformations = set(list(feasible_transformations - set(['migrate'])))

        # if no valid transformation left, issue the identity transformation (where nothing changes and a simple copying is done)
        if len(list(feasible_transformations)) == 0:
            feasible_transformations = ["identity"]

        print(list(feasible_transformations))
        random.seed(datetime.now().microsecond)
        transformation = random.choice(list(feasible_transformations))
        if transformation == "migrate":
            batch_mode = "single"
        elif transformation == "split":
            # see if any task can run in parallel
            if self.dh.can_any_task_on_block_run_in_parallel(ex_dp, selected_krnl.get_task(), hot_blck_synced):
                batch_mode = "batch"
            else:
                batch_mode = "single"
        else:
            batch_mode = "irrelevant"
        return transformation, batch_mode

    # calculate the cost impact of a kernel improvement
    def get_swap_improvement_cost(self, sim_dp, kernels, selected_metric, dir):
        def get_subtype_for_cost(block):
            if block.type == "pe" and block.subtype == "ip":
                return "ip"
            if block.type == "pe" and block.subtype == "gpp":
                if "A53" in block.instance_name or "ARM" in block.instance_name:
                    return "arm"
                if "G3" in block.instance_name:
                    return "dsp"
            else:
                return block.type

        # Figure out whether there is a mapping that improves kernels performance
        def no_swap_improvement_possible(sim_dp, selected_metric, dir, krnl):
            hot_block = sim_dp.get_dp_stats().get_hot_block_of_krnel(krnl.get_task_name(), selected_metric)
            imm_block = self.dh.get_immediate_block(hot_block, selected_metric, dir, [krnl.get_task()])
            blah  = hot_block.get_generic_instance_name()
            blah2  = imm_block.get_generic_instance_name()
            return hot_block.get_generic_instance_name() == imm_block.get_generic_instance_name()


        # find the cost of improvement by comparing the current and accelerated design (for the kernel)
        kernel_improvement_cost = {}
        kernel_name_improvement_cost = {}
        for krnel in kernels:
            hot_block = sim_dp.get_dp_stats().get_hot_block_of_krnel(krnel.get_task_name(), selected_metric)
            hot_block_subtype = get_subtype_for_cost(hot_block)
            current_cost = self.database.db_input.porting_effort[hot_block_subtype]
            #if hot_block_subtype == "ip":
            #    print("what")
            imm_block = self.dh.get_immediate_block(hot_block,selected_metric, dir,[krnel.get_task()])
            imm_block_subtype = get_subtype_for_cost(imm_block)
            imm_block_cost =  self.database.db_input.porting_effort[imm_block_subtype]
            improvement_cost = (imm_block_cost - current_cost)
            kernel_improvement_cost[krnel] = improvement_cost

        # calcualte inverse so lower means worse
        max_val =  max(kernel_improvement_cost.values()) # multiply by
        kernel_improvement_cost_inverse = {}
        for k, v in kernel_improvement_cost.items():
            kernel_improvement_cost_inverse[k] = max_val - kernel_improvement_cost[k]

        # get sum and normalize
        sum_ = sum(list(kernel_improvement_cost_inverse.values()))
        for k, v in kernel_improvement_cost_inverse.items():
            # normalize
            if not (sum_ == 0):
                kernel_improvement_cost_inverse[k] = kernel_improvement_cost_inverse[k]/sum_
            kernel_improvement_cost_inverse[k] = max(kernel_improvement_cost_inverse[k], .0000001)
            if no_swap_improvement_possible(sim_dp, selected_metric, dir, k):
                kernel_improvement_cost_inverse[k] = .0000001
            kernel_name_improvement_cost[k.get_task_name()] = kernel_improvement_cost_inverse[k]

        return kernel_improvement_cost_inverse

    # select a metric to improve on
    def select_metric(self, sim_dp):
        # prioritize metrics based on their distance contribution to goal
        metric_prob_dict = {}  # (metric:priority value) each value is in [0 ,1] interval
        for metric in config.budgetted_metrics:
            metric_prob_dict[metric] = sim_dp.dp_stats.dist_to_goal_per_metric(metric, config.metric_sel_dis_mode)/\
                                                   sim_dp.dp_stats.dist_to_goal(["power", "area", "latency"],
                                                                                config.metric_sel_dis_mode)

        # sort the metric based on distance (and whether the sort is probabilistic or exact).
        # probabilistic sorting, first sort exactly, then use the exact value as a probability of selection
        metric_prob_dict_sorted = {k: v for k, v in sorted(metric_prob_dict.items(), key=lambda item: item[1])}
        if config.move_metric_ranking_mode== "exact":
            selected_metric = list(metric_prob_dict_sorted.keys())[len(metric_prob_dict_sorted.keys()) -1]
        else:
            selected_metric = self.pick_from_prob_dict(metric_prob_dict_sorted)
        return selected_metric, metric_prob_dict

    def select_dir(self, sim_dp):
        move_dir = 1  # try to increase the metric value
        if not sim_dp.dp_stats.fits_budget(1):
            move_dir = -1  # try to reduce the metric value
        return move_dir

    def select_kernel(self, sim_dp, selected_metric, move_dir):
        krnl_prob_dict = {}  # (kernel, metric_value)
        krnls = sim_dp.get_dp_stats().get_kernels()
        metric_total = sum([krnl.stats.get_metric(selected_metric) for krnl in krnls])
        # sort kernels based on their contribution to the metric of interest
        for krnl in krnls:
            krnl_prob_dict[krnl] = krnl.stats.get_metric(selected_metric)/metric_total

        # only if cost and bottleneck should be considered simultenously
        krnl_improvement_cost = self.get_swap_improvement_cost(sim_dp, krnls, selected_metric, move_dir)
        for krnl, prob in krnl_prob_dict.items():
            krnl_prob_dict[krnl] = prob * krnl_improvement_cost[krnl]

        # sort
        krnl_prob_dict_sorted = {k: v for k, v in sorted(krnl_prob_dict.items(), key=lambda item: item[1])}
        # quick sanity check
        if not(config.move_krnel_ranking_mode == "exact") and sum(krnl_prob_dict_sorted.values()) < .99:
            x = sum(krnl_prob_dict_sorted.values())
            print("This should not happen")

        if config.move_krnel_ranking_mode == "exact":  # for area to allow us pick scenarios that are not necessarily the worst
            selected_krnl = list(krnl_prob_dict_sorted.keys())[
                len(krnl_prob_dict_sorted.keys()) - 1 - self.krnel_rnk_to_consider]
        else:
            selected_krnl = self.pick_from_prob_dict(krnl_prob_dict_sorted)
        return selected_krnl, krnl_prob_dict

    def select_block(self, sim_dp, ex_dp, selected_krnl, selected_metric):
        # get the hot block for the kernel. Hot means the most contributing block for the kernel/metric of interest
        hot_blck = sim_dp.get_dp_stats().get_hot_block_of_krnel(selected_krnl.get_task_name(), selected_metric)
        # hot_blck_synced is the same block but ensured that the block instance
        # is chosen from ex instead of sim, so it can be modified
        hot_blck_synced = self.dh.find_cores_hot_kernel_blck_bottlneck(ex_dp, hot_blck)
        block_prob_dict = sim_dp.get_dp_stats().get_hot_block_of_krnel_sorted(selected_krnl.get_task_name(), selected_metric)
        return hot_blck_synced, block_prob_dict

    def select_block_without_sync(self, sim_dp, selected_krnl, selected_metric):
        # get the hot block for the kernel. Hot means the most contributing block for the kernel/metric of interest
        hot_blck = sim_dp.get_dp_stats().get_hot_block_of_krnel(selected_krnl.get_task_name(), selected_metric)
        # hot_blck_synced is the same block but ensured that the block instance
        # is chosen from ex instead of sim, so it can be modified
        block_prob_dict = sim_dp.get_dp_stats().get_hot_block_of_krnel_sorted(selected_krnl.get_task_name(), selected_metric)
        return hot_blck, block_prob_dict

    # ------------------------------
    # Functionality:
    #    generate a move to apply.  A move consists of a metric, direction, kernel, block and transformation.
    #    At the moment, we target the metric that is most further from the budget. Kernel and block are chosen
    #    based on how much they contribute to the distance.
    #
    # Variables:
    #      des_tup: (design, simulated design)
    # ------------------------------
    def sel_moves_based_on_dis(self, des_tup):
        ex_dp, sim_dp = des_tup
        if config.DEBUG_FIX: random.seed(0)
        else: time.sleep(.00001), random.seed(datetime.now().microsecond)

        # select move components
        selected_metric, metric_prob_dict = self.select_metric(sim_dp)
        if selected_metric == "power":
            print("what make sure now to del")
        move_dir = self.select_dir(sim_dp)
        selected_krnl, krnl_prob_dict = self.select_kernel(sim_dp, selected_metric, move_dir)
        selected_block, block_prob_dict = self.select_block(sim_dp, ex_dp, selected_krnl, selected_metric)
        transformation_name,transformation_batch_mode = self.select_transformation(ex_dp, sim_dp, selected_block, selected_metric, selected_krnl, move_dir)

        # prepare for move
        # if bus, (forgot which exception), if IP, avoid split .
        if sim_dp.dp_stats.fits_budget(1) or self.is_cleanup_iter():
            transformation_name = "cleanup"
            transformation_batch_mode = "single"
            selected_metric = "cost"
            #config.VIS_GR_PER_GEN = True
            self.cleanup_ctr += 1
            #config.VIS_GR_PER_GEN = False

        # log the data for future profiling/data collection/debugging
        move_to_apply = move(transformation_name, transformation_batch_mode, move_dir, selected_metric, selected_block, selected_krnl)
        move_to_apply.set_logs(sim_dp.dp_stats.get_system_complex_metric("cost"), "cost")
        move_to_apply.set_logs(krnl_prob_dict, "kernels")
        move_to_apply.set_logs(metric_prob_dict, "metrics")
        move_to_apply.set_logs(block_prob_dict, "blocks")
        move_to_apply.set_logs(self.krnel_rnk_to_consider, "kernel_rnk_to_consider")
        move_to_apply.set_logs(sim_dp.dp_stats.dist_to_goal(["power", "area", "latency"],
                                                                                config.metric_sel_dis_mode),"dist_to_goal")
        # ------------------------
        # prepare for the move
        # ------------------------
        if move_to_apply.get_transformation_name() == "identity":
            move_to_apply.set_validity(False, "NoValidTransformationException")
        if move_to_apply.get_transformation_name() == "swap":
            self.dh.unload_read_mem(ex_dp)  # unload memories
            if not move_to_apply.get_block_ref().type == "ic":
                self.dh.unload_buses(ex_dp)  # unload buses
            else:
                self.dh.unload_read_buses(ex_dp)  # unload buses
            # get immediate superior/inferior block (based on the desired direction)
            imm_block = self.dh.get_immediate_block(move_to_apply.get_block_ref(),
                                                 move_to_apply.get_metric(), move_to_apply.get_dir(),
                                                 move_to_apply.get_block_ref().get_tasks_of_block())  # immediate block either superior or
            move_to_apply.set_dest_block(imm_block)

            move_to_apply.set_tasks(move_to_apply.get_block_ref().get_tasks_of_block())
        elif move_to_apply.get_transformation_name() in ["split"]:
            # select tasks to migrate
            migrant_tasks = self.dh.migrant_selection(ex_dp, move_to_apply.get_block_ref(), move_to_apply.get_kernel_ref(),
                                                      move_to_apply.get_transformation_batch())
            #migrant_tasks = self.dh.migrant_selection(move_to_apply.get_block_ref(), move_to_apply.get_kernel_ref(),
            #                                          tch", sim_dp.dp_stats.get_parallel_kernels())
            move_to_apply.set_tasks(migrant_tasks)
            if len(migrant_tasks) == 0:
                move_to_apply.set_validity(False, "NoParallelTaskException")
            if move_to_apply.get_block_ref().subtype == "ip": # makes no sense to split the IPs,
                                                              # it can actually cause problems where
                                                              # we end up duplicating the hardware
                move_to_apply.set_validity(False, "IPSplitException")
        elif move_to_apply.get_transformation_name() == "migrate":
            self.dh.unload_buses(ex_dp)  # unload buses
            self.dh.unload_read_mem(ex_dp)  # unload memories
            if not selected_block.type == "ic":  # ic migration is not supported
                migrant_tasks = self.dh.migrant_selection(sim_dp, move_to_apply.get_block_ref(), move_to_apply.get_kernel_ref(),
                                                          move_to_apply.get_transformation_batch())
                imm_block_present = self.dh.get_equal_immediate_block_present(ex_dp, move_to_apply.get_block_ref(),
                                                                     move_to_apply.get_metric(), move_to_apply.get_dir(),
                                                                     migrant_tasks)  # immediate block either superior or

                # TODO: this is caused by detecting a memory read as the bottlenck, hence needs to be fixed
                if len(migrant_tasks) == 0:
                    move_to_apply.set_validity(False, "NoParallelTaskException")
                move_to_apply.set_tasks(migrant_tasks)
                move_to_apply.set_dest_block(imm_block_present)
            else:
                move_to_apply.set_validity(False, "ICMigrationException")
        elif move_to_apply.get_transformation_name() == "cleanup":
            self.dh.unload_buses(ex_dp)  # unload buses
            self.dh.unload_read_mem(ex_dp)  # unload memories
            task_1, block_task_1, task_2, block_task_2 = self.find_task_with_similar_mappable_ips(des_tup)
            # we also randomize
            if not (task_1 is None) and (random.choice(np.arange(0,1,.1))>.5):
                move_to_apply.set_ref_block(block_task_1)
                migrant_tasks = [task_1]
                imm_block_present = block_task_2
                move_to_apply.set_tasks(migrant_tasks)
                move_to_apply.set_dest_block(imm_block_present)
            else:
                pair = self.gen_block_match_cleanup_move(des_tup)
                if len(pair) == 0 or True:
                    move_to_apply.set_validity(False, "CostPairingException")
                else:
                    ref_block = pair[0]
                    if not ref_block.type == "ic":  # ic migration is not supported
                        move_to_apply.set_ref_block(ref_block)
                        migrant_tasks = ref_block.get_tasks_of_block()
                        imm_block_present = pair[1]
                        move_to_apply.set_tasks(migrant_tasks)
                        move_to_apply.set_dest_block(imm_block_present)


        move_to_apply.set_breadth_depth(self.SA_current_breadth, self.SA_current_depth)  # set depth and breadth (for debugging/ plotting)
        return move_to_apply

    # ------------------------------
    # Functionality:
    #       How to choose the move.
    # Variables:
    #      des_tup: (design, simulated design)
    # ------------------------------
    def sel_moves(self, des_tup, mode="dist_rank"):  # TODO: add mode
        if mode == "dist_rank":  # rank and choose probabilistically based on distance
            return self.sel_moves_based_on_dis(des_tup)
        else:
            print("mode" + mode + " is not supported")
            exit(0)

    # ------------------------------
    # Functionality:
    #       Calculate possible neighbours, though not randomly.
    #       des_tup: design tuple. Contains a design tuple (ex_dp, sim_dp). ex_dp: design to find neighbours for.
    #                                                                       sim_dp: simulated ex_dp.
    # ------------------------------
    def gen_some_neighs_orchestrated(self, des_tup):
        all_possible_moves = config.navigation_moves
        ctr = 0
        kernel_pos_to_hndl = self.hot_krnl_pos  # for now, but change it

        # generate neighbours until you hit the threshold
        while(ctr < self.num_neighs_to_try):
            ex_dp, sim_dp = des_tup
            # Copy to avoid modifying the current designs.
            new_ex_dp_1 = copy.deepcopy(ex_dp)
            new_sim_dp_1 = copy.deepcopy(sim_dp)
            new_ex_dp = copy.deepcopy(new_ex_dp_1)
            new_sim_dp = copy.deepcopy(new_sim_dp_1)

            # apply the move
            yield self.dh.apply_move(new_ex_dp, new_sim_dp, all_possible_moves[ctr%len(all_possible_moves)], kernel_pos_to_hndl)
            ctr += 1
        return 0

    def simulated_annealing_energy(self, sim_dp_stats):
        return ()

    # find the best design from a list
    def find_best_design(self, sim_dp_stat_ann_delta_energy_dict, best_sim_dp_stat_so_far):
        def blocks_are_equal(block_1, block_2):
            if not selected_block.get_generic_instance_name() == best_sim_selected_block.get_generic_instance_name():
                return False
            elif selected_block.get_generic_instance_name() == best_sim_selected_block.get_generic_instance_name():
                # make sure tasks are not the same.
                # this is to avoid scenarios where a block is improved (but it's generic name) equal to the
                # next block bottleneck. Here we make sure tasks are different
                block_1_tasks = [tsk.name for tsk in block_1.get_tasks_of_block()]
                block_2_tasks = [tsk.name for tsk in block_1.get_tasks_of_block()]
                task_diff = list(set(block_1_tasks) - set(block_2_tasks))
                return len(task_diff) == 0

        # sort the design base on distance
        sorted_sim_dp_stat_ann_delta_energy_dict = sorted(sim_dp_stat_ann_delta_energy_dict.items(), key=lambda x: x[1])
        #best_neighbour_stat, best_neighbour_delta_energy = sorted_sim_dp_stat_ann_delta_energy_dict[0]  # here we can be smarter

        # get the best_sim info
        sim_dp = best_sim_dp_stat_so_far.dp
        best_sim_selected_metric, metric_prob_dict = self.select_metric(sim_dp)
        best_sim_move_dir = self.select_dir(sim_dp)
        best_sim_selected_krnl, krnl_prob_dict = self.select_kernel(sim_dp, best_sim_selected_metric, best_sim_move_dir)
        best_sim_selected_block, block_prob_dict = self.select_block_without_sync(sim_dp, best_sim_selected_krnl, best_sim_selected_metric)

        if sorted_sim_dp_stat_ann_delta_energy_dict[0][1] < 0:
            # if a better design (than the best exist), return
            return sorted_sim_dp_stat_ann_delta_energy_dict[0], True
        elif sorted_sim_dp_stat_ann_delta_energy_dict[0][1] == 0:
            if len(sorted_sim_dp_stat_ann_delta_energy_dict[0]) == 1:
                return sorted_sim_dp_stat_ann_delta_energy_dict[0], False
            else:
                # filter out the designs  which hasn't seen a distance improvement
                sim_dp_to_select_from = []
                for sim_dp_stat, energy in sorted_sim_dp_stat_ann_delta_energy_dict:
                    if energy == 0:
                        sim_dp_to_select_from.append((sim_dp_stat, energy))

                designs_to_consider = []
                for sim_dp_stat, energy in sim_dp_to_select_from:
                    sim_dp = sim_dp_stat.dp
                    selected_metric, metric_prob_dict = self.select_metric(sim_dp)
                    move_dir = self.select_dir(sim_dp)
                    selected_krnl, krnl_prob_dict = self.select_kernel(sim_dp, selected_metric, move_dir)
                    selected_block, block_prob_dict = self.select_block_without_sync(sim_dp, selected_krnl,
                                                                                     selected_metric)
                    if not selected_krnl.get_task_name() == best_sim_selected_krnl.get_task_name():
                        designs_to_consider.append((sim_dp_stat, energy))
                    elif not blocks_are_equal(selected_block, best_sim_selected_block):
                    #elif not selected_block.get_generic_instance_name() == best_sim_selected_block.get_generic_instance_name():
                        designs_to_consider.append((sim_dp_stat, energy))

                if len(designs_to_consider) == 0:
                    return sim_dp_to_select_from[0], False
                else:
                    return designs_to_consider[0], True # can be smarter here

    # use simulated annealing to pick the next design(s).
    # Use this link to understand simulated annealing (SA) http://www.cs.cmu.edu/afs/cs.cmu.edu/project/learn-43/lib/photoz/.g/we /glossary/anneal.html
    # cur_temp: current temperature for simulated annealing
    def SA_design_selection(self, sim_dp_stat_list, best_sim_dp_so_far_stats, cur_temp):
        # get the worse case cost for normalizing the cost when calculating the distance
        best_cost = min([sim_dp.get_system_complex_metric("cost") for sim_dp in (sim_dp_stat_list + [best_sim_dp_so_far_stats])])
        self.database.set_ideal_metric_value("cost", "glass", best_cost)

        # find if any of the new designs meet the budget
        new_designs_meeting_budget = []  # designs that are meeting the budget
        for sim_dp_stat in sim_dp_stat_list:
            if sim_dp_stat.fits_budget(1):
                new_designs_meeting_budget.append(sim_dp_stat)

        # find each design's simulated annealing Energy difference with the best design's energy
        # if any of the designs meet the budget or it's a cleanup iteration, include cost in distance calculation.
        # note that when we compare, we need to use the same dist_to_goal calculation, hence
        # ann_energy_best_dp_so_far needs to use the same calculation

        metric_to_target , metric_prob_dict = self.select_metric(best_sim_dp_so_far_stats.dp)
        include_cost_in_distance = best_sim_dp_so_far_stats.fits_budget(1) or (len(new_designs_meeting_budget) > 0) or self.is_cleanup_iter()
        if include_cost_in_distance:
            ann_energy_best_dp_so_far = best_sim_dp_so_far_stats.dist_to_goal(["cost", "latency", "power", "area"],
                                                                              "eliminate")
            ann_energy_best_dp_so_far_all_metrics = best_sim_dp_so_far_stats.dist_to_goal(["cost", "latency", "power", "area"],
                                                                              "eliminate")
        else:
            ann_energy_best_dp_so_far = best_sim_dp_so_far_stats.dist_to_goal([metric_to_target], "dampen")
            ann_energy_best_dp_so_far_all_metrics = best_sim_dp_so_far_stats.dist_to_goal(["power", "area", "latency"],
                                                                              "dampen")
        sim_dp_stat_ann_delta_energy_dict = {}
        sim_dp_stat_ann_delta_energy_dict_all_metrics = {}

        # deleteee the following debugging lines
        print("--------%%%%%%%%%%%---------------")
        print("--------%%%%%%%%%%%---------------")
        print("all the designs tried")
        print("first the best design from the previous iteration")
        print(" des" + " latency:" + str(best_sim_dp_so_far_stats.get_system_complex_metric("latency")))
        print(" des" + " power:" + str(
            best_sim_dp_so_far_stats.get_system_complex_metric("power")))
        print("energy :" + str(ann_energy_best_dp_so_far))


        sim_dp_to_look_at = [] # which designs to look at.
        # only look at the designs that meet the budget (if any), basically  prioritize these designs first
        if len(new_designs_meeting_budget) > 0:
            sim_dp_to_look_at = new_designs_meeting_budget
        else:
            sim_dp_to_look_at = sim_dp_stat_list

        for sim_dp_stat in sim_dp_to_look_at:
            if include_cost_in_distance:
                sim_dp_stat_ann_delta_energy_dict[sim_dp_stat] = sim_dp_stat.dist_to_goal(
                    ["cost", "latency", "power", "area"], "eliminate") - ann_energy_best_dp_so_far
                sim_dp_stat_ann_delta_energy_dict_all_metrics[sim_dp_stat] = sim_dp_stat.dist_to_goal(
                    ["latency", "power", "area"], "eliminate") - ann_energy_best_dp_so_far_all_metrics
            else:
                new_design_energy = sim_dp_stat.dist_to_goal([metric_to_target], "dampen")
                sim_dp_stat_ann_delta_energy_dict[sim_dp_stat] = new_design_energy - ann_energy_best_dp_so_far
                new_design_energy_all_metrics = sim_dp_stat.dist_to_goal(["power", "latency", "area"], "dampen")
                sim_dp_stat_ann_delta_energy_dict_all_metrics[sim_dp_stat] = new_design_energy_all_metrics - ann_energy_best_dp_so_far_all_metrics

                # deleteee this later
                print(" des" + " latency:" + str(
                    sim_dp_stat.get_system_complex_metric("latency")))
                print(" des" +" power:" + str(
                    sim_dp_stat.get_system_complex_metric("power")))
                print("delta energy :" + str(new_design_energy))
                print("*****")


        # changing the seed for random selection
        if config.DEBUG_FIX: random.seed(0)
        else: time.sleep(.00001), random.seed(datetime.now().microsecond)

        # find the best design
        sorted_sim_dp_stat_ann_delta_energy_dict_all_metrics = sorted(sim_dp_stat_ann_delta_energy_dict_all_metrics.items(), key=lambda x: x[1])
        if sorted_sim_dp_stat_ann_delta_energy_dict_all_metrics[0][1] < 0 and not True:
            design_improved = True
            best_neighbour_stat = sorted_sim_dp_stat_ann_delta_energy_dict_all_metrics[0][0]
            best_neighbour_delta_energy = sorted_sim_dp_stat_ann_delta_energy_dict_all_metrics[0][1]
        else:
            result, design_improved = self.find_best_design(sim_dp_stat_ann_delta_energy_dict, best_sim_dp_so_far_stats)
            best_neighbour_stat, best_neighbour_delta_energy = result

        # if any negative (desired move) value is detected or there is a design in the new batch
        #  that meet the budget, but the previous best design didn't, we have at least one improved solution
        found_an_improved_solution = any([(el<0) for el in sim_dp_stat_ann_delta_energy_dict.values()]) or \
                                     (len(new_designs_meeting_budget)>0 and not(best_sim_dp_so_far_stats).fits_budget(1)) or design_improved


        # for debugging. delete later
        if (len(new_designs_meeting_budget)>0 and not(best_sim_dp_so_far_stats).fits_budget(1)):
            print("what")

        if not found_an_improved_solution:
            # avoid not improving
            self.krnel_stagnation_ctr +=1
            self.des_stag_ctr += 1
            if self.krnel_stagnation_ctr > config.max_krnel_stagnation_ctr:
                self.krnel_rnk_to_consider = min(self.krnel_rnk_to_consider + 1, len(best_sim_dp_so_far_stats.get_kernels()) -1)
                self.krnel_stagnation_ctr = 0
                #self.recently_seen_design_ctr = 0
        elif best_neighbour_stat.dp.dp_rep.get_hardware_graph().get_SOC_design_code() in self.recently_cached_designs:
            # avoid circular exploration
            self.recently_seen_design_ctr += 1
            self.des_stag_ctr += 1
            if self.recently_seen_design_ctr > config.max_recently_seen_design_ctr:
                self.krnel_rnk_to_consider = min(self.krnel_rnk_to_consider + 1,
                                                 len(best_sim_dp_so_far_stats.get_kernels()) - 1)
                self.krnel_stagnation_ctr = 0
                #self.recently_seen_design_ctr = 0
        else:
            self.krnel_stagnation_ctr = 0
            self.krnel_rnk_to_consider = 0
            self.cleanup_ctr +=1
            self.des_stag_ctr = 0
            self.recently_seen_design_ctr = 0

        # initialize selected_sim_dp
        selected_sim_dp = best_sim_dp_so_far_stats.dp
        if found_an_improved_solution:
            selected_sim_dp = best_neighbour_stat.dp
        else:
            try:
                if math.e**(best_neighbour_delta_energy/max(cur_temp, .001)) < random.choice(range(0, 1)):
                    selected_sim_dp = best_neighbour_stat.dp
            except:
                selected_sim_dp = best_neighbour_stat.dp

        # cache the best design
        if len(self.recently_cached_designs) < config.recently_cached_designs_queue_size:
            self.recently_cached_designs.append(selected_sim_dp.dp_rep.get_hardware_graph().get_SOC_design_code())
        else:
            self.recently_cached_designs[self.total_itr_ctr%config.recently_cached_designs_queue_size] = selected_sim_dp.dp_rep.get_hardware_graph().get_SOC_design_code()

        return selected_sim_dp, found_an_improved_solution

    # ------------------------------
    # Functionality:
    #     select the next best design (from the sorted dp)
    # Variables
    #       ex_sim_dp_dict: example_simulate_design_point_list. List of designs to pick from.
    # ------------------------------
    def sel_next_dp(self, ex_sim_dp_dict, best_sim_dp_so_far, cur_temp):
        # convert to stats
        sim_dp_list = list(ex_sim_dp_dict.values())
        sim_dp_stat_list = [sim_dp.dp_stats for sim_dp in sim_dp_list]

        # find the ones that fit the expanded budget (note that budget radius shrinks)
        selected_sim_dp, found_improved_solution = self.SA_design_selection(sim_dp_stat_list, best_sim_dp_so_far.dp_stats, cur_temp)

        if not found_improved_solution:
            selected_sim_dp = self.so_far_best_sim_dp
            selected_ex_dp = self.so_far_best_ex_dp
        else:
            # extract the design
            for key, val in ex_sim_dp_dict.items():
                key.sanity_check()
                if val == selected_sim_dp:
                    selected_ex_dp = key
                    break

        # generate verification data
        if found_improved_solution and config.RUN_VERIFICATION_PER_IMPROVMENT:
            self.gen_verification_data(selected_sim_dp, selected_ex_dp)
        return selected_ex_dp, selected_sim_dp

    # ------------------------------
    # Functionality:
    #    simulate one design.
    # Variables
    #      ex_dp: example design point. Design point to simulate.
    #      database: hardware/software data base to simulated based off of.
    # ------------------------------
    def sim_one_design(self, ex_dp, database):
        if config.simulation_method == "power_knobs":
            sim_dp = self.dh.convert_to_sim_des_point(ex_dp)
            power_knob_sim_dp = self.dh.convert_to_sim_des_point(ex_dp)
            OSA = OSASimulator(sim_dp, database, power_knob_sim_dp)
        else:
            sim_dp = self.dh.convert_to_sim_des_point(ex_dp)
            # Simulator initialization
            OSA = OSASimulator(sim_dp, database)  # change

        # Does the actual simulation
        t = time.time()
        OSA.simulate()
        print("sim time" + str(time.time() -t))
        return sim_dp

    # ------------------------------
    # Functionality:
    #       Sampling from the task distribution. This is used for jitter  incorporation.
    # Variables:
    #       ex_dp: example design point.
    # ------------------------------
    def generate_sample(self, ex_dp, hw_sampling):
        new_ex_dp = copy.deepcopy(ex_dp)
        new_ex_dp.sample_hardware_graph(hw_sampling)
        return new_ex_dp

    # ------------------------------
    # Functionality:
    #       Evaluate the design. 1. simulate 2. collect (profile) data.
    # Variables:
    #       ex_dp: example design point.
    #       database: database containing hardware/software modeled characteristics.
    # ------------------------------
    def eval_design(self, ex_dp:ExDesignPoint, database):
        #start = time.time()
        # according to config singular runs
        if config.eval_mode == "singular":
            print("this mode is deprecated. just use statistical. singular is simply a special case")
            exit(0)
            return self.sim_one_design(ex_dp, database) # evaluation the design directly
        elif config.eval_mode == "statistical":
            # generate a population (geneate_sample), evaluate them and reduce to some statistical indicator
            ex_dp_pop_sample = [self.generate_sample(ex_dp, self.database.hw_sampling) for i in range(0, self.database.hw_sampling["population_size"])] # population sample
            ex_dp.get_tasks()[0].task_id_for_debugging_static += 1
            sim_dp_pop_sample = list(map(lambda ex_dp_: self.sim_one_design(ex_dp_, self.database), ex_dp_pop_sample)) # evaluate the population sample

            # collect profiling information
            sim_dp_statistical = SimDesignPointContainer(sim_dp_pop_sample, self.database, config.statistical_reduction_mode)
            #print("time is:" + str(time.time() -start))
            return sim_dp_statistical
        else:
            print("mode" + config.eval_mode + " is not defined for eval design")

    # ------------------------------
    # Functionality:
    #       generate verification (platform architect digestible) designs.
    # Variables:
    #       sim_dp: simulated design point.
    # ------------------------------
    def gen_verification_data(self, sim_dp_, ex_dp_):
        #from data_collection.FB_private.verification_utils.PA_generation.PA_generators import *
        import_ver = importlib.import_module("data_collection.FB_private.verification_utils.PA_generation.PA_generators")
        # iterate till you can make a directory
        while True:
            date_time = datetime.now().strftime('%m-%d_%H-%M_%S')
            result_folder = os.path.join(self.result_dir, "data_per_design",
                                         date_time+"_"+str(self.name_ctr))
            if not os.path.isdir(result_folder):
                os.makedirs(result_folder)
                collection_ctr = self.name_ctr # used to realize which results to compare
                break

        ex_with_PA = [] #
        pa_ver_obj = import_ver.PAVerGen()  # initialize a PA generator
        # make all the combinations
        knobs_list, knob_order = pa_ver_obj.gen_all_PA_knob_combos(import_ver.PA_knobs_to_explore)  # generate knob combinations
        #   for different PA designs. Since PA has extra knobs, we'd like sweep this knobs for verification purposes.
        knob_ctr = 0
        # Iterate though the knob combos and generate a (PA digestible) design accordingly
        for knobs in knobs_list:
            result_folder_for_knob = os.path.join(result_folder, "PA_knob_ctr_"+str(knob_ctr))
            for sim_dp in sim_dp_.get_design_point_list():
                sim_dp.reset_PA_knobs()
                sim_dp.update_ex_id(date_time+"_"+str(collection_ctr)+"_" + str(self.name_ctr))
                sim_dp.update_FARSI_ex_id(date_time+"_"+str(collection_ctr))
                sim_dp.update_PA_knob_ctr_id(str(knob_ctr))
                sim_dp.update_PA_knobs(knobs, knob_order, import_ver.auto_tuning_knobs)
                PA_result_folder = os.path.join(result_folder_for_knob, str(sim_dp.id))
                os.makedirs(PA_result_folder)

                #  dump data for bus-memory data with connection (used for bw calculation)
                sim_dp.dump_mem_bus_connection_bw(PA_result_folder) # write the results into a file
                # initialize and do some clean up
                #vis_hardware.vis_hardware(sim_dp, config.hw_graphing_mode, PA_result_folder)
                sim_dp.dp_stats.dump_stats(PA_result_folder)
                if config.VIS_SIM_PROG: vis_sim.plot_sim_data(sim_dp.dp_stats, sim_dp, PA_result_folder)
                block_names = [block.instance_name for block in sim_dp.hardware_graph.blocks]
                vis_hardware.vis_hardware(sim_dp, config.hw_graphing_mode, PA_result_folder)
                pa_obj = import_ver.PAGen(self.database, self.database.db_input.proj_name, sim_dp, PA_result_folder, config.sw_model)

                pa_obj.gen_all()  # generate the PA digestible design
                sim_dp.dump_props(PA_result_folder)  # write the results into a file
                #  pickle the results for (out of run) verifications.
                ex_dp_pickled_file = open(os.path.join(PA_result_folder, "ex_dp_pickled.txt"), "wb")
                dill.dump(ex_dp_, ex_dp_pickled_file)
                ex_dp_pickled_file.close()

                database_pickled_file = open(os.path.join(PA_result_folder, "database_pickled.txt"), "wb")
                dill.dump(self.database, database_pickled_file)
                database_pickled_file.close()

                sim_dp_pickled_file = open(os.path.join(PA_result_folder, "sim_dp_pickled.txt"), "wb")
                dill.dump(sim_dp, sim_dp_pickled_file)
                sim_dp_pickled_file.close()
                self.name_ctr += 1
            knob_ctr += 1

    # ------------------------------
    # Functionality:
    #       generate one neighbour and evaluate it.
    # Variables:
    #       des_tup: starting point design point tuple (design point, simulated design point)
    # ------------------------------
    def gen_neigh_and_eval(self, des_tup):
        self.SA_current_depth += 1
        # generate on neighbour
        ex_dp, move_to_try = self.gen_one_neigh(des_tup)

        # generate a code for the design (that specifies the topology, mapping and scheduling).
        # look into cache and see if this design has been seen before. If so, just use the
        # cached value, other wise just use the sim from cache
        design_unique_code = ex_dp.get_hardware_graph().get_SOC_design_code()  # cache index
        if design_unique_code not in self.cached_SOC_sim.keys():
            sim_dp = self.eval_design(ex_dp, self.database)  # evaluate the designs
            #if config.cache_seen_designs: # this seems to be slower than just simulation, because of deepcopy
            #    self.cached_SOC_sim[design_unique_code] = (ex_dp, sim_dp)
        else:
            ex_dp = self.cached_SOC_sim[design_unique_code][0]
            sim_dp = self.cached_SOC_sim[design_unique_code][1]


        # collect the moves for debugging/visualization
        if config.DEBUG_MOVE:
            if (self.total_itr_ctr % config.vis_reg_ctr_threshold) == 0:
                self.move_profile.append(move_to_try)  # for debugging
            self.last_move = move_to_try
            sim_dp.set_move_applied(move_to_try)

        # visualization and verification
        if config.VIS_GR_PER_GEN:
            vis_hardware.vis_hardware(sim_dp.get_dp_rep())
        if config.RUN_VERIFICATION_PER_GEN or \
                (config.RUN_VERIFICATION_PER_NEW_CONFIG and
                 not(sim_dp.dp.get_hardware_graph().get_SOC_design_code() in self.seen_SOC_design_codes)):
            self.gen_verification_data(sim_dp, ex_dp)
        self.seen_SOC_design_codes.append(sim_dp.dp.get_hardware_graph().get_SOC_design_code())

        return (ex_dp, sim_dp)

    # ------------------------------
    # Functionality:
    #       generate neighbours and evaluate them.
    #       neighbours are generated based on the depth and breath count determined in the config file.
    #       Depth means vertical, i.e., daisy chaining of the moves). Breadth means horizontal exploration.
    # Variables:
    #       des_tup: starting point design point tuple (design point, simulated design point)
    #       breadth: the breadth according to which to generate designs  (used for breadth wise search)
    #       depth: the depth according to which to generate designs (used for look ahead)
    # ------------------------------
    def gen_some_neighs_and_eval(self, des_tup, breath_length, depth_length, des_tup_list):
        # base case
        if depth_length == 0:
            return [des_tup]
        #des_tup_list = []
        # iterate on breath
        for i in range(0, breath_length):
            if not(breath_length == 1):
                self.SA_current_breadth += 1
                self.SA_current_depth = -1
                print("--------breadth--------")
            # iterate on depth (generate one neighbour and evaluate it)
            des_tup_new = self.gen_neigh_and_eval(des_tup)

            # collect the generate design in a list and run sanity check on it
            des_tup_list.append(des_tup_new)
            self.gen_some_neighs_and_eval(des_tup_new, 1, depth_length-1, des_tup_list)
            #des_tup_list.extend(self.gen_some_neighs_and_eval(des_tup_new, 1, depth_length-1))

            # visualization and sanity checks
            if config.VIS_MOVE_TRAIL:
                if (self.total_itr_ctr % config.vis_reg_ctr_threshold) == 0:
                    self.des_trail_list.append((copy.deepcopy(self.so_far_best_sim_dp), copy.deepcopy(des_tup_list[-1][1])))
                    self.last_des_trail = (copy.deepcopy(self.so_far_best_sim_dp), copy.deepcopy(des_tup_list[-1][1]))

            #self.vis_move_ctr += 1
            if config.DEBUG_SANITY: des_tup[0].sanity_check()
        #return des_tup_list

    # simple simulated annealing
    def simple_SA(self):
        # define the result dictionary
        this_itr_ex_sim_dp_dict:Dict[ExDesignPoint: SimDesignPoint] = {}
        this_itr_ex_sim_dp_dict[self.so_far_best_ex_dp] = self.so_far_best_sim_dp  # init the res dict

        # navigate the space using depth and breath parameters
        strt = time.time()
        print("------------------------ next itr ---------------------")
        self.SA_current_breadth = -1
        self.SA_current_depth = -1

        # generate some neighbouring design points and evaluate them
        des_tup_list =[]
        self.gen_some_neighs_and_eval((self.so_far_best_ex_dp, self.so_far_best_sim_dp), config.SA_breadth, config.SA_depth, des_tup_list)
        print("sim time + neighbour generation per design point " + str((time.time() - strt)/max(len(des_tup_list), 1)))

        # convert (outputed) list to dictionary of (ex:sim) specified above.
        # Also, run sanity check on the design, making sure everything is alright
        for ex_dp, sim_dp in des_tup_list:
            this_itr_ex_sim_dp_dict[ex_dp] = sim_dp
            if config.DEBUG_SANITY:
                ex_dp.sanity_check()
        return this_itr_ex_sim_dp_dict




    # ------------------------------
    # Functionality:
    #      Explore the initial design. Basically just simulated the initial design
    # Variables
    #       it uses the config parameters that are used to instantiate the object.
    # ------------------------------
    def explore_one_design(self):
        self.so_far_best_ex_dp = self.init_ex_dp
        self.so_far_best_sim_dp = self.eval_design(self.so_far_best_ex_dp, self.database)
        self.init_sim_dp = self.eval_design(self.so_far_best_ex_dp, self.database)

        # visualize/checkpoint/PA generation
        vis_hardware.vis_hardware(self.so_far_best_sim_dp.get_dp_rep())
        if config.RUN_VERIFICATION_PER_GEN or config.RUN_VERIFICATION_PER_IMPROVMENT or config.RUN_VERIFICATION_PER_NEW_CONFIG:
            self.gen_verification_data(self.so_far_best_sim_dp, self.so_far_best_ex_dp)

    # ------------------------------
    # Functionality:
    #      Explore the design space.
    # Variables
    #       it uses the config parameters that are used to instantiate the object.
    # ------------------------------
    def explore_ds(self):
        self.so_far_best_ex_dp = self.init_ex_dp
        self.so_far_best_sim_dp = self.eval_design(self.so_far_best_ex_dp, self.database)
        self.init_sim_dp = self.eval_design(self.so_far_best_ex_dp, self.database)

        # visualize/checkpoint/PA generation
        vis_hardware.vis_hardware(self.so_far_best_sim_dp.get_dp_rep())
        if config.RUN_VERIFICATION_PER_GEN or config.RUN_VERIFICATION_PER_IMPROVMENT or config.RUN_VERIFICATION_PER_NEW_CONFIG:
            self.gen_verification_data(self.so_far_best_sim_dp, self.so_far_best_ex_dp)

        des_per_iteration = [0]
        start = True
        cur_temp = config.annealing_max_temp

        while True:
            this_itr_ex_sim_dp_dict = self.simple_SA()   # run simple simulated annealing

            # collect profiling information about moves and designs generated
            if config.VIS_MOVE_TRAIL and (self.total_itr_ctr% config.vis_reg_ctr_threshold) == 0:
                plot.des_trail_plot(self.des_trail_list, self.move_profile, des_per_iteration)
                plot.move_profile_plot(self.move_profile)

            # select the next best design
            self.cur_best_ex_dp, self.cur_best_sim_dp = self.sel_next_dp(this_itr_ex_sim_dp_dict,
                                                                         self.so_far_best_sim_dp, cur_temp)

            print("-------:):):):):)----------")
            print("Best design's latency: " + str(self.cur_best_sim_dp.dp_stats.get_system_complex_metric("latency")))
            print("Best design's power: " + str(self.cur_best_sim_dp.dp_stats.get_system_complex_metric("power")))
            if  not self.cur_best_sim_dp.move_applied == None:
                self.cur_best_sim_dp.move_applied.print_info()

            if config.VIS_GR_PER_ITR and (self.total_itr_ctr% config.vis_reg_ctr_threshold) == 0:
                vis_hardware.vis_hardware(self.cur_best_sim_dp.get_dp_rep())

            # collect statistics about the design
            self.collect_stats(this_itr_ex_sim_dp_dict)

            # determine if the design has met the budget, if so, terminate
            should_terminate, reason_to_terminate = self.update_ctrs()
            if should_terminate:
                print("reason to terminate is:" + reason_to_terminate)
                vis_hardware.vis_hardware(self.cur_best_sim_dp.get_dp_rep())
                if not (self.last_des_trail == None):
                    if self.last_des_trail == None:
                        self.last_des_trail = (copy.deepcopy(self.so_far_best_sim_dp), copy.deepcopy(self.so_far_best_sim_dp))
                    else:
                        self.des_trail_list.append(self.last_des_trail)
                if not (self.last_move == None):
                    self.move_profile.append(self.last_move)
                plot.des_trail_plot(self.des_trail_list, self.move_profile, des_per_iteration)
                plot.move_profile_plot(self.move_profile)
                self.reason_to_terminate = reason_to_terminate
                return
            cur_temp -= config.annealing_temp_dec
            self.vis_move_trail_ctr += 1

    # ------------------------------
    # Functionality:
    #       generating plots for data analysis
    # -----------------------------
    def plot_data(self):
        iterations = [iter*config.num_neighs_to_try for iter in self.design_itr]
        if config.DATA_DELIVEYRY == "obfuscate":
            plot.scatter_plot(iterations, [area/self.area_explored[0] for area in self.area_explored], ("iteration", "area"), self.database)
            plot.scatter_plot(iterations, [power/self.power_explored[0] for power in self.power_explored], ("iteration", "power"), self.database)
            latency_explored_normalized = [el/self.latency_explored[0] for el in self.latency_explored]
            plot.scatter_plot(iterations, latency_explored_normalized, ("iteration", "latency"), self.database)
        else:
            plot.scatter_plot(iterations, [1000000*area/self.area_explored[0] for area in self.area_explored], ("iteration", "area"), self.database)
            plot.scatter_plot(iterations, [1000*power/self.power_explored[0] for power in self.power_explored], ("iteration", "power"), self.database)
            plot.scatter_plot(iterations, self.latency_explored/self.latency_explored[0], ("iteration", "latency"), self.database)

    # ------------------------------
    # Functionality:
    #       report the data collected in a humanly readable way.
    # Variables:
    #      explorations_start_time: to exploration start time used to determine the end-to-end exploration time.
    # -----------------------------
    def report(self, exploration_start_time):
        exploration_end_time = time.time()
        total_sim_time = exploration_end_time - exploration_start_time
        print("*********************************")
        print("------- Best Designs Metrics ----")
        print("*********************************")
        print("Best design's latency: " + str(self.so_far_best_sim_dp.dp_stats.get_system_complex_metric("latency")) + \
              ", ---- time budget:" + str(config.budgets_dict["glass"]["latency"]))
        print("Best design's thermal power: " + str(self.so_far_best_sim_dp.dp_stats.get_system_complex_metric("power"))+
              ", ---- thermal power budget:" + str(config.budgets_dict["glass"]["power"]))
        print("Best design's area: " + str(self.so_far_best_sim_dp.dp_stats.get_system_complex_metric("area")) + \
              ", ---- area budget:" + str(config.budgets_dict["glass"]["area"]))
        print("Best design's energy: " + str(self.so_far_best_sim_dp.dp_stats.get_system_complex_metric("energy")))
        print("*********************************")
        print("------- DSE performance --------")
        print("*********************************")
        print("Initial design's latency: " + str(self.init_sim_dp.dp_stats.get_system_complex_metric("latency")))
        print("Speed up: " + str(self.init_sim_dp.dp_stats.get_system_complex_metric("latency")/self.so_far_best_sim_dp.dp_stats.get_system_complex_metric("latency")))
        print("Number of design points examined:" + str(self.total_itr_ctr*config.num_neighs_to_try))
        print("Time spent per design point:" + str(total_sim_time/(self.total_itr_ctr*config.num_neighs_to_try)))
        print("The design meet the latency requirement: " + str(self.so_far_best_sim_dp.dp_stats.get_system_complex_metric("latency") < config.objective_budget))
        vis_hardware.vis_hardware(self.so_far_best_ex_dp)
        if config.VIS_FINAL_RES:
            vis_hardware.vis_hardware(self.so_far_best_ex_dp, config.hw_graphing_mode)

        # write the output
        home_dir = os.getcwd()
        FARSI_result_dir = config.FARSI_result_dir
        FARSI_result_directory_path = os.path.join(home_dir, 'data_collection/data/', FARSI_result_dir)
        output_file_verbose = os.path.join(FARSI_result_directory_path, config.FARSI_outputfile_prefix_verbose +".txt")
        output_file_minimal = os.path.join(FARSI_result_directory_path, config.FARSI_outputfile_prefix_minimal +".csv")

        # minimal output
        output_fh_minimal = open(output_file_minimal, "w")
        for metric in config.all_metrics:
            output_fh_minimal.write(metric+ ",")
        output_fh_minimal.write("\n")
        for metric in config.all_metrics:
            output_fh_minimal.write(str(self.so_far_best_sim_dp.dp_stats.get_system_complex_metric(metric))+ ",")
        output_fh_minimal.close()

        # verbose
        output_fh_verbose = open(output_file_verbose, "w")
        output_fh_verbose.write("iter_cnt" + ": ")
        for el in range(0, len(self.power_explored)):
            output_fh_verbose.write(str(el) +",")

        output_fh_verbose.write("\npower" + ": ")
        for el in self.power_explored:
            output_fh_verbose.write(str(el) +",")

        output_fh_verbose.write("\nlatency" + ": ")
        for el in self.latency_explored:
            output_fh_verbose.write(str(el) +",")
        output_fh_verbose.write("\narea" + ": ")
        for el in self.area_explored:
            output_fh_verbose.write(str(el) +",")

        output_fh_verbose.close()

    # ------------------------------
    # Functionality:
    #      collect the profiling information for all the design generated  by the explorer.  For data analysis.
    # Variables:
    #     ex_sim_dp_dict: example_design_simulated_design_dictionary. A dictionary containing the
    #     (example_design, simulated_design) tuple.
    # -----------------------------
    def collect_stats(self, ex_sim_dp_dict):
        for sim_dp in ex_sim_dp_dict.values():
            self.area_explored.append(sim_dp.dp_stats.get_system_complex_metric("area"))
            self.power_explored.append(sim_dp.dp_stats.get_system_complex_metric("power"))
            self.latency_explored.append(sim_dp.dp_stats.get_system_complex_metric("latency"))
            self.design_itr.append(self.total_itr_ctr)

    # ------------------------------
    # Functionality:
    #       calculate the budget coefficients. This is used for simulated annealing purposes.
    #       Concretely, first we use relax budgets to allow wider exploration, and then
    #       incrementally tighten the budget to direct the explorer more toward the goal.
    # ------------------------------
    def calc_budget_coeff(self):
        self.budget_coeff = int(((self.TOTAL_RUN_THRESHOLD - self.total_itr_ctr)/self.coeff_slice_size) + 1)

    # ------------------------------
    # Functionality:
    #       Update the counters to determine the exploration (navigation heuristic) control path to follow.
    # ------------------------------
    def update_ctrs(self):
        should_terminate = False
        reason_to_terminate = ""

        self.so_far_best_ex_dp = self.cur_best_ex_dp
        self.so_far_best_sim_dp = self.cur_best_sim_dp

        self.total_itr_ctr += 1
        stat_result = self.so_far_best_sim_dp.dp_stats

        if stat_result.fits_budget(1) :
            config.VIS_GR_PER_GEN = True  # visualize the graph per design point generation
            config.VIS_SIM_PER_GEN = True  # if true, we visualize the simulation progression
            self.fitted_budget_ctr +=1
        if (self.fitted_budget_ctr > config.fitted_budget_ctr_threshold):
            reason_to_terminate = "met the budget"
            should_terminate = True
        elif self.des_stag_ctr > self.DES_STAG_THRESHOLD:
            reason_to_terminate = "des_stag_ctr exceeded"
            should_terminate = True
        elif self.krnel_rnk_to_consider >= len(self.so_far_best_sim_dp.get_kernels()) - 2:
            reason_to_terminate = "all kernels already targeted without improvement"
            should_terminate = True
        elif self.total_itr_ctr > self.TOTAL_RUN_THRESHOLD:
            reason_to_terminate = "exploration (total itr_ctr) iteration threshold reached"
            should_terminate = True


        return should_terminate, reason_to_terminate
