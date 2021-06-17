#Copyright (c) Facebook, Inc. and its affiliates.
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.

import numpy as np
import copy
from design_utils.components.hardware import *
from design_utils.components.workload import *
from design_utils.components.mapping import *
from design_utils.components.scheduling import *
import operator
from collections import OrderedDict
from collections import defaultdict
from design_utils.common_design_utils import  *
import warnings
import queue
from collections import deque


# This is a proprietary Queue class. Not used at the moment, but possible will be
# used for Queue modeling
class MyQueue():
    def __init(self, max_size):
        self.max_size = max_size
        self.q_data = deque()

    def enqueue(self, data):
        if self.is_full():
            return False
        self.q_data.insert(0, data)
        return True

    def dequeue(self):
        if self.is_empty():
            return False
        self.q_data.pop()
        return True

    def peek(self):
        if self.is_empty():
            return False
        return self.q_data[-1]

    def size(self):
        return len(self.q_data)

    def is_empty(self):
        return (self.size() == 0)

    def is_full(self):
        return self.size() == self.max_size

    def __str__(self):
        return str(self.q_data)


# This class emulates the concept of waves, or rather a kernel that has multiple fronts
# where each front is only a burst of data. We are not using this class any more and
# at the moment, the concept of front means
class Wave:
    def __init__(self, block, channel_name, task_name, total_work, work_unit_size, id):
        self.task_name = task_name
        self.channel_name = channel_name
        self.block = block
        self.total_work = total_work
        self.id = id

        # set up the fronts
        total_work_to_distribute = total_work
        fq_size = math.ceil(total_work/work_unit_size)  # front queue size
        self.front_queue = MyQueue(fq_size)
        for i in range(0, fq_size):
            if total_work_to_distribute < work_unit_size:
               front_work = total_work_to_distribute
            else:
                front_work = work_unit_size
            self.front_queue(Front(block, channel_name, task_name, front_work, i))
            total_work_to_distribute += work_unit_size


# This class emulates a stream of data/instruction.
# for PEs, it's just a bunch of instructions sliced up to chunks
# for memories and buses, it's a stream of data that needs to move from one block (bus, memory) to another block (bus, memory).
# This class is not used for the moment
class Front:
    def __init__(self, host_block, src_block, dest_block, channel_name, main_task, family_task, total_work, work_unit):
        self.main_task = main_task  # the task that the stream is doing work for
        self.family_task = family_task  # the task that the stream reads/writes to. In case of instructions, family task is yourself
        self.channel_name = channel_name  # read/write or same
        self.host_block = host_block  # the block that
        # only one of the following  is populated depending on read (src) or write(dest)
        self.src_block = src_block
        self.dest_block = dest_block
        self.total_work = total_work
        self.work_unit = work_unit
        self.work_left = total_work
        self.state = "in_active"  # ["waiting", "to_be_waiting", "running"]

    def update_work_left(self, work_done):
        work_left_copy = self.work_left
        self.work_left -= work_done
        if self.work_left < -.001:
            print("this should not happen")
            exit(0)

    def update_state(self, state):
        assert(state in ["waiting", "to_be_waiting", "running", "done"])
        self.state = state


# class for getting relevant metrics (latency, energy, ...) for the task
class KernelStats:
    def __init__(self):
        self.latency = 0
        self.area= 0
        self.power = 0
        self.energy= 0
        self.throughput = None
        self.blck_pwr = {}  # (blck and power associated with it after execution)
        self.blck_energy = {}  # (blck and area associated with it after execution)
        self.blck_area = {}  # (block, area of block)
        self.phase_latency_dict = {}  # duration consumed per phase
        self.phase_energy_dict = {}  # energy consumed per phase
        self.phase_leakage_energy_dict = {}  # leakage energy consumed per phase
        self.phase_area_dict = {}    # area consumed per phase (note that we will be double counting the statically size
                                     # blocks

        self.phase_bytes_dict = {}
        self.design_cost = None
        self.phase_block_duration_bottleneck:Dict[int, (Block, float)] = {}  # dictionary containing phases and how different
                                                                             # blocks for different durations become the kernel
                                                                             # bottleneck
        self.block_phase_energy_dict = {}
        self.starting_time = 0  # when a kernel starts
        self.completion_time = 0  # when a kernel completes
        self.latency_till_earliest_child_start = 0

    # ------------------------------
    # Functionality
    #       get the block bottleneck for the task from the latency perspective
    # Variables:
    #       phase: the phase which the bottleneck qurried for. By default, the querry is for all the phases.
    # ------------------------------
    def get_block_latency_bottleneck(self, phase="all"):
        # sanity checks
        if not self.phase_block_duration_bottleneck:
            raise Exception('block_bottleneck is not identified yet.')
        elif isinstance(phase, int):
            raise Exception("bottleneck for specific phases are not supported yet.")
        else:
            block_duration_bottleneck: Dict[Block, float] = {}
            # iterate and accumulate the duration that blocks are the bottleneck for the kernel
            for phase, block_duration in self.phase_block_duration_bottleneck.items():
                block = block_duration[0]
                duration = block_duration[1]
                if block in block_duration_bottleneck:
                    block_duration_bottleneck[block] += duration
                else:
                    block_duration_bottleneck[block] = duration

            # sort from worst to best and return the worst (the bottleneck)
            sorted_block_duration_bottleneck = OrderedDict(
                sorted(block_duration_bottleneck.items(), key=operator.itemgetter(1)))

            return list(sorted_block_duration_bottleneck.keys())[-1]

    # get the block bottleneck from power perspective
    # phase: Simulation phases
    def get_block_power_bottleneck(self, phase):
        return (sorted(self.blck_pwr.items(), key=operator.itemgetter(1))[-1])[0]

    # get the block bottleneck from area perspective
    # phase: Simulation phases
    def get_block_area_bottleneck(self, phase):
        return (sorted(self.blck_area.items(), key=operator.itemgetter(1))[-1])[0]

    # get the block bottleneck from energy perspective
    # phase: Simulation phases
    def get_block_energy_bottleneck(self, phase):
        return (sorted(self.blck_energy.items(), key=operator.itemgetter(1))[-1])[0]

    # get the block bottleneck from cost perspective
    # phase: Simulation phases
    def get_block_cost_bottleneck(self, phase):
        return (sorted(self.blck_cost.items(), key=operator.itemgetter(1))[-1])[0]

    # ------------------------------
    # Functionality
    #       get the block bottleneck for the task
    # Variables:
    #       phase: the phase which the bottleneck querried for. By default, the query is for all the phases.
    # ------------------------------
    def get_block_latency_sorted(self, phase="all"):
        # sanity checks
        if not self.phase_block_duration_bottleneck:
            raise Exception('block_bottleneck is not identified yet.')
        elif isinstance(phase, int):
            raise Exception("bottleneck for specific phases are not supported yet.")
        else:
            block_duration_bottleneck: Dict[Block, float] = {}
            for phase, block_duration in self.phase_block_duration_bottleneck.items():
                block = block_duration[0]
                duration = block_duration[1]
                if block in block_duration_bottleneck:
                    block_duration_bottleneck[block] += duration
                else:
                    block_duration_bottleneck[block] = duration

            sorted_block_duration_bottleneck = OrderedDict(
                sorted(block_duration_bottleneck.items(), key=operator.itemgetter(1)))
            sorted_normalized = [(key, 100*value / sum(sorted_block_duration_bottleneck.values())) for key, value in sorted_block_duration_bottleneck.items()]

            # for latency, we need to zero out the rest since there is only one bottleneck through out each phase
            non_bottleneck_blocks =[]
            for block in self.blck_area.keys():
                if block.instance_name == sorted_normalized[0][0]:
                    continue
                non_bottleneck_blocks.append(block)
            for block in non_bottleneck_blocks:
                sorted_normalized.append((block, 0))
            return sorted_normalized

    # get the block bottlenecks (from power perspective) sorted
    # phase: Simulation phases
    def get_block_power_sorted(self, phase):
        sorted_list = sorted(self.blck_pwr.items(), key=operator.itemgetter(1))
        values =  [tuple_[1] for tuple_ in  sorted_list]
        sorted_normalized = [(key, 100*value/max(sum(values), .00000001))  for key, value in sorted_list]
        return sorted_normalized

    # get the block bottlenecks (from area perspective) sorted
    # phase: Simulation phases
    def get_block_area_sorted(self, phase):
        sorted_list = sorted(self.blck_area.items(), key=operator.itemgetter(1))
        values =  [tuple_[1] for tuple_ in  sorted_list]
        sorted_normalized = [(key, 100*value/sum(values))  for key, value in sorted_list]
        return sorted_normalized

    # get the block bottlenecks (from energy perspective) sorted
    # phase: Simulation phases
    def get_block_energy_sorted(self, phase):
        sorted_list = sorted(self.blck_energy.items(), key=operator.itemgetter(1))
        values =  [tuple_[1] for tuple_ in  sorted_list]
        sorted_normalized = [(key, 100*value/sum(values))  for key, value in sorted_list]
        return sorted_normalized

    # get the block bottlenecks (from cost perspective) sorted
    # phase: Simulation phases
    def get_block_cost_sorted(self, phase):
        sorted_list = sorted(self.blck_cost.items(), key=operator.itemgetter(1))
        values =  [tuple_[1] for tuple_ in  sorted_list]
        sorted_normalized = [(key, 100*value/max(sum(values),.00000001))  for key, value in sorted_list]
        return sorted_normalized

    # get the block bottlenecks (from the metric of interest perspective) sorted
    # phase: Simulation phases
    def get_block_sorted(self, metric="latency", phase="all"):
        if metric == "latency":
            return self.get_block_latency_sorted(phase)
        elif metric == "power":
            return self.get_block_power_sorted(phase)
        elif metric == "area":
            return self.get_block_area_sorted(phase)
        elif metric == "energy":
            return self.get_block_energy_sorted(phase)
        elif metric == "cost":
            return self.get_block_cost_sorted(phase)

    # get the block bottlenecks from the metric of interest perspective
    # phase: Simulation phases
    # metric: metric of interest to pick the bottleneck for
    def get_block_bottleneck(self, metric="latency", phase="all"):
        if metric == "latency":
            return self.get_block_latency_bottleneck(phase)
        elif metric == "power":
            return self.get_block_power_bottleneck(phase)
        elif metric == "area":
            return self.get_block_area_bottleneck(phase)
        elif metric == "energy":
            return self.get_block_energy_bottleneck(phase)
        elif metric == "cost":
            return self.get_block_cost_bottleneck(phase)

    # -----
    # setter
    # -----
    def set_stats(self):
        for metric in config.all_metrics:
            if metric == "latency":
                self.set_latency()
            if metric == "energy":
                self.set_energy()
            if metric == "power":
                self.set_power()
            if metric == "cost":
                return
                # already set before
            if metric == "area":
                # already set before
                return

    def set_latency(self):
        # already
        return

    # sets power for the entire kernel and also per blocks hosting the kernel
    def set_power(self):
        # get energy first
        sorted_listified_phase_latency_dict = sorted(self.phase_latency_dict.items(), key=operator.itemgetter(0))
        sorted_durations = [duration for phase, duration in sorted_listified_phase_latency_dict]
        sorted_phase_latency_dict = collections.OrderedDict(sorted_listified_phase_latency_dict)
        sorted_listified_phase_energy_dict = sorted(self.phase_energy_dict.items(), key=operator.itemgetter(0))
        sorted_phase_energy_dict = collections.OrderedDict(sorted_listified_phase_energy_dict)
        phase_bounds_lists = slice_phases_with_PWP(sorted_phase_latency_dict)

        # calculate power
        power_list = []  # list of power values collected based on the power collection freq
        for lower_bnd, upper_bnd in phase_bounds_lists:
            if sum(sorted_durations[lower_bnd:upper_bnd]) > 0:
                power_list.append(
                    sum(list(sorted_phase_energy_dict.values())[lower_bnd:upper_bnd]) / sum(sorted_durations[lower_bnd:upper_bnd]))
            else:
                power_list.append(0)
        self.power = max(power_list)

        # now calculate the above per block
        blck_pwr_list = defaultdict(list)
        for block, phase_energy_dict in self.block_phase_energy_dict.items():
            sorted_listified_phase_energy_dict = sorted(phase_energy_dict.items(), key=operator.itemgetter(0))
            sorted_phase_energy_dict = collections.OrderedDict(sorted_listified_phase_energy_dict)
            for lower_bnd, upper_bnd in phase_bounds_lists:
                if sum(sorted_durations[lower_bnd:upper_bnd]) > 0:
                    blck_pwr_list[block].append(
                        sum(list(sorted_phase_energy_dict.values())[lower_bnd:upper_bnd]) / sum(sorted_durations[lower_bnd:upper_bnd]))
                else:
                    blck_pwr_list[block].append(0)

        for blck in blck_pwr_list.keys() :
            self.blck_pwr[blck] = max(blck_pwr_list[blck])

    def set_area(self, area):
        self.area = area

    def set_block_area(self, area_dict):
        self.blck_area = area_dict
        self.set_area(sum(list(area_dict.values())))

    def set_cost(self, cost):
        self.cost = cost

    def set_block_cost(self, cost_dict):
        self.blck_cost = cost_dict
        self.set_cost(sum(list(cost_dict.values())))

    def set_energy(self):
        for block, phase_energy_dict in self.block_phase_energy_dict.items():
            self.blck_energy[block] = sum(list(phase_energy_dict.values()))
        self.energy = sum(self.phase_energy_dict.values())

    def set_stats_directly(self, metric_name, metric_value):
        if (metric_name == "latency"):
            self.latency = metric_value
        elif (metric_name == "energy"):
            self.energy = metric_value
        elif (metric_name == "power"):
            self.power = metric_value
        elif (metric_name == "area"):
            self.area = metric_value
        elif (metric_name == "cost"):
            self.cost = metric_value
        else:
            print("metric:" + metric_name + " is not supported in the stats")
            exit(0)

    # --------
    # getters
    # ------
    def get_latency(self):
        return self.latency

    def get_cost(self):
        return self.cost

    def get_power(self):
        return self.power

    def get_area(self):
        return self.area

    def get_energy(self):
        return self.energy

    def get_metric(self, metric):
        if metric == "latency":
            return self.get_latency()
        if metric == "energy":
            return self.get_energy()
        if metric == "power":
            return self.get_power()
        if metric == "area":
            return self.get_area()
        if metric == "cost":
            return self.get_cost()


# This class emulates the a task within the workload.
# The difference between kernel and task is that kernel is a simulation construct containing timing/energy/power information.
class Kernel:
    def __init__(self, task_to_blocks_map: TaskToBlocksMap): #, task_to_pe_block_schedule: TaskToPEBlockSchedule):
        # constructor argument vars, any changes to these need to initiate a reset
        self.__task_to_blocks_map = task_to_blocks_map  # mapping of the task to blocks
        self.kernel_total_work = self.__task_to_blocks_map.task.get_self_task_work()  # work (number of instructions for the task)

        # status vars
        self.cur_phase_bottleneck = ""  # which phase does the bottleneck occurs
        self.block_att_work_rate_dict = defaultdict(dict)  # for the kernel, block and their attainable work rate.
                                                           # attainable work rate is peak work rate (BW or IPC) of the block
                                                           # but attenuated as it is being shared among multiple kernels/fronts
        self.stats = KernelStats()
        self.workload_pe_total_work, self.workload_fraction, self.pe_s_work_left, self.progress, self.status = [None]*5
        self.block_dir_work_left = defaultdict(dict)  # block and the direction (write/read or loop) and how much work is left for the
                                                      # the kernel on this block
        self.phase_num = -1  # since the very first time, doesn't count
        self.block_phase_work_dict = defaultdict(dict)  # how much work per block and phase is done
        self.block_phase_energy_dict = defaultdict(dict)  # how much energy phase block and phase is consumed
        # how much leakage energy phase block and phase is consumed (PE and mem)
        self.block_phase_leakage_energy_dict = defaultdict(dict)
        self.block_phase_area_dict = defaultdict(dict)  # how much area phase block and phase is consumed
        self.SOC_type = ""
        self.SOC_id = ""
        self.set_SOC()
        self.task_name = self.__task_to_blocks_map.task.name

        # The variable shows what power_knob is being used!
        # 0 means baseline, and any other number is a DVFS/power_knob mode
        self.power_knob_id = 0
        self.starting_time = 0
        self.completion_time = 0

        # Shows what block is the bottleneck at every phase of the kernel execution
        self.kernel_phase_bottleneck_blocks_dict = defaultdict(dict)
        self.block_num_shared_blocks_dict = {}  # how many other kernels shared this block
        #self.work_unit_dict = self.__task_to_blocks_map.task.__task_to_family_task_work_unit  # determines for each burst how much work needs
                                                                                         # to be done.

    # This function is used for the power knobs simulator; After each run the stats
    #  gathered for each kernel is removed to avoid conflicting with past simulations
    def reset_sim_stats(self):
        # status vars
        self.cur_phase_bottleneck = ""
        self.block_att_work_rate_dict = defaultdict(dict)
        self.stats = KernelStats()
        self.workload_pe_total_work, self.workload_fraction, self.pe_s_work_left, self.progress, self.status = [None]*5
        self.phase_num = -1 # since the very first time, doesn't count
        self.block_normalized_work_rate = {}  # blocks and their work_rate (not normalized) including sharing
        self.block_phase_work_dict = defaultdict(dict)  # how much work per block and phase is done
        self.block_phase_energy_dict = defaultdict(dict)  # how much energy phase block and phase is consumed
        self.block_phase_leakage_energy_dict = defaultdict(dict)
        self.block_phase_area_dict = defaultdict(dict)  # how much area phase block and phase is consumed
        self.SOC_type = ""
        self.SOC_id = ""
        self.set_SOC()
        self.task_name = self.__task_to_blocks_map.task.name
        self.work_unit_dict = self.__task_to_blocks_map.task.__task_to_family_task_work_unit  # determines for each burst how much work needs
        self.power_knob_id = 0
        self.starting_time = 0
        self.completion_time = 0
        self.kernel_phase_bottleneck_blocks_dict = defaultdict(dict)
        self.block_num_shared_blocks_dict = {}

    # populate the statistics
    def set_stats(self):
        self.stats.set_block_area(self.calc_area_used_per_block())
        self.stats.set_block_cost(self.calc_cost_per_block())
        self.stats.set_stats()

    # set the SOC for the kernel
    def set_SOC(self):
        SOC_list = [(block.SOC_type, block.SOC_id) for block in self.__task_to_blocks_map.get_blocks()]
        if len(set(SOC_list)) > 1:
            raise Exception("kernel can not be resident in more than 1 SOC")
        SOC = SOC_list[0]
        self.SOC_type = SOC[0]
        self.SOC_id = SOC[1]

    # --------------
    # getters
    # --------------
    def get_task(self):
        return self.__task_to_blocks_map.task

    def get_task_name(self):
        return self.__task_to_blocks_map.task.name

    # work here is PE's work, so specified in terms of number of instructions
    def get_total_work(self):
        return self.kernel_total_work

    # get the list of blocks the kernel uses
    def get_block_list_names(self):
        return [block.instance_name for block in self.__task_to_blocks_map.get_blocks()]

    def get_blocks(self):
        return self.__task_to_blocks_map.get_blocks()

    # get the reference block for the kernel (reference block is what work rate is calculated
    # based off of.
    def get_ref_block(self):
        for block in self.__task_to_blocks_map.get_blocks():
            if block.type == "pe":
                return block

    # get kernel's memory blocks
    # dir: direction of interest (read/write)
    def get_kernel_s_mems(self, dir):
        mems = []
        for block in self.__task_to_blocks_map.get_blocks():
            if block.type == "mem":
                task_dir = block.get_task_dir_by_task_name(self.get_task())
                for task, dir_ in task_dir:
                    if dir_ == dir:
                        mems.append(block)
                        break

        if "souurce" in self.get_task_name() and dir == "read":
            if not len(mems) == 0:
                raise Exception(" convention is that no read memory for souurce task")
            else:
                return []
        elif "siink" in self.get_task_name() and dir == "write":
            if not len(mems) == 0:
                raise Exception(" convention is that no write memory for siink task")
            else:
                return []
        else:
            return mems

    def set_power_knob_id(self, pk_id):
        self.power_knob_id = pk_id
        return

    def get_power_knob_id(self):
        return self.power_knob_id

    # return the a dictionary containing the kernel bottleneck across different phases of execution
    def get_bottleneck_dict(self):
        return self.kernel_phase_bottleneck_blocks_dict

    # calculate the area
    def calc_area_used(self):
        total_area = 0
        for my_block in self.__task_to_blocks_map.get_blocks():
            if my_block.get_block_type_name() in ["ic", "pe"]:
                total_area += my_block.get_area()
            elif my_block.get_block_type_name() in ["mem"]:
                mem_work_ratio_read = self.__task_to_blocks_map.get_workRatio_by_block_name_and_dir(my_block.instance_name, "read")
                mem_work_ratio_write = self.__task_to_blocks_map.get_workRatio_by_block_name_and_dir(my_block.instance_name, "write")
                total_area += self.get_total_work()*mem_work_ratio_read/my_block.get_work_over_area()
                total_area += self.get_total_work()*mem_work_ratio_write/my_block.get_work_over_area()
        return total_area

    # calculate area per block
    def calc_area_used_per_block(self):
        area_dict = {}
        total_area = 0
        for my_block in self.__task_to_blocks_map.get_blocks():
            if my_block.get_block_type_name() in ["ic", "pe"]:
                area_dict[my_block] = my_block.get_area()
            elif my_block.get_block_type_name() in ["mem"]:
                mem_work_ratio_read = self.__task_to_blocks_map.get_workRatio_by_block_name_and_dir(my_block.instance_name, "read")
                mem_work_ratio_write = self.__task_to_blocks_map.get_workRatio_by_block_name_and_dir(my_block.instance_name, "write")
                area_dict[my_block] = self.get_total_work()*mem_work_ratio_read/my_block.get_work_over_area()
                area_dict[my_block] += self.get_total_work()*mem_work_ratio_write/my_block.get_work_over_area()
        return area_dict

    # calculate the cost per block
    def calc_cost_per_block(self):
        cost_dict = {}
        for my_block in self.__task_to_blocks_map.get_blocks():
            cost_dict[my_block] = 0
        return cost_dict

    # return the kernels that currently are on the block and channel of the block.
    def kernel_currently_uses_the_block_pipe_cluster(self, block, task, pipe_cluster):
        return pipe_cluster.is_task_present(task)

    # how many kernels using the block
    def get_block_num_shared_krnels(self, block, pipe_cluster, scheduled_kernels):
        krnl_using_block_channl = list(filter(lambda krnl: krnl.kernel_currently_uses_the_block_pipe_cluster(block, krnl.get_task(), pipe_cluster), scheduled_kernels))
        return len(krnl_using_block_channl)

    # creates a list of the blocks that are shared with other tasks in the current phase
    # Stores the numbers of tasks using the same resources to divide them among all later
    def get_blocks_num_shared_krnels(self, scheduled_kernels):
        #block_num_shared_blocks_dict = defaultdict(defaultdict(defaultdict))
        block_num_shared_blocks_dict = {}

        blocks = self.get_blocks()
        for block in blocks:
            for pipe_cluster in block.get_pipe_clusters_of_task(self.get_task()):
                dir = pipe_cluster.get_dir()
                cluster_UN = pipe_cluster.get_unique_name() # cluster unique name
                if block not in block_num_shared_blocks_dict.keys():
                    block_num_shared_blocks_dict[block] = {}
                if dir not in block_num_shared_blocks_dict[block].keys():
                    block_num_shared_blocks_dict[block][dir] = {}
                block_num_shared_blocks_dict[block][dir][cluster_UN] = self.get_block_num_shared_krnels(block, pipe_cluster, scheduled_kernels)
        return block_num_shared_blocks_dict

    # return the pipes that have kernels on them
    def filter_in_active_pipes(self, incoming_pipes, outcoming_pipe, scheduled_kernels):
        active_pipes_with_duplicates = []
        for in_pipe_ in incoming_pipes:
            for krnl in scheduled_kernels:
                if outcoming_pipe == None: # for memory
                    task_present_on_outcoming_pipe = True
                else:
                    task_present_on_outcoming_pipe = outcoming_pipe.is_task_present(krnl.__task_to_blocks_map.task)

                if  in_pipe_.is_task_present(krnl.__task_to_blocks_map.task) and task_present_on_outcoming_pipe:
                    active_pipes_with_duplicates.append(in_pipe_)

        active_pipes = list(set(active_pipes_with_duplicates))
        assert(len(active_pipes) <= len(incoming_pipes))
        return active_pipes

    # calculate what the work rate (BW or IPC) of each block is (for the kernel at hand)
    # This method assumes that each pipe gets an equal portion of the bandwidth, in other words,
    # equal pipe arbitration policy
    def alotted_work_rate_equal_pipe(self, pipe_cluster, scheduled_kernels):
        block = pipe_cluster.get_block_ref()
        dir = pipe_cluster.get_dir()
        # helper function
        def pipes_serial_work_ratio(pipe, scheduled_kernels, mode = "equal_per_kernel"):
            mode = "proportional_to_kernel"
            #mode = "equal_per_kernel"
            if mode == "equal_per_kernel":
                num_tasks_present = 0
                num_tasks_present = sum([int(pipe.is_task_present(krnel.__task_to_blocks_map.task)) for krnel in scheduled_kernels])
                task_present = int(pipe.is_task_present(self.__task_to_blocks_map.task))
                if num_tasks_present == 0:  # this scenario happens when we have schedulued only one task (specifically souurce or siink on a processor)
                    num_tasks_present = 1
                serial_work_rate = task_present / num_tasks_present
            elif mode == "proportional_to_kernel":
                all_kernels_work = sum([(pipe.get_task_work_unit(krnel.__task_to_blocks_map.task)) for krnel in scheduled_kernels])
                own_kernel_work = pipe.get_task_work_unit(self.__task_to_blocks_map.task)
                if all_kernels_work == 0:  # this scenario happens when we have schedulued only one task (sepcifically souurce or siink on a processor)
                    all_kernels_work = 1
                serial_work_rate = own_kernel_work/all_kernels_work
            return serial_work_rate

        if block.type == "pe": # pipes are not important for PEs
            allotted_work_rate = 1/self.block_num_shared_blocks_dict[block][dir][pipe_cluster.get_unique_name()]
        else:
            # get the pipes that kernel is running on and use bottleneck analysis
            # to find the work rate
            incoming_pipes = pipe_cluster.get_incoming_pipes()
            outgoing_pipe = pipe_cluster.get_outgoing_pipe()
            pipes_with_traffic = self.filter_in_active_pipes(incoming_pipes,outgoing_pipe, scheduled_kernels)
            allotted_work_rate = 0
            for pipe in pipes_with_traffic:
                pipe_serial_work_rate = pipes_serial_work_ratio(pipe, scheduled_kernels)
                allotted_work_rate += (1/len(pipes_with_traffic)) * pipe_serial_work_rate
        return allotted_work_rate

    # calculate the work rate (BW or IPC depending on the hardware block) of each kernel, while considering
    # sharing of the block across live kernels
    def calc_allotted_work_rate_relative_to_other_kernles(self, mode, pipe_cluster, scheduled_kernels):
        assert(mode in ["equal_rate_per_kernel", "equal_rate_per_pipe"])
        if mode == "equal_rate_per_kernel":
            return float(1./self.block_num_shared_blocks_dict[pipe_cluster.get_ref_block()][pipe_cluster.dir][pipe_cluster.get_unique_name()])
        elif mode == "equal_rate_per_pipe":
            return self.alotted_work_rate_equal_pipe(pipe_cluster, scheduled_kernels)

    def get_block_family_tasks_in_use(self, block):
        blocks_family_members = self.__task_to_blocks_map.get_block_family_members_allocated(block.instance_name)
        return blocks_family_members

    # get each blocks work-rate while considering sharing the block across the active kernels
    # Normalization is the process of normalizing the work_rate of each block with respect of the
    # reference work (work done by the PE). This then allows us to easily find the bottleneck
    # for the block as we have already normalized the data.
    def calc_all_block_normalized_work_rate(self, scheduled_kernels):
        self.block_num_shared_blocks_dict = self.get_blocks_num_shared_krnels(scheduled_kernels)
        #mode = "equal_rate_per_kernel"
        mode = "equal_rate_per_pipe"
        block_work_rate_norm_dict = defaultdict(defaultdict)

        # iterate through each block, channel.
        # (1) calculate the share of each kernel for the channel. (2) get their work ratio to normalize to
        # (3) use peak rate, share of each kernel and work ratio to generate final results
        for block in self.get_blocks():
            for pipe_cluster in block.get_pipe_clusters_of_task(self.get_task()):
                # calculate share of each kernel
                allocated_work_rate_relative_to_other_kernels = self.calc_allotted_work_rate_relative_to_other_kernles(mode, pipe_cluster, scheduled_kernels)
                if allocated_work_rate_relative_to_other_kernels == 0:  # when the channel in the block is not being used
                    continue

                # get work ratio (so you can normalize to it)
                dir = pipe_cluster.get_dir()
                work_ratio = self.__task_to_blocks_map.get_workRatio_by_block_name_and_family_member_names_and_channel_eliminating_fake(
                    block.instance_name, self.get_block_family_tasks_in_use(block), dir)

                if "souurce" in self.get_task_name() or "siink" in self.get_task_name():
                    block_work_rate_norm_dict[block][pipe_cluster] = 1
                else:
                    if work_ratio == 0:
                        print("this should be looked at")
                        work_ratio = self.__task_to_blocks_map.get_workRatio_by_block_name_and_family_member_names_and_channel_eliminating_fake(
                            block.instance_name, self.get_block_family_tasks_in_use(block), dir)


                    work_rate =  float(block.get_peak_work_rate(self.get_power_knob_id()))*allocated_work_rate_relative_to_other_kernels/work_ratio
                    block_work_rate_norm_dict[block][pipe_cluster] = float(block.get_peak_work_rate(self.get_power_knob_id()))*allocated_work_rate_relative_to_other_kernels/work_ratio
                    if block_work_rate_norm_dict[block][pipe_cluster] == 0:
                        print("what")

        return block_work_rate_norm_dict

    # simply go through all the block work rate and pick the smallest
    def calc_block_s_bottleneck(self, block_work_rate_norm_dict):
        # only if work unit is left
        block_bottleneck = None
        bottleneck_work_rate = 0
        # iterate through all the blocks/channels and ge the minimum work rate. Since
        # the data is normalized, minimum is the bottleneck
        for block, pipe_cluster_work_rate in block_work_rate_norm_dict.items():
            for pipe_cluster, work_rate in pipe_cluster_work_rate.items():
                if not block_bottleneck:
                    block_bottleneck = block
                    bottleneck_work_rate = work_rate
                else:
                    if work_rate < bottleneck_work_rate:
                        bottleneck_work_rate = work_rate
                        block_bottleneck = block

        return block_bottleneck, bottleneck_work_rate

    # calculate the unnormalized work rate.
    # Normalization is the process of normalizing the work_rate of each block with respect of the
    # reference work (work done by the PE). This then allows us to easily find the bottleneck
    # for the block as we have already normalized the data. Unnormalization is the reverse
    # process
    def calc_unnormalize_work_rate(self, block_work_rate_norm_dict, bottleneck_work_rate):
        block_att_work_rate_dict = defaultdict(dict)
        for block, pipe_cluster_work_rate in block_work_rate_norm_dict.items():
            for pipe_cluster, work_rate in pipe_cluster_work_rate.items():
                dir = pipe_cluster.get_dir()
                work_ratio = self.__task_to_blocks_map.get_workRatio_by_block_name_and_family_member_names_and_channel_eliminating_fake(
                    block.instance_name, self.get_block_family_tasks_in_use(block), dir)

                if "souurce" in self.get_task_name() or "siink" in self.get_task_name():
                    work_ratio = 1
                block_att_work_rate_dict[block][pipe_cluster] = bottleneck_work_rate*work_ratio

        return block_att_work_rate_dict



    def read_latency_per_request(self, mem, pe):
        pass

    def write_latency_per_request(self, mem, pe):
        pass

    # consolidate read/write channels in order to emulate DMA read/write serialization.
    # Consolidation is essentially the process of ensuring that we can closely emulate the fact
    # that DMA might serialize read/writes.
    def consolidate_channels(self, block_normalized_work_rate):
        block_normalized_work_rate_consolidated = defaultdict(dict)
        assert config.DMA_mode in ["serialized_read_write", "parallelized_read_write"]
        if config.DMA_mode == "serialized_read_write":
            for block, pipe_cluster_work_rate in block_normalized_work_rate.items():
                for pipe_cluster, work_rate in pipe_cluster_work_rate.items():
                    block_normalized_work_rate_consolidated[block][pipe_cluster] = 1/sum([1/norm_wr for  norm_wr in pipe_cluster_work_rate.values()])
        elif config.DMA_mode == "parallelized_read_write":
            block_normalized_work_rate_consolidated = block_normalized_work_rate

        return block_normalized_work_rate_consolidated

    # calculate the attainable work rate of the block
    def update_block_att_work_rate(self, scheduled_kernels):
        # get block work rate. In this step we calculate the normalized work rate.
        # normalized work rate is actual work rate normalized to the work rate of
        # the ref block (which is usally PE) work rate. Normalizing allows us
        # to identify the bottleneck easily since every one has the same unit and reference
        self.block_normalized_work_rate_unconsolidated = self.calc_all_block_normalized_work_rate(scheduled_kernels)

        # consolidate read/write channels since DMA serializes read/writes
        self.block_normalized_work_rate = self.consolidate_channels(self.block_normalized_work_rate_unconsolidated)

        # identify the block bottleneck
        self.cur_phase_bottleneck, bottleneck_work_rate = self.calc_block_s_bottleneck(self.block_normalized_work_rate)
        self.kernel_phase_bottleneck_blocks_dict[self.phase_num] = self.cur_phase_bottleneck
        ref_block = self.get_ref_block()

        # unnormalized the results (unnormalizing means that actually provide the work rate as opposed
        # to normalizing it to the ref block (which is usally PE) work rate
        self.block_att_work_rate_dict = self.calc_unnormalize_work_rate(self.block_normalized_work_rate, bottleneck_work_rate)

    # calculate the completion time for the kernel
    def calc_kernel_completion_time(self):
        return self.pe_s_work_left/self.block_att_work_rate_dict[self.get_ref_block()][self.get_ref_block().get_pipe_clusters()[0]]

    # launch the kernel
    # Variables:
    #       cur_time: current time (s)
    def launch(self, cur_time):
        self.pe_s_work_left = self.kernel_total_work
        # keeping track of how much work left for every block
        for block_dir_work_ratio in self.__task_to_blocks_map.block_dir_workRatio_dict.keys():
            if block_dir_work_ratio not in self.block_dir_work_left.keys():
                self.block_dir_work_left[block_dir_work_ratio] = {}
            for task, ratio in self.__task_to_blocks_map.block_dir_workRatio_dict[block_dir_work_ratio].items():
                self.block_dir_work_left[block_dir_work_ratio][task] = (self.pe_s_work_left*ratio)

        self.status = "in_progress"
        self.starting_time = cur_time

    # has kernel completed already
    def has_completed(self):
        return self.status == "completed"

    # has kernel started already
    def has_started(self):
        return self.status == "in_progress"

    # update the status of the kernel to specify whether it's done or not
    def update_status(self, time_step_size):
        if self.kernel_total_work == 0:  self.progress = 1  # for dummy tasks (with the suffix of souurce and siink)
        else: self.progress = 1 - float(self.pe_s_work_left/self.kernel_total_work)

        self.stats.latency += time_step_size

        if self.progress >= .99:
            self.status = "completed"
            self.completion_time = self.stats.latency + self.starting_time
        elif self.progress == 0:
            self.status = "not_scheduled"
        else:
            self.status = "in_progress"

        self.update_stats(time_step_size)

    # given the time (time_step_size) of the tick, calculate how much work has the
    # kernel accomplished. Note that work concept varies depending on the hardware block, i.e.,
    # work = bytes for memory/uses and its instructions for PEs
    def calc_work_consumed(self, time_step_size):
        # iterate through each blocks attainable work rate and calculate
        # how much work it can do for the kernel of interest
        for block, pipe_clusters_work_rate in self.block_att_work_rate_dict.items():
            for pipe_cluster, work_rate in pipe_clusters_work_rate.items():
                if self.phase_num in self.block_phase_work_dict[block].keys():
                    self.block_phase_work_dict[block][self.phase_num] += work_rate* time_step_size
                else:
                    self.block_phase_work_dict[block][self.phase_num] = work_rate* time_step_size

    # Calculates the leakage power of the phase for PE and IC
    # memory leakage power should be accumulated for the whole execution time
    # since we cannot turn off the memory but the rest can be in cut-off (C7) mode
    def calc_leakage_energy_consumed(self, time_step_size):
        for block, work in self.block_phase_work_dict.items():
            # taking care of dummy corner case
            if "souurce" in self.get_task_name() or "siink" in self.get_task_name():
                self.block_phase_leakage_energy_dict[block][self.phase_num] = 0
            else:
                if block.get_block_type_name() == "mem":
                    self.block_phase_leakage_energy_dict[block][self.phase_num] = 0
                else:
                    # changed to get by Hadi
                    self.block_phase_leakage_energy_dict[block][self.phase_num] = \
                        block.get_leakage_power(self.get_power_knob_id()) * time_step_size

    # calculate energy consumed
    def calc_energy_consumed(self):
        for block, work in self.block_phase_work_dict.items():
            # Dynamic energy consumption
            if "souurce" in self.get_task_name() or "siink" in self.get_task_name():  # taking care of dummy corner case
                 self.block_phase_energy_dict[block][self.phase_num] = 0
            else:
                # changed to get by Hadi
                this_phase_energy = self.block_phase_work_dict[block][self.phase_num] / block.get_work_over_energy(self.get_power_knob_id())
                if this_phase_energy < 0:
                    print("energy can't be a negative value")
                    block.get_work_over_energy(self.get_power_knob_id())
                    exit(0)
                self.block_phase_energy_dict[block][self.phase_num] = this_phase_energy

                pass

    # for read, we release memory (the entire input worth of data) once the kernel is done
    # for write, we assign memory (the entire output worth of data) once the kernel starts
    # if a sibling of a task depends on the same data (that resides in the memory), we can't let
    # go of that till the sibling is done
    # coeff is gonna determine whether to retract or expand the memory
    # Todo: include the case where there are multiple siblings
    def update_mem_size(self, coef):
        if "souurce" in self.get_task_name() and coef == -1: return
        elif "siink" in self.get_task_name() and coef == 1: return

        dir_ = "write"
        mems = self.get_kernel_s_mems(dir=dir_)
        if "souurce" in self.get_task_name():
            if dir_ == "write":
                #memory_total_work = config.souurce_memory_work
                for mem in mems:
                    # mem_work_ratio = self.__task_to_blocks_map.get_workRatio_by_block_name_and_dir(mem.instance_name, dir_)
                    tasks_name = self.__task_to_blocks_map.get_tasks_of_block_dir(
                        mem.instance_name, dir_)
                    # Sends the memory size that needs to be occupied (positive for write and negative for read)
                    # then updates the memory mapping in the memory block to know what is the in use capacity
                    # changed to get by Hadi
                    for tsk in tasks_name:
                        memory_total_work = config.souurce_memory_work[tsk]
                        mem.update_area(coef*memory_total_work/mem.get_work_over_area(self.get_power_knob_id()), self.get_task_name())
            else: memory_total_work = 0
        elif "siink" in self.get_task_name():
            memory_total_work = 0
        else:
            pe_s_total_work = self.kernel_total_work
            for mem in mems:
                #mem_work_ratio = self.__task_to_blocks_map.get_workRatio_by_block_name_and_dir(mem.instance_name, dir_)
                mem_work_ratio = self.__task_to_blocks_map.get_workRatio_by_block_name_and_dir_eliminating_fake(mem.instance_name, dir_)
                memory_total_work = pe_s_total_work * mem_work_ratio
                # changed to get by Hadi
                mem.update_area(coef*memory_total_work/mem.get_work_over_area(self.get_power_knob_id()), self.get_task_name())

                mem_work_ratio = self.__task_to_blocks_map.get_workRatio_by_block_name_and_dir(mem.instance_name, dir_)

    # update pe allocation -> allocate a part of pe quantum for current task
    def update_pe_size(self):
        pe = self.get_ref_block()
        # the convention is ot provide 1/fixe_area for these blocks
        # DSPs and Processors are among statically sized blocks while accelerators are not among the list
        if pe.subtype in config.statically_sized_blocks: work = 1
        else: work = self.kernel_total_work
        pe.update_area(work/pe.get_work_over_area(self.get_power_knob_id()), self.get_task_name())

    # TODO: need to include ic as well
    def update_ic_size(self):
        return 0

    # calculate how much of the work is left for the kernel.
    # Note that work concept varies depending on the hardware block, i.e.,
    # work = bytes for memory/uses and its instructions for PEs
    def calc_work_left(self):
        if not config.transaction_base_simulation:
            self.pe_s_work_left -= self.block_phase_work_dict[self.get_ref_block()][self.phase_num]
        else:
            # use the following for phase based simulation
            self.pe_s_work_left -= self.block_phase_work_dict[self.get_ref_block()][self.phase_num]
            for block_dir_work_ratio in self.__task_to_blocks_map.block_dir_workRatio_dict.keys():
                if block_dir_work_ratio not in self.block_dir_work_left.keys():
                    self.block_dir_work_left[block_dir_work_ratio] = {}
                for task, ratio in self.__task_to_blocks_map.block_dir_workRatio_dict[block_dir_work_ratio].items():
                    self.block_dir_work_left[block_dir_work_ratio][task] = (self.pe_s_work_left*ratio)

    # accumulate how much area has been used for a phase of execution
    def aggregate_area_of_phase(self):
        total_area_consumed = 0
        for block, phase_area_dict in self.block_phase_area_dict.items():
            total_area_consumed += sum(list(phase_area_dict.values()))
        return total_area_consumed


    # aggregate the energy consumed for all the blocks for a specific phase
    def aggregate_area_of_for_every_phase(self):
        aggregate_phase_area = {}
        for block, phase_area_dict in self.block_phase_area_dict.items():
            for phase, area in phase_area_dict.items():
                this_phase_area = phase_area_dict[phase]
                if phase not in aggregate_phase_area.keys():
                    aggregate_phase_area [phase] = 0
                aggregate_phase_area[phase] += this_phase_area
        return aggregate_phase_area


    # aggregate the energy consumed for all the blocks for a specific phase
    def aggregate_energy_of_for_every_phase(self):
        aggregate_phase_energy = {}
        for block, phase_energy_dict in self.block_phase_energy_dict.items():
            for phase, energy in phase_energy_dict.items():
                this_phase_energy = phase_energy_dict[phase]
                if phase not in aggregate_phase_energy.keys():
                    aggregate_phase_energy[phase] = 0
                aggregate_phase_energy[phase] += this_phase_energy
        return aggregate_phase_energy


    # aggregate the energy consumed for all the blocks for a specific phase
    def aggregate_energy_of_phase(self):
        total_energy_consumed = 0
        for block, phase_energy_dict in self.block_phase_energy_dict.items():
            this_phase_energy = phase_energy_dict[self.phase_num]
            total_energy_consumed += this_phase_energy
        return total_energy_consumed

    def aggregate_leakage_energy_of_phase(self):
        total_leakage_energy_consumed = 0
        for block, phase_leakage_energy_dict in self.block_phase_leakage_energy_dict.items():
            this_phase_leakage_energy = phase_leakage_energy_dict[self.phase_num]
            total_leakage_energy_consumed += this_phase_leakage_energy
        return total_leakage_energy_consumed

    # Checks if there was memory bounded phases in the kernel execution
    def is_kernel_memory_bounded(self):
        blocks = self.kernel_phase_bottleneck_blocks_dict.values()
        for block in blocks:
            if block.get_block_type_name() == "mem":
                return True
        return False

    # Checks if there was any compute intensive phases in the kernel execution
    def is_kernel_processing_bounded(self):
        blocks = self.kernel_phase_bottleneck_blocks_dict.values()
        for block in blocks:
            if block.get_block_type_name() == "pe":
                return True
        return False

    # Checks if there was any IC intensive phases in the kernel execution
    def is_kernel_interconnects_bounded(self):
        blocks = self.kernel_phase_bottleneck_blocks_dict.values()
        for block in blocks:
            if block.get_block_type_name() == "ic":
                return True
        return False

    # update the progress of kernel (i.e., how much work is left)
    def update_progress(self, time_step_size):
        # calculate the metric consumed for each phase
        self.calc_work_consumed(time_step_size)
        self.calc_work_left()
        self.calc_energy_consumed()
        #self.calc_leakage_energy_consumed(time_step_size)

        # Calculates the leakage power of the phase for PE and IC
        # memory leakage power should be accumulated for the whole execution time
        # since we cannot turn off the memory but the rest can be in cut-off (C7) mode
        if config.simulation_method == "power_knobs":
            self.calc_leakage_energy_consumed(time_step_size)

    # update the stats for a kernel (e.g., energy, start time, ...)
    def update_stats(self, time_step_size):
        self.stats.phase_block_duration_bottleneck[self.phase_num] = (self.cur_phase_bottleneck, time_step_size)
        self.stats.phase_energy_dict[self.phase_num] = self.aggregate_energy_of_phase()
        self.stats.phase_latency_dict[self.phase_num] = time_step_size
        self.stats.block_phase_energy_dict = self.block_phase_energy_dict
        if config.simulation_method == "power_knobs":
            # aggregate the energy consumed among all the blocks corresponding to the task in current phase
            self.stats.phase_leakage_energy_dict[self.phase_num] = \
                self.aggregate_leakage_energy_of_phase()

            # Update the starting and completion time of the kernel -> used for power knob simulator
            self.stats.starting_time = self.starting_time
            self.stats.completion_time = self.completion_time

    # step the kernel progress forward
    # Variables:
    #       phase_num: phase number
    def step(self, time_step_size, phase_num):
        self.phase_num = phase_num
        # update the amount of work remaining per block
        self.update_progress(time_step_size)


    def get_schedule(self):
        return self.__schedule