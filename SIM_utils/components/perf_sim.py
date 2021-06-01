#Copyright (c) Facebook, Inc. and its affiliates.
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.

from design_utils.design import  *
from functools import reduce


# This class is the performance simulator of FARSI
class PerformanceSimulator:
    def __init__(self, sim_design):
        self.design = sim_design  # design to simulate
        self.scheduled_kernels = []   # kernels already scheduled
        self.completed_kernels = []   # kernels already completed

        # List of all the kernels that are not scheduled yet (to be launched)
        self.yet_to_schedule_kernels = self.design.get_kernels()[:]  # kernels to be scheduled
        self.old_clock_time = self.clock_time = 0
        self.program_status = "idle"  # specifying the status of the program at the current tick
        self.phase_num = -1


    def reset_perf_sim(self):
        self.scheduled_kernels = []
        self.completed_kernels = []
        # List of all the kernels that are not scheduled yet (to be launched)
        self.yet_to_schedule_kernels = self.design.get_kernels()[:]
        self.old_clock_time = self.clock_time = 0
        self.program_status = "idle"  # specifying the status of the program at the current tick
        self.phase_num = -1

    # ------------------------------
    # Functionality:
    #   tick the simulator clock_time forward
    # ------------------------------
    def tick(self, clock_time):
        self.clock_time = clock_time

    # ------------------------------
    # Functionality:
    #   find the next kernel to be scheduled time
    # ------------------------------
    def next_kernel_to_be_scheduled_time(self):
        timely_sorted_kernels = sorted(self.yet_to_schedule_kernels, key=lambda kernel: kernel.get_schedule().starting_time)
        return timely_sorted_kernels[0].get_schedule().starting_time

    # ------------------------------
    # Functionality:
    #   convert the task to kernel
    # ------------------------------
    def get_kernel_from_task(self, task):
        for kernel in self.design.get_kernels()[:]:
            if kernel.get_task() == task:
                return kernel
        raise Exception("kernel associated with task with name" + task.name + " is not found")

    # ------------------------------
    # Functionality:
    #   find the completion time of kernel that will be done the fastest
    # ------------------------------
    def next_kernel_to_be_completed_time(self):
        comp_time_list = []  # contains completion time of the running kernels
        for kernel in self.scheduled_kernels:
            comp_time_list.append(kernel.calc_kernel_completion_time())
        if comp_time_list:
            return min(comp_time_list) + self.clock_time
        else:
            return self.clock_time

    # ------------------------------
    # Functionality:
    #   all the dependencies of a kernel are done or no?
    # ------------------------------
    def kernel_parents_all_done(self, kernel):
        kernel_s_task = kernel.get_task()
        parents_s_task = self.design.get_hardware_graph().get_task_graph().get_task_s_parents(kernel_s_task)
        completed_tasks = [kernel.get_task() for kernel in self.completed_kernels]
        for task in parents_s_task:
            if task not in completed_tasks:
                return False
        return True

    # ------------------------------
    # Functionality:
    #   Finds the kernels that are free to be scheduled (their parents are completed)
    # ------------------------------
    def schedule_kernels(self):
        if config.scheduling_policy == "FRFS":
            kernels_to_schedule = [kernel_ for kernel_ in self.yet_to_schedule_kernels
                                   if self.kernel_parents_all_done(kernel_)]
        elif config.scheduling_policy == "time_based":
            kernels_to_schedule = [kernel_ for kernel_ in self.yet_to_schedule_kernels
                                   if self.clock_time >= kernel_.get_schedule().starting_time]
        else:
            raise Exception("scheduling policy not supported")

        for kernel in kernels_to_schedule:
            self.scheduled_kernels.append(kernel)
            self.yet_to_schedule_kernels.remove(kernel)
            # initialize #insts, tick, and kernel progress status
            kernel.launch(self.clock_time)
            # update memory size -> allocate memory regions on different mem blocks
            kernel.update_mem_size(1)
            # update pe allocation -> allocate a part of pe quantum for current task
            # (Hadi Note: allocation looks arbitrary and without any meaning though - just to know that something
            # is allocated or it is floating)
            kernel.update_pe_size()
            # empty function!
            kernel.update_ic_size()


    # ------------------------------
    # Functionality:
    #   update the status of each kernel, this means update
    #   how much work is left for each kernel (that is already schedulued)
    # ------------------------------
    def update_scheduled_kernel_list(self):
        scheduled_kernels = self.scheduled_kernels[:]
        for kernel in scheduled_kernels:
            if kernel.status == "completed":
                self.scheduled_kernels.remove(kernel)
                self.completed_kernels.append(kernel)
                kernel.set_stats()

                # iterate though parents and check if for each parent, all the children are completed.
                # if so, retract the memory
                all_parent_kernels = [self.get_kernel_from_task(parent_task) for parent_task in
                                      kernel.get_task().get_parents()]
                for parent_kernel in all_parent_kernels:
                    all_children_kernels = [self.get_kernel_from_task(child_task) for child_task in
                                           parent_kernel.get_task().get_children()]
                    if all([child_kernel in self.completed_kernels for child_kernel in all_children_kernels]):
                        parent_kernel.update_mem_size(-1)

    # ------------------------------
    # Functionality:
    #   iterate through all kernels and step them
    # ------------------------------
    def step_kernels(self):
        # by stepping the kernels, we calculate how much work each kernel has done and how much of their
        # work is left for them
        _ = [kernel_.step(self.time_step_size, self.phase_num) for kernel_ in self.scheduled_kernels]

        # update kernel's status, sets the progress
        _ = [kernel_.update_status(self.time_step_size) for kernel_ in self.scheduled_kernels]

    # ------------------------------
    # Functionality:
    #   update the status of the program, i.e., whether it's done or still in progress
    # ------------------------------
    def update_program_status(self):
        if len(self.scheduled_kernels) == 0 and len(self.yet_to_schedule_kernels) == 0:
            self.program_status = "done"
        elif len(self.scheduled_kernels) == 0:
            self.program_status = "idle"  # nothing scheduled yet
        elif len(self.yet_to_schedule_kernels) == 0:
            self.program_status = "all_kernels_scheduled"
        else:
            self.program_status = "in_progress"

    # ------------------------------
    # Functionality:
    #   find the next tick time
    # ------------------------------
    def calc_new_tick_position(self):
        if config.scheduling_policy == "FRFS":
            new_clock = self.next_kernel_to_be_completed_time()
        elif self.program_status == "in_progress":
            if self.program_status == "in_progress":
                if config.scheudling_policy == "time_based":
                    new_clock = min(self.next_kernel_to_be_scheduled_time(), self.next_kernel_to_be_completed_time())
            elif self.program_status == "all_kernels_scheduled":
                new_clock = self.next_kernel_to_be_completed_time()
            elif self.program_status == "idle":
                new_clock = self.next_kernel_to_be_scheduled_time()
            if self.program_status == "done":
                new_clock = self.clock_time
        else:
            raise Exception("scheduling policy:" + config.scheduling_policy + " is not supported")
        return new_clock

    # ------------------------------
    # Functionality:
    #   determine the work-rate of each kernel.
    #   work-rate is how quickly each kernel can be done, which depends on it's bottleneck
    # ------------------------------
    def update_kernels_work_rate_for_next_tick(self):
        _ = [kernel.update_block_att_work_rate(self.scheduled_kernels) for kernel in self.scheduled_kernels]

    # ------------------------------
    # Functionality:
    #   how much work does each block do for each phase
    # ------------------------------
    def calc_design_work(self):
        for SOC_type, SOC_id in self.design.get_designs_SOCs():
            blocks_seen = []
            for kernel in self.scheduled_kernels:
                if kernel.SOC_type == SOC_type and kernel.SOC_id == SOC_id:
                    for block, work in kernel.block_phase_work_dict.items():
                        if block not in blocks_seen :
                            blocks_seen.append(block)
                        #if block in self.block_phase_work_dict.keys():
                        if self.phase_num in self.design.block_phase_work_dict[block].keys():
                            self.design.block_phase_work_dict[block][self.phase_num] += work[self.phase_num]
                        else:
                            self.design.block_phase_work_dict[block][self.phase_num] = work[self.phase_num]
            all_blocks = self.design.get_blocks()
            for block in all_blocks:
                if block in  blocks_seen:
                    continue
                self.design.block_phase_work_dict[block][self.phase_num] = 0

    # ------------------------------
    # Functionality:
    #   calculate the utilization of each block in the design
    # ------------------------------
    def calc_design_utilization(self):
        for SOC_type, SOC_id in self.design.get_designs_SOCs():
            for block,phase_work in self.design.block_phase_work_dict.items():
                if self.design.phase_latency_dict[self.phase_num] == 0:
                    work_rate = 0
                else:
                    work_rate = (self.design.block_phase_work_dict[block][self.phase_num])/self.design.phase_latency_dict[self.phase_num]
                self.design.block_phase_utilization_dict[block][self.phase_num] = work_rate/block.peak_work_rate

    # ------------------------------
    # Functionality:
    #   Aggregates the energy consumed for current phase over all the blocks
    # ------------------------------
    def calc_design_energy(self):
        for SOC_type, SOC_id in self.design.get_designs_SOCs():
            self.design.SOC_phase_energy_dict[(SOC_type, SOC_id)][self.phase_num] = \
                sum([kernel.stats.phase_energy_dict[self.phase_num] for kernel in self.scheduled_kernels
                    if kernel.SOC_type == SOC_type and kernel.SOC_id == SOC_id])
            if config.simulation_method == "power_knobs":
                # Add up the leakage energy to the total energy consumption
                # Please note the phase_leakage_energy_dict only counts for PE and IC energy (no mem included)
                # since memory cannot be cut-off; otherwise will lose its contents
                self.design.SOC_phase_energy_dict[(SOC_type, SOC_id)][self.phase_num] += \
                    sum([kernel.stats.phase_leakage_energy_dict[self.phase_num] for kernel in self.scheduled_kernels
                         if kernel.SOC_type == SOC_type and kernel.SOC_id == SOC_id])

                # Add the leakage power for memories
                self.design.SOC_phase_energy_dict[(SOC_type, SOC_id)][self.phase_num] += \
                    sum([block.get_leakage_power() * self.time_step_size for block in self.design.get_blocks()
                         if block.get_block_type_name() == "mem"])

    # ------------------------------
    # Functionality:
    #   step the simulator forward, by moving all the kernels forward in time
    # ------------------------------
    def step(self):
        # the time step of the previous phase
        self.time_step_size = self.clock_time - self.old_clock_time
        # add the time step (time spent in the phase) to the design phase duration dictionary
        self.design.phase_latency_dict[self.phase_num] = self.time_step_size

        # advance kernels
        self.step_kernels()

        # Aggregates the energy consumed for current phase over all the blocks
        self.calc_design_energy()   # needs be done after kernels have stepped, to aggregate their energy and divide
        self.calc_design_work()   # calculate how much work does each block do for this phase
        self.calc_design_utilization()

        self.update_scheduled_kernel_list()  # if a kernel is done, schedule it out
        self.schedule_kernels()  # schedule ready to be scheduled kernels
        self.old_clock_time = self.clock_time  # update clock

        # check if execution is completed or not!
        self.update_program_status()
        self.update_kernels_work_rate_for_next_tick()  # update each kernels' work rate

        self.phase_num += 1

        # return the new tick position
        return self.calc_new_tick_position(), self.program_status

    # ------------------------------
    # Functionality:
    #   call the simulator
    # ------------------------------
    def simulate(self, clock_time):
        self.tick(clock_time)
        return self.step()