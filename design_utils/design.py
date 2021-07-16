#Copyright (c) Facebook, Inc. and its affiliates.
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.
import _pickle as cPickle
from design_utils.components.hardware import *
from design_utils.components.workload import *
from design_utils.components.mapping import *
from design_utils.components.scheduling import *
from design_utils.components.krnel import *
from design_utils.common_design_utils import  *
import collections
import datetime
from datetime import datetime
from error_handling.custom_error import  *
import gc

if config.use_cacti:
    from misc.cacti_hndlr import cact_handlr

if config.simulation_method == "power_knobs":
    from specs import database_input_powerKnobs as database_input
elif config.simulation_method == "performance":
    from specs import database_input
else:
    raise NameError("Simulation method unavailable")


# This class logs the insanity (opposite of sanity (check), so the flaw) with the design
class Insanity:
    def __init__(self, task, block, name):
        self.name = name
        self.task = task
        self.block = block

    # the problematic task
    def set_task(self, task_):
        self.task = task_

    # the problematic block
    def set_block(self, block_):
        self.block = block_

    # name of the insanity
    def set_name(self, name_):
        self.name= name_

    def gen_msg(self):
        output = "sanity check failed with:  "
        output += "insanity name:"  + self.name
        if not(self.task== "_"):
            output += self.task.name
        if not(self.block == "_"):
            output += self.block.instance_name
        return output


# This class emulates a design point containing
# hardware/software, their mapping and scheduling
class ExDesignPoint:
    def __init__(self, hardware_graph:HardwareGraph):
        self.hardware_graph = hardware_graph   # hardware graph contains the hardware blocks
                                               # and their connections
        self.PA_prop_dict = {}   # PA prop is used for PA design generation
        self.id = str(-1)  # this means it hasn't been set
        self.valid = True
        self.FARSI_ex_id = str(-1)
        self.PA_knob_ctr_id = str(-1)

    def eliminate_system_bus(self):
        all_drams = [el for el in self.get_hardware_graph().get_blocks() if el.subtype == "dram"]
        ics_with_dram = []
        for dram in all_drams:
            for neigh in dram.get_neighs():
                if neigh.type == "ic":
                    ics_with_dram.append(neigh)

        # can only have one ic with dram hanging from it
        return len(ics_with_dram) == 1


    def has_system_bus(self):
        return False
        # find all the drams and their ics
        all_drams = [el for el in self.get_hardware_graph().get_blocks() if el.subtype == "dram"]
        ics_with_dram = []
        for dram in all_drams:
            for neigh in dram.get_neighs():
                if neigh.type == "ic":
                    ics_with_dram.append(neigh)

        # can only have one ic with dram hanging from it
        return len(ics_with_dram) == 1

    def get_system_bus(self):
        if not self.has_system_bus():
            return None
        else:
            all_drams = [el for el in self.get_hardware_graph().get_blocks() if el.subtype == "dram"]
            ics_with_dram = []
            for dram in all_drams:
                for neigh in dram.get_neighs():
                    if neigh.type == "ic":
                        neigh.set_as_system_bus()
                        return neigh

    # get hardware blocks of a design
    def get_blocks(self):
        return self.hardware_graph.blocks

    # get hardware blocks within a specific SOC of the design
    def get_blocks_of_SOC(self,SOC_type, SOC_id):
        return [block for block in self.hardware_graph.blocks if block.SOC_type == SOC_type and SOC_id == SOC_id]

    # get tasks (software tasks) of the design
    def get_tasks(self):
        return self.hardware_graph.get_all_tasks()

    def get_tasks_of_SOC(self, SOC_type, SOC_id):
        return [task for task in self.get_tasks() if task.SOC_type == SOC_type and SOC_id == SOC_id]

    # samples the task distribution within the hardware graph.
    # used for jitter modeling.
    def sample_hardware_graph(self, hw_sampling):
       self.hardware_graph.sample(hw_sampling)

    # get blocks that a task uses (host the task)
    def get_blocks_of_task(self, task):
        blocks = []
        for block in self.get_blocks():
            if task in block.get_tasks_of_block():
                blocks.append(block)
        return blocks

    # if set, the design is complete and valid
    def set_validity(self, validity):
        self.valid = validity

    def get_validity(self):
        return self.valid

    # delete this later. Used for debugging
    def check_mem_fronts_sanity(self):
        fronts_1 = sum([len(block.get_fronts("task_name_dir")) for block in self.get_blocks() if block.type == "mem"])
        fronts_2 = sum(
            [len(block.get_fronts("task_dir_work_ratio")) for block in self.get_blocks() if block.type == "mem"])


    def check_system_ic_exist(self, block):
        assert (block.type == "ic"), "should be checking this with non ic input"
        system_ic_exist = False
        connectd_ics = [block_ for block_ in block.get_neighs() if block_.type == "ic"]

        # iterate though the connected ics, get their neighbouring ics
        # and make sure there is a ic with only dram
        system_ic_list = []
        for neigh_ic in connectd_ics:
            has_dram = len([neigh for neigh in neigh_ic.get_neighs() if neigh.subtype == "dram"]) >= 1
            has_pe = len([neigh for neigh in neigh_ic.get_neighs() if neigh.type == "pe"]) >= 1
            if has_dram:
                if has_pe:
                    pass
                    #return False
                    #print(" system ic can not have a pe")
                    #exit(0)
                else:
                    system_ic_list.append(neigh_ic)

        if self.block_is_system_ic(block):
            system_ic_list.append(block)

        if len(set(system_ic_list)) > 1:
            print("can only have one system ic")
            exit(0)

        return len(system_ic_list) == 1


    def block_is_system_ic(self, block):
        assert (block.type == "ic"), "should be checking this with non ic input"
        # iterate though the connected ics, get their neighbouring ics
        # and make sure there is a ic with only dram
        system_ic_list = []
        has_dram = len([neigh for neigh in block.get_neighs() if neigh.subtype == "dram"]) >= 1
        has_pe = len([neigh for neigh in block.get_neighs() if neigh.type == "pe"]) >= 1
        if has_dram:
            if has_pe:
                pass
                #print(" system ic can not have a pe")
                #exit(0)
            else:
                return True
        else:
            return False
        return False

    # sanity check the design
    def sanity_check(self):
        insanity_list = [] # list of Inanities

        # fronts check
        fronts_1 = sum([len(block.get_fronts("task_name_dir")) for block in self.get_blocks() if block.type == "mem"])
        fronts_2 = sum([len(block.get_fronts("task_dir_work_ratio")) for block in self.get_blocks() if block.type== "mem"])
        if not fronts_1 == fronts_2:
            pre_mvd_fronts_1 = [block.get_fronts("task_name_dir") for block in self.get_blocks() if block.type == "mem"]
            pre_mvd_fronts_2 = [block.get_fronts("task_dir_work_ratio") for block in self.get_blocks() if block.type == "mem"]
            raise UnEqualFrontsError

        # all the tasks have pe and mem
        for task in self.get_tasks():
            pe_blocks = self.get_blocks_of_task_by_block_type(task, "pe")
            mem_blocks = self.get_blocks_of_task_by_block_type(task, "mem")
            if len(pe_blocks) == 0:
                print("task:" + task.name + " does not have any pes")
                insanity = Insanity("_", "_", "none")
                insanity.set_block("_")
                insanity.set_name("no_pe")
                insanity_list.append(insanity)
                pe_blocks = self.get_blocks_of_task_by_block_type(task, "pe")
                print(insanity.gen_msg())
                raise NoPEError
                #break
            elif (len(mem_blocks) == 0 and not("siink" in task.name)):
                print("task:" + task.name + " does not have any mems")
                insanity = Insanity("_", "_", "none")
                insanity.set_block("_")
                insanity.set_name("no_mem")
                insanity_list.append(insanity)
                print(insanity.gen_msg())
                mem_blocks = self.get_blocks_of_task_by_block_type(task, "mem")
                raise NoMemError
                #break

        # every pe or memory needs to be connected to a bus
        for block in self.get_blocks():
            if block.type in ["pe", "mem"]:
                connectd_ics = [True  for block_ in block.get_neighs() if block_.type =="ic" ]
                if len(connectd_ics) > 1:
                    print("block: " + block.instance_name + " is connected to more than one ic")
                    insanity = Insanity("_", "_", "multi_bus")
                    insanity.set_name("multi_bus")
                    insanity_list.append(insanity)
                    print(insanity.gen_msg())
                    raise MultiBusBlockError
                    #break
                elif len(connectd_ics) < 1:
                    print("block: " + block.instance_name + " is not connected any ic")
                    insanity = Insanity("_", "_", "none")
                    insanity.set_block(block)
                    insanity.set_name("no_bus")
                    insanity_list.append(insanity)
                    print(insanity.gen_msg())
                    raise NoBusError
                    #break

        # every bus needs to have at least one pe and mem
        for block in self.get_blocks():
            if block.type in ["ic"]:
                connectd_pes = [True  for block_ in block.get_neighs() if block_.type =="pe" ]
                connectd_mems = [True  for block_ in block.get_neighs() if block_.type =="mem" ]
                connectd_ics = [True  for block_ in block.get_neighs() if block_.type =="ic" ]

                system_ic_exist = self.check_system_ic_exist(block)

                if len(connectd_mems) == 0 and not system_ic_exist:
                    insanity = Insanity("_",block, "bus_with_no_mem")
                    print(insanity.gen_msg())
                    if self.hardware_graph.generation_mode == "user_generated":
                        print("deactivated Bus with No memory error, since hardware graph was directly user generated/parsed ")
                    else:
                        raise BusWithNoMemError
                    """
                    elif len(connectd_pes) > 0 and self.block_is_system_ic(block):
                        insanity = Insanity("_", block, "system_ic_with_pe")
                        insanity_list.append(insanity)
                        print(insanity.gen_msg())
                        if self.hardware_graph.generation_mode == "user_generated":
                            print(
                                "deactivated Bus with No Bus error, since hardware graph was directly user generated/parsed ")
                        else:
                            raise SystemICWithPEException
                    """
                elif len(connectd_pes) == 0 and not self.block_is_system_ic(block):
                    insanity = Insanity("_", block, "bus_with_no_pes")
                    insanity_list.append(insanity)
                    print(insanity.gen_msg())
                    if self.hardware_graph.generation_mode == "user_generated":
                        print("deactivated Bus with No Bus error, since hardware graph was directly user generated/parsed ")
                    else:
                        raise BusWithNoPEError
        # every design needs to have at least on pe, mem, and bus
        block_type_count_dict = {}
        block_type_count_dict["mem"] = 0
        block_type_count_dict["pe"] = 0
        block_type_count_dict["ic"] = 0
        for block in self.get_blocks():
            block_type_count_dict[block.type] +=1
        for type_, count in block_type_count_dict.items():
            if count < 1:
                print("no block of type " + type_ + " found")
                insanity = Insanity("_", "_", "none")
                insanity.set_name("not_enough_ip_of_certain_type")
                insanity_list.append(insanity)
                print(insanity.gen_msg())
                raise NotEnoughIPOfCertainType
                #break

       # every block should host at least one task
        for block in self.get_blocks():
            if block.type == "ic":   # since we unload
                continue
            if len(block.get_tasks_of_block()) == 0:
                print( "block: " + block.instance_name + " does not host any tasks")
                insanity = Insanity("_", "_", "none")
                insanity.set_block(block)
                insanity.set_name("no_task")
                insanity_list.append(insanity)
                print(insanity.gen_msg())
                raise BlockWithNoTaskError

    # get blocks within the design (filtered by type)
    def get_blocks_by_type(self, block_type):
        return [block for block in self.get_blocks() if block.type == block_type]

    # gets blocks for a task, and filter them based on hardware type (pe, mem, ic)
    def get_blocks_of_task_by_block_type(self, task, block_type):
        blocks_of_task = self.get_blocks_of_task(task)
        blocks_by_type = []
        for block in blocks_of_task:
            if block.type == block_type:
                blocks_by_type.append(block)
        return blocks_by_type


    def get_write_mem_tasks(self, task, mem):
        # get conncted ic
        ics = [el for el in mem.get_neighs() if el.type =="ic"]
        assert(len(ics) <= 1), "Each memory can be only connected to one bus master"

        # get the pipes
        pipes = self.get_hardware_graph().get_pipes_between_two_blocks(ics[0], mem, "write")
        assert(len(pipes) <= 1), "can only have one pipe (in a direction) between a memory and a ic"

        # traffic
        traffics = pipes[0].get_traffic()

        return [trf.child for trf in traffics if trf.parent.name == task.name]


    # for a specific task, find all the specific blocks of a type and their direction
    def get_blocks_of_task_by_block_type_and_task_dir(self, task, block_type, task_dir=""):
        assert ((block_type == "pe") != task_dir) # XORing the expression
        blocks_of_task = self.get_blocks_of_task(task)
        blocks_of_task_by_type = [block for block in blocks_of_task if block.type == block_type]
        blocks_of_task_by_type_and_task_dir = [block for block in blocks_of_task_by_type if block.get_task_dir_by_task_name(task)[0][1] == task_dir]
        return blocks_of_task_by_type_and_task_dir

    # get the properties of the design. This is used for the more accurate simulation
    def filter_props_by_keyword(self, knob_order, knob_values, type_name):
        prop_value_dict = collections.OrderedDict()
        for knob_name, knob_value in zip(knob_order, knob_values):
            if type_name+"_props" in knob_name:
                knob_name_refined = knob_name.split("__")[-1]
                prop_value_dict[knob_name_refined] = knob_value
        return prop_value_dict

    def filter_auto_tune_props(self, type_name, auto_tune_props):
        auto_tune_list = []
        for knob_name in auto_tune_props:
            if type_name+"_props" in knob_name:
                knob_name_refined = knob_name.split("__")[-1]
                auto_tune_list.append(knob_name_refined)
        return auto_tune_list

    # get id associated with a design. Each design has it's unique id.
    def get_ex_id(self):
        if self.id == str(-1):
            print("experiments id is:" + str(self.id) + ". This means id has not been set")
            exit(0)
        return self.id

    def update_ex_id(self, id):
        self.id = id

    def get_FARSI_ex_id(self):
        if self.FARSI_ex_id == str(-1):
            print("experiments id is:" + str(self.id) + ". This means id has not been set")
            exit(0)
        return self.FARSI_ex_id

    def get_PA_knob_ctr_id(self):
        if self.PA_knob_ctr_id == str(-1):
            print("experiments id is:" + str(self.PA_knob_ctr_id) + ". This means id has not been set")
            exit(0)
        return self.PA_knob_ctr_id

    def update_FARSI_ex_id(self, FARSI_ex_id):
        self.FARSI_ex_id = FARSI_ex_id

    def update_PA_knob_ctr_id(self, knob_ctr):
        self.PA_knob_ctr_id = knob_ctr

    def reset_PA_knobs(self, mode="batch"):
        if mode == "batch":
            # parse and set design props
            self.PA_prop_dict = collections.OrderedDict()
            # parse and set hw and update the props
            for keyword in ["pe", "ic", "mem"]:
                blocks_ = self.get_blocks_by_type(keyword)
                for block in blocks_:
                    block.reset_PA_props()
            # parse and set sw props
            for keyword in ["sw"]:
                tasks = self.get_tasks()
                for task_ in tasks:
                    task_.reset_PA_props()
        else:
            print("mode:" + mode + " is not defind for apply_PA_knobs")
            exit(0)

    def update_PA_knobs(self, knob_values, knob_order, all_auto_tunning_knobs, mode="batch"):
        if mode == "batch":
            # parse and set design props
            prop_value_dict = {}
            prop_value_dict["ex_id"] = self.get_ex_id()
            prop_value_dict["FARSI_ex_id"] = self.get_FARSI_ex_id()
            prop_value_dict["PA_knob_ctr_id"] = self.get_PA_knob_ctr_id()
            self.PA_prop_dict.update(prop_value_dict)
            # parse and set hw and update the props
            for keyword in ["pe", "ic", "mem"]:
                blocks_ = self.get_blocks_by_type(keyword)
                prop_value_dict = self.filter_props_by_keyword(knob_order, knob_values, keyword)
                prop_auto_tuning_list = self.filter_auto_tune_props(keyword, all_auto_tunning_knobs)
                for block in blocks_:
                    block.update_PA_props(prop_value_dict)
                    block.update_PA_auto_tunning_knob_list(prop_auto_tuning_list)
            # parse and set sw props
            for keyword in ["sw"]:
                tasks = self.get_tasks()
                prop_value_dict = self.filter_props_by_keyword(knob_order, knob_values, keyword)
                prop_auto_tuning_list = self.filter_auto_tune_props(keyword, all_auto_tunning_knobs)
                for task_ in tasks:
                    task_.update_PA_props(prop_value_dict)
                    task_.update_PA_auto_tunning_knob_list(prop_auto_tuning_list)
        else:
            print("mode:"+ mode +" is not defined for apply_PA_knobs")
            exit(0)

    # write all the props into the design file.
    # design file is set in the config file (config.verification_result_file)
    def dump_props(self, result_folder, mode="batch"): # batch means that all the blocks of similar type have similar props
        file_name = config.verification_result_file
        file_addr = os.path.join(result_folder, file_name)
        if mode == "batch":
            with open(file_addr, "a+") as output:
                for prop_name, prop_value in self.PA_prop_dict.items():
                    prop_name_modified = "\"design" + "__" + prop_name +"\""
                    if not str(prop_value).isdigit():
                        prop_value = "\"" + prop_value + "\""
                    output.write(prop_name_modified + ": " + str(prop_value) + ",\n")
                # writing the hardware props
                for keyword in ["pe", "ic", "mem"]:

                    block = self.get_blocks_by_type(keyword)[0] # since in batch mode, the first element shares the same prop values
                                                              # as all
                    for prop_name, prop_value in block.PA_prop_dict.items():
                        prop_name_modified = "\""+ keyword+"__"+prop_name + "\""
                        if "ic__/Buffer/enable" in prop_name_modified:   # this is just because now parsing throws an error
                            continue
                        if not str(prop_value).isdigit():
                           prop_value = "\"" + prop_value +"\""
                        output.write(prop_name_modified+": " + str(prop_value) + ",\n")
                # writing the software props
                for keyword in ["sw"]:
                    task_ = self.get_tasks()[0]
                    for prop_name, prop_value in task_.PA_prop_dict.items():
                        prop_name_modified = "\""+ keyword + "__" + prop_name +"\""
                        if not str(prop_value).isdigit():
                           prop_value = "\"" + prop_value +"\""
                        output.write(prop_name_modified + ": " + str(prop_value) + ",\n")

        else:
            print("mode:" + mode + " is not defind for apply_PA_knobs")

    def get_hardware_graph(self):
        return self.hardware_graph

    def get_task_by_name(self, task_name):
        return [task_ for task_ in self.get_tasks() if task_.name == task_name][0]


# collection of the simulated design points.
# not that you can query this container with the same functions as the
# SimDesignPoint (i.e, same modules are provide). However, this is not t
class SimDesignPointContainer:
    def __init__(self, design_point_list, database, reduction_mode = "avg"):
        self.design_point_list = design_point_list
        self.reduction_mode = reduction_mode  # how to reduce the results.
        self.database = database  # hw/sw database
        self.dp_rep = self.design_point_list[0]  # design representative
        self.dp_stats = DPStatsContainer(self, self.dp_rep, self.database, reduction_mode)    # design point stats
        self.dp = self.dp_rep  # design point is used to fill up some default values
                               # we use dp_rep, i.e., design point representative for this

        self.move_applied = None
        self.dummy_tasks = [krnl.get_task() for krnl in self.dp.get_kernels() if (krnl.get_task()).is_task_dummy()]

    def get_dummy_tasks(self):
        return self.dummy_tasks

    # bootstrap the design from scratch
    def reset_design(self, workload_to_hardware_map=[], workload_to_hardware_schedule=[]):
        self.dp_rep.reset_design()

    def set_move_applied(self, move_applied):
        self.move_applied = move_applied

    def get_move_applied(self):
        return self.move_applied


    def get_dp_stats(self):
        return self.dp_stats
    # -----------------
    # getters
    # -----------------
    def get_task_graph(self):
        return self.dp_rep.get_hardware_graph().get_task_graph()

    # Functionality:
    #       get the mapping
    def get_workload_to_hardware_map(self):
        return self.dp_rep.get_workload_to_hardware_map()

    # Functionality
    #       get the scheduling
    def get_workload_to_hardware_schedule(self):
        return self.dp_rp.get_workload_to_hardware_schedule()

    def get_kernels(self):
        return self.dp_rp.get_kernels()

    def get_kernel_by_task_name(self, task: Task):
        return self.dp_rep.get_kernel_by_task_name(task)

    # get the kernels of the design
    def get_kernels(self):
        return self.dp_rep.get_kernels()

    # get the sw to hw mapping
    def get_workload_to_hardware_map(self):
        return self.dp_rep.get_workload_to_hardware_map()

    # get the SOCs that the design resides in
    def get_designs_SOCs(self):
        return self.dp_rep.get_designs_SOCs()

    # get all the design points
    def get_design_point_list(self):
        return self.design_point_list

    # get the representative design point.
    def get_dp_rep(self):
        return self.dp_rep


# Container for all the design point stats.
# In order to collect profiling information, we reduce the statistics
# according to the desired reduction function.
# reduction semantically happens at two different levels depending on the question
# that we are asking.
#   Level 1 Questions: Within/intra design questions to compare components of a
#                      single design. Example: finding the hottest kernel?
#            To answer, reduce the results across at the task/kernel
#   level 2 Questions: Across/inter design question to compare different designs?
#            To answer, reduce the results from the end-to-end perspective, i.e.,
#            reduce(end-to-end latency), reduce(end-to-end energy), ...
# PS: at the moment,  a design here is defined as a sw/hw tuple with only sw
#     characteristic changing.
class DPStatsContainer():
    def __init__(self, sim_dp_container, dp_rep, database, reduction_mode):
        self.comparison_mode = "latency"  # metric to compare different design points
        self.sim_dp_container = sim_dp_container
        self.design_point_list = self.sim_dp_container.design_point_list # design point container (containing list of designs)
        self.dp_rep = dp_rep #self.dp_container[0] # which design to use as representative (for plotting and so on
        self.__kernels = self.sim_dp_container.design_point_list[0].get_kernels()
        self.SOC_area_dict = defaultdict(lambda: defaultdict(dict))  # area of all blocks within each SOC
        self.SOC_area_subtype_dict = defaultdict(lambda: defaultdict(dict))  # area of all blocks within each SOC
        self.system_complex_area_dict = defaultdict()
        self.SOC_metric_dict = defaultdict(lambda: defaultdict(dict))
        self.system_complex_metric_dict = defaultdict(lambda: defaultdict(dict))
        self.system_complex_area_dram_non_dram = defaultdict(lambda: defaultdict(dict))
        self.database = database # hw/sw database
        self.latency_list =[]  # list of latency values associated with each design point
        self.power_list =[]   # list of power values associated with each design point
        self.energy_list =[]  # list of energy values associated with each design point
        self.reduction_mode = reduction_mode  # how to statistically reduce the values
        # collect the data
        self.collect_stats()
        self.dp = self.sim_dp_container  # container that has all the designs
        self.parallel_kernels = dp_rep.parallel_kernels


    def get_parallel_kernels(self):
        return self.parallel_kernels

    # helper function to apply an operator across two dictionaries
    def operate_on_two_dic_values(self,dict1, dict2, operator):
        dict_res = {}
        for key in list(dict2.keys()) + list(dict1.keys()):
            if key in dict1.keys() and dict2.keys():
                dict_res[key] = operator(dict2[key], dict1[key])
            else:
                if key in dict1.keys():
                    dict_res[key] = dict1[key]
                elif key in dict2.keys():
                    dict_res[key] = dict2[key]
        return dict_res

    # operate on multiple dictionaries. The operation is determined by the operator input
    def operate_on_dicionary_values(self, dictionaries, operator):
        res = {}
        for SOCs_latency in dictionaries:
            #res = copy.deepcopy(self.operate_on_two_dic_values(res, SOCs_latency, operator))
            #gc.disable()
            res = cPickle.loads(cPickle.dumps(self.operate_on_two_dic_values(res, SOCs_latency, operator), -1))
            #gc.enable()
        return res

    # reduce the (list of) values based on a statistical  parameter (such as average)
    def reduce(self, list_):
        if self.reduction_mode == 'avg':
            if isinstance(list_[0],dict):
                dict_added = self.operate_on_dicionary_values(list_, operator.add)
                for key,val in dict_added.items():
                    dict_added[key] = val/len(list_)
                return dict_added
            else:
                return sum(list_)/len(list_)
        elif self.reduction_mode == 'min':
            return min(list_)
        elif self.reduction_mode == 'max':
            #if (len(list_) == 0):
            #    print("What")
            return max(list_)
        else:
            print("reduction mode "+ self.reduction_mode + ' is not defined')
            exit(0)

    # iterate through all the design points and
    # collect their stats
    def collect_stats(self):
        for type, id in self.dp_rep.get_designs_SOCs():
            # level 1 reduction for intra design questions
            self.intra_design_reduction(type, id)
            # level 2 questions for across/inter design questions
            self.inter_design_reduction(type, id)

    # level 1 reduction for intra design questions
    def intra_design_reduction(self, SOC_type, SOC_id):
        kernel_latency_dict = {}
        latency_list = []
        kernel_metric_values = defaultdict(lambda: defaultdict(list))
        for dp in self.sim_dp_container.design_point_list:
            for kernel_ in dp.get_kernels():
                for metric in config.all_metrics:
                    kernel_metric_values[kernel_.get_task_name()][metric].append\
                        (kernel_.stats.get_metric(metric))

        for kernel in self.__kernels:
            for metric in config.all_metrics:
                kernel.stats.set_stats_directly(metric,
                    self.reduce(kernel_metric_values[kernel.get_task_name()][metric]))

    def get_kernels(self):
        return self.__kernels

    # Functionality: level 2 questions for across/inter design questions
    def inter_design_reduction(self, SOC_type, SOC_id):
        for metric_name in config.all_metrics:
            self.set_SOC_metric_value(metric_name, SOC_type, SOC_id)
            self.set_system_complex_metric(metric_name)  # data per System

    # hot = longest latency
    def get_hot_kernel_SOC(self, SOC_type, SOC_id, metric="latency", krnel_rank=0):
        kernels_on_SOC = [kernel for kernel in self.__kernels if kernel.SOC_type == SOC_type and kernel.SOC_id == SOC_id]
        for k in kernels_on_SOC:
            if (k.stats.get_metric(metric) is None):
                print("metric is " + metric)
        sorted_kernels_hot_to_cold = sorted(kernels_on_SOC, key=lambda kernel: kernel.stats.get_metric(metric), reverse=True)
        return sorted_kernels_hot_to_cold[krnel_rank]

    # get the hot kernels if the system. Hot means the bottleneck or rather the
    # most power/energy/area/performance consuming of the system. This is determined
    # by the input argument metric.
    # Variables:
    #   knrle_rank: rank of the kernel to pick once we have already sorted the kernels
    #               based on hot ness. 0, means the hottest and higher values mean colder ones.
    #               We use this value to sometimes target less hot kernels if the hot kernel
    #               can not be improved any mot less hot kernels if the hot kernel
    #               can not be improved any more.
    def get_hot_kernel_system_complex(self, metric="latency", krnel_rank=0):
        hot_krnel_list = []
        for SOC_type, SOC_id in self.get_designs_SOCs():
            hot_krnel_list.append(self.get_hot_kernel_SOC(SOC_type, SOC_id, metric, krnel_rank))

        return sorted(hot_krnel_list, key=lambda kernel: kernel.stats.get_metric(metric), reverse=True)[0]

    # sort the blocks for a kernel based how much impact they have on a metric
    def get_hot_block_of_krnel_sorted(self, krnl_task_name, metric="latency"):
        # find the hottest kernel
        #hot_krnel = self.get_hot_kernel_SOC(SOC_type, SOC_id, metric, krnel_rank)
        krnel_of_interest = [krnel for  krnel in self.__kernels if krnel.get_task_name() == krnl_task_name]
        assert(len(krnel_of_interest) == 1), "can't have no krnel with this name or more than one"
        krnl = krnel_of_interest[0]

        # find the hot block accordingly
        # TODO: this is not quit right since
        #       hot kernel of different designs might have different
        #       block bottlenecks, but here we just use the
        #       the block bottleneck of the representative design
        #       since self.__kernels are set to this designs kernels
        kernel_blck_sorted : Block = krnl.stats.get_block_sorted(metric)
        return kernel_blck_sorted

    # -------------------------------------------
    # Functionality:
    #   get the hot block among the blocks that a kernel resides in based how much impact they have on a metric.
    #   Hot means the bottleneck or rather the
    #   most power/energy/area/performance consuming of the system. This is determined
    #   by the input argument metric.
    # -------------------------------------------
    def get_hot_block_of_krnel(self, krnl_task_name, metric="latency"):
        # find the hottest kernel
        #hot_krnel = self.get_hot_kernel_SOC(SOC_type, SOC_id, metric, krnel_rank)
        krnel_of_interest = [krnel for  krnel in self.__kernels if krnel.get_task_name() == krnl_task_name]
        assert(len(krnel_of_interest) == 1), "can't have no krnel with this name or more than one"
        krnl = krnel_of_interest[0]


        # find the hot block accordingly
        # TODO: this is not quit right since
        #       hot kernel of different designs might have different
        #       block bottlenecks, but here we just use the
        #       the block bottleneck of the representative design
        #       since self.__kernels are set to this designs kernels
        kernel_blck_bottleneck: Block = krnl.stats.get_block_bottleneck(metric)
        return kernel_blck_bottleneck

    # -------------------------------------------
    # Functionality:
    #   get the hot block among the blocks of the entire SOC based on the metric and kernel rank.
    #   Hot means the bottleneck or rather the
    #   most power/energy/area/performance consuming of the system. This is determined
    #   by the input argument metric.
    # Variables:
    #       krnel_rank: rank of the kernel to pick once we have already sorted the kernels
    #               based on hot ness. 0, means the hottest and higher values mean colder ones.
    #               We use this value to sometimes target less hot kernels if the hot kernel
    #               can not be improved any mot less hot kernels if the hot kernel
    #               can not be improved any more.
    # -------------------------------------------
    def get_hot_block_SOC(self, SOC_type, SOC_id, metric="latency", krnel_rank=0):
        # find the hottest kernel
        hot_krnel = self.get_hot_kernel_SOC(SOC_type, SOC_id, metric, krnel_rank)

        # find the hot block accordingly
        # TODO: this is not quit right since
        #       hot kernel of different designs might have different
        #       block bottlenecks, but here we just use the
        #       the block bottleneck of the representative design
        #       since self.__kernels are set to this designs kernels
        hot_kernel_blck_bottleneck:Block = hot_krnel.stats.get_block_bottleneck(metric)
        return hot_kernel_blck_bottleneck
        # corresponding block bottleneck. We need this since we make a copy of the the sim_dp,
        # and hence, sim_dp and ex_dp won't be synced any more
        #coress_hot_kernel_blck_bottleneck = self.find_cores_hot_kernel_blck_bottleneck(ex_dp, hot_kernel_blck_bottleneck)
        #return cores_hot_kernel_blck_bottleneck

    # -------------------------------------------
    # Functionality:
    #   get the hot block among the blocks of the entire system complex based on the metric and kernel rank.
    #   Hot means the bottleneck or rather the
    #   most power/energy/area/performance consuming of the system. This is determined
    #   by the input argument metric.
    # Variables:
    #       krnel_rank: rank of the kernel to pick once we have already sorted the kernels
    #               based on hot ness. 0, means the hottest and higher values mean colder ones.
    #               We use this value to sometimes target less hot kernels if the hot kernel
    #               can not be improved any mot less hot kernels if the hot kernel
    #               can not be improved any more.
    # -------------------------------------------
    def get_hot_block_system_complex(self, metric="latency", krnel_rank=0):
        hot_blck_list = []
        for SOC_type, SOC_id in self.get_designs_SOCs():
            hot_blck_list.append(self.get_hot_block_SOC(SOC_type, SOC_id, metric, krnel_rank))

        return sorted(hot_blck_list, key=lambda blck: blck.get_metric(metric), reverse=True)[0]

    # -------------------------------------------
    # Functionality:
    #   calculating the metric (power,performance,area) value
    # Variables:
    #   metric_type: which metric to calculate for
    #   SOC_type: type of the SOC, since we can accept multi SOC designs
    #   SOC_id: id of the SOC to target
    # -------------------------------------------
    def calc_SOC_metric_value(self, metric_type, SOC_type, SOC_id):
        self.unreduced_results = []
        # call dp_stats of each design and then reduce
        for dp in self.sim_dp_container.design_point_list:
            self.unreduced_results.append(dp.dp_stats.get_SOC_metric_value(metric_type, SOC_type, SOC_id))
        return self.reduce(self.unreduced_results)

    # -------------------------------------------
    # Functionality:
    #   calculating the area value
    # Variables:
    #   type: mem, ic, pe
    #   SOC_type: type of the SOC, since we can accept multi SOC designs
    #   SOC_id: id of the SOC to target
    # -------------------------------------------
    def calc_SOC_area_base_on_type(self, type_, SOC_type, SOC_id):
        area_list = []
        for dp in self.sim_dp_container.design_point_list:
            area_list.append(dp.dp_stats.get_SOC_area_base_on_type(type_, SOC_type, SOC_id))
        return self.reduce(area_list)

    def calc_SOC_area_base_on_subtype(self, subtype_, SOC_type, SOC_id):
        area_list = []
        for dp in self.sim_dp_container.design_point_list:
            area_list.append(dp.dp_stats.get_SOC_area_base_on_subtype(subtype_, SOC_type, SOC_id))
        return self.reduce(area_list)


    def set_SOC_metric_value(self,metric_type, SOC_type, SOC_id):
        assert(metric_type in config.all_metrics), metric_type + " is not supported"
        if metric_type == "area":
            for block_type in ["mem", "ic", "pe"]:
                self.SOC_area_dict[block_type][SOC_type][SOC_id] = self.calc_SOC_area_base_on_type(block_type, SOC_type, SOC_id)
            for block_subtype in ["dram", "sram", "ic", "ip", "gpp"]:
                self.SOC_area_subtype_dict[block_subtype][SOC_type][SOC_id] = self.calc_SOC_area_base_on_subtype(block_subtype, SOC_type, SOC_id)
        self.SOC_metric_dict[metric_type][SOC_type][SOC_id] =  self.calc_SOC_metric_value(metric_type, SOC_type, SOC_id)

    def set_system_complex_metric(self, metric_type):
        type_id_list = self.dp_rep.get_designs_SOCs()
        # only corner case is for area as
        # we want area specific even to the blocks
        if (metric_type == "area"):
            for block_type in ["pe", "mem", "ic"]:
                for type_, id_ in type_id_list:
                    self.system_complex_area_dict[block_type] = sum([self.get_SOC_area_base_on_type(block_type, type_, id_)
                                                             for type_, id_ in type_id_list])

            self.system_complex_area_dram_non_dram["non_dram"] = 0
            for block_subtype in ["sram", "ic", "gpp", "ip"]:
                for type_, id_ in type_id_list:
                    self.system_complex_area_dram_non_dram["non_dram"] += sum([self.get_SOC_area_base_on_subtype(block_subtype, type_, id_)
                                                             for type_, id_ in type_id_list])
            for block_subtype in ["dram"]:
                for type_, id_ in type_id_list:
                    self.system_complex_area_dram_non_dram["dram"] = sum([self.get_SOC_area_base_on_subtype(block_subtype, type_, id_)
                                                             for type_, id_ in type_id_list])

        if metric_type in ["area", "energy", "cost"]:
            self.system_complex_metric_dict[metric_type] = sum([self.get_SOC_metric_value(metric_type, type_, id_)
                                                                for type_, id_ in type_id_list])
        elif metric_type in ["latency"]:
            self.system_complex_metric_dict[metric_type] = self.operate_on_dicionary_values([self.get_SOC_metric_value(metric_type, type_, id_)
                                                                                             for type_, id_ in type_id_list], operator.add)
        elif metric_type in ["power"]:
            self.system_complex_metric_dict[metric_type] = max([self.get_SOC_metric_value(metric_type, type_, id_)
                                                                 for type_, id_ in type_id_list])
        else:
            raise Exception("metric_type:" + metric_type + " is not supported")

    # ------------------------
    # getters
    # ------------------------
    # sort kernels. At the moment we just sort based on latency.
    def get_kernels_sort(self):
        def get_kernels_sort(self):
            sorted_kernels_hot_to_cold = sorted(self.__kernels, key=lambda kernel: kernel.stats.latency, reverse=True)
            return sorted_kernels_hot_to_cold

    # return the metric of interest for the SOC. metric_type is the metric you are interested in
    def get_SOC_metric_value(self, metric_type, SOC_type, SOC_id):
        return self.SOC_metric_dict[metric_type][SOC_type][SOC_id]

    def get_SOC_area_base_on_type(self, block_type, SOC_type, SOC_id):
        assert(block_type in ["pe", "ic", "mem"]), "block_type" + block_type + " is not supported"
        return self.SOC_area_dict[block_type][SOC_type][SOC_id]

    def get_SOC_area_base_on_subtype(self, block_subtype, SOC_type, SOC_id):
        assert(block_subtype in ["dram", "sram", "gpp", "ip", "ic"]), "block_subtype" + block_subtype + " is not supported"
        if block_subtype not in self.SOC_area_subtype_dict.keys():  # this element does not exist
            return 0
        return self.SOC_area_subtype_dict[block_subtype][SOC_type][SOC_id]

    # return the metric of interest for the system complex. metric_type is the metric you are interested in.
    # Note that system complex can contain multiple SOCs.
    def get_system_complex_metric(self, metric_type):
        return self.system_complex_metric_dict[metric_type]

    def get_system_complex_area_stacked_dram(self):
        return self.system_complex_area_dram_non_dram


    # get system_complex area. type_ is selected from ("pe", "mem", "ic")
    def get_system_complex_area_base_on_type(self, type_):
        return self.system_complex_area_type[type_]

    def get_designs_SOCs(self):
        return self.dp_rep.get_designs_SOCs()

    # check if dp_rep is meeting the budget
    def workload_fits_budget_for_metric(self, workload, metric_name, budget_coeff):
        for type, id in self.dp_rep.get_designs_SOCs():
            if not all(self.fits_budget_for_metric_and_workload(type, id, metric_name, workload, 1)):
                return False
        return True



    # check if dp_rep is meeting the budget
    def workload_fits_budget(self, workload, budget_coeff):
        for type, id in self.dp_rep.get_designs_SOCs():
            for metric_name in self.database.get_budgetted_metric_names(type):
                if not all(self.fits_budget_for_metric_and_workload(type, id, metric_name, workload, 1)):
                    return False
        return True


    # check if dp_rep is meeting the budget
    def fits_budget(self, budget_coeff):
        for type, id in self.dp_rep.get_designs_SOCs():
            for metric_name in self.database.get_budgetted_metric_names(type):
                if not all(self.fits_budget_for_metric(type, id, metric_name, 1)):
                    return False
        return True

    def fits_budget_for_metric_for_SOC(self, metric_name, budget_coeff):
        for type, id in self.dp_rep.get_designs_SOCs():
            if not all(self.fits_budget_for_metric(type, id, metric_name, 1)):
                return False
        return True


    # returns a list of values
    def fits_budget_for_metric_and_workload(self, type, id, metric_name, workload, budget_coeff):
        dist = self.normalized_distance_for_workload(type, id, metric_name, workload)
        if not isinstance(dist, list):
            dist = [dist]
        return [dist_el<.001 for dist_el in dist]


    # returns a list of values
    def fits_budget_for_metric(self, type, id, metric_name, budget_coeff):
       dist = self.normalized_distance(type, id, metric_name)
       if not isinstance(dist, list):
           dist = [dist]
       return [dist_el<.001 for dist_el in dist]

    def normalized_distance_for_workload(self, type, id, metric_name, dampening_coeff=1):
        if config.dram_stacked:
            return self.normalized_distance_for_workload_for_stacked_dram(type, id, metric_name, dampening_coeff)
        else:
            return self.normalized_distance_for_workload_for_non_stacked_dram(type, id, metric_name, dampening_coeff)

    # normalized the metric to the budget
    def normalized_distance_for_workload_for_non_stacked_dram(self, type, id, metric_name, workload, dampening_coeff=1):
        metric_val = self.get_SOC_metric_value(metric_name, type, id)
        if isinstance(metric_val, dict):
            value_list= []
            for workload_, val in  metric_val.items():
                if not (workload == workload_):
                    continue
                dict_ = self.database.get_ideal_metric_value(metric_name, type)
                value_list.append((val - dict_[workload])/(dampening_coeff*dict_[workload]))
            return value_list
        else:
            return [(metric_val - self.database.get_ideal_metric_value(metric_name, type))/ (dampening_coeff*self.database.get_ideal_metric_value(metric_name, type))]

    def normalized_distance_for_workload_for_stacked_dram(self, type, id, metric_name, workload, dampening_coeff=1):
        metric_val = self.get_SOC_metric_value(metric_name, type, id)
        if metric_name == 'latency':
            value_list= []
            for workload_, val in  metric_val.items():
                if not (workload == workload_):
                    continue
                dict_ = self.database.get_ideal_metric_value(metric_name, type)
                value_list.append((val - dict_[workload_])/(dampening_coeff*dict_[workload_]))
            return value_list
        elif metric_name == "area":
            # get area aggregation of all the SOC minus dram and normalize it
            subtypes_no_dram = ["gpp", "ip", "ic", "sram"]
            area_no_dram = 0
            for el in subtypes_no_dram:
                area_no_dram += self.get_SOC_area_base_on_subtype(el, type, id)
            area_no_dram_norm = (area_no_dram - self.database.get_ideal_metric_value(metric_name, type))/ (dampening_coeff*self.database.get_ideal_metric_value(metric_name, type))
            # get dram area and normalize it
            area_dram = self.get_SOC_area_base_on_subtype("dram", type, id)
            area_dram_norm = [(area_dram - self.database.get_ideal_metric_value(metric_name, type))/ (dampening_coeff*self.database.get_ideal_metric_value(metric_name, type))]
            return  [area_no_dram_norm, area_dram]
        else:
            return [(metric_val - self.database.get_ideal_metric_value(metric_name, type))/ (dampening_coeff*self.database.get_ideal_metric_value(metric_name, type))]

    def normalized_distance(self, type, id, metric_name, dampening_coeff=1):
        if config.dram_stacked:
           return self.normalized_distance_for_stacked_dram(type, id, metric_name, dampening_coeff)
        else:
            return self.normalized_distance_for_non_stacked_dram(type, id, metric_name, dampening_coeff)

    # normalized the metric to the budget
    def normalized_distance_for_non_stacked_dram(self, type, id, metric_name, dampening_coeff=1):
        metric_val = self.get_SOC_metric_value(metric_name, type, id)
        if isinstance(metric_val, dict):
            value_list= []
            for workload, val in  metric_val.items():
                dict_ = self.database.get_ideal_metric_value(metric_name, type)
                value_list.append((val - dict_[workload])/(dampening_coeff*dict_[workload]))
            return value_list
        else:
            return [(metric_val - self.database.get_ideal_metric_value(metric_name, type))/ (dampening_coeff*self.database.get_ideal_metric_value(metric_name, type))]

    def normalized_distance_for_stacked_dram(self, type, id, metric_name, dampening_coeff=1):
        metric_val = self.get_SOC_metric_value(metric_name, type, id)
        if metric_name == 'latency':
            value_list= []
            for workload, val in  metric_val.items():
                dict_ = self.database.get_ideal_metric_value(metric_name, type)
                value_list.append((val - dict_[workload])/(dampening_coeff*dict_[workload]))
            return value_list
        elif metric_name == "area":
            # get area aggregation of all the SOC minus dram and normalize it
            subtypes_no_dram = ["gpp", "ip", "ic", "sram"]
            area_no_dram = 0
            for el in subtypes_no_dram:
                area_no_dram += self.get_SOC_area_base_on_subtype(el, type, id)
            area_no_dram_norm = (area_no_dram - self.database.get_ideal_metric_value(metric_name, type))/ (dampening_coeff*self.database.get_ideal_metric_value(metric_name, type))
            # get dram area and normalize it
            area_dram = self.get_SOC_area_base_on_subtype("dram", type, id)
            area_dram_norm = (area_dram - self.database.get_ideal_metric_value(metric_name, type))/ (dampening_coeff*self.database.get_ideal_metric_value(metric_name, type))
            return  [area_no_dram_norm, area_dram_norm]
        else:
            return [(metric_val - self.database.get_ideal_metric_value(metric_name, type))/ (dampening_coeff*self.database.get_ideal_metric_value(metric_name, type))]




    # normalized to the budget
    def dist_to_goal_per_metric(self, metric_name, mode):
        dist_list = []
        for type, id in self.dp_rep.get_designs_SOCs():
            meet_the_budgets = self.fits_budget_for_metric(type, id, metric_name, 1)
            for idx, meet_the_budget in enumerate(meet_the_budgets):
                if meet_the_budget:
                    if mode == "eliminate":
                        dist_list.append(0.000000001)
                    elif mode == "dampen" and meet_the_budget:
                        norm_dist = [math.fabs(el) for el in
                                     self.normalized_distance(type, id, metric_name, config.annealing_dampening_coef)]
                        dist_list.append(math.fabs(norm_dist[idx]))
                    elif mode == "simple":
                        norm_dist = [math.fabs(el) for el in
                                     self.normalized_distance(type, id, metric_name, 1)]
                        dist_list.append(math.fabs(norm_dist[idx]))
                    else:
                        print("mode: " + mode + " is not defined for dist_to_goal_per_metric")
                        exit(0)
                else:
                    norm_dist = [math.fabs(el) for el in
                                 self.normalized_distance(type, id, metric_name, 1)]
                    dist_list.append(math.fabs(norm_dist[idx]))
        city_dist = sum(dist_list)
        return city_dist


    # check if dp_rep is meeting the budget
    # modes: {"simple", "eliminate", "dampen"}.
    #        Simple: just calculates the city distance
    #        eliminates: eliminate the metric that has already met the budget
    #        dampen: dampens the impact of the metric that has already met the budget
    def dist_to_goal(self,  metrics_to_look_into=["all"], mode="simple"):  # mode simple, just calculate
        if metrics_to_look_into == ["all"]:
            metrics_to_look_into = self.database.get_budgetted_metric_names_all_SOCs() + self.database.get_other_metric_names_all_SOCs()  # which metrics to use for distance calculation

        dist_list = []
        for metric_name in metrics_to_look_into:
            dist_list.append(self.dist_to_goal_per_metric(metric_name, mode))

        city_dist = sum(dist_list)   # we use city distance to allow for probability prioritizing
        return city_dist

    # todo: change, right now it only uses the reduce value
    def __lt__(self, other):
        comp_list = []
        for metric in config.objectives:
            comp_list.append(self.get_system_complex_metric(metric) < other.get_system_complex_metric(metric))
        return all(comp_list)

    # todo: change, right now it only uses the reduce value
    def __gt__(self, other):
        comp_list = []
        for metric in config.objectives:
            comp_list.append(self.get_system_complex_metric(metric) > other.get_system_complex_metric(metric))
        return all(comp_list)


# This module emulates the simulated design point.
# It contains the information for the simulation of a design point
class SimDesignPoint(ExDesignPoint):
    def __init__(self, hardware_graph, workload_to_hardware_map=[], workload_to_hardware_schedule=[]):
        # primitive variables
        self.__workload_to_hardware_map:WorkloadToHardwareMap = None
        self.__workload_to_hardware_schedule:WorkloadToPEBlockSchedule = None
        self.hardware_graph = hardware_graph  # contains all the hardware blocks + their topology (how they are connected)

        self.__hardware, self.__workload, self.__kernels = [[]]*3
        # bootstrap the design and it's stats
        self.reset_design(workload_to_hardware_map, workload_to_hardware_schedule)

        self.SOC_phase_energy_dict = defaultdict(dict)  # energy associated with each phase
        self.phase_latency_dict = {}   # duration (time) for each phase.
        self.dp_stats = None   # design point statistics
        self.block_phase_work_dict = {}  # work done by the block as the system goes through different phases
        self.block_phase_utilization_dict = {}  # utilization done by the block as the system goes through different phases
        self.pipe_cluster_path_phase_work_rate_dict = {}
        self.parallel_kernels = {}
        self.krnl_phase_present = {}
        self.phase_krnl_present = {}
        self.iteration_number = 0  # the iteration which the simulation is done
        self.population_observed_number = 0
        self.depth_number = 0  # the depth (within on iteration) which the simulation is done
        self.simulation_time = 0  # how long did it take to do the simulation
        if config.use_cacti:
            self.cacti_hndlr = cact_handlr.CactiHndlr(config.cact_bin_addr, config.cacti_param_addr,
                                                      config.cacti_data_log_file, config.cacti_input_col_order,
                                                      config.cacti_output_col_order)

        for block in self.get_blocks():
            self.block_phase_work_dict[block] = {}
            self.block_phase_utilization_dict[block] = {}

    def set_simulation_time(self, simulation_time):
        self.simulation_time= simulation_time

    def get_simulation_time(self):
        return self.simulation_time

    def set_iteration_number(self, iteration_number):
        self.iteration_number = iteration_number

    def set_population_observed_number(self, population_observed_number):
        self.population_observed_number = population_observed_number

    def set_depth_number(self, depth_number):
        self.depth_number = depth_number

    def get_depth_number(self):
        return self.depth_number

    def get_iteration_number(self):
        return self.iteration_number

    def get_population_observed_number(self):
        return self.population_observed_number


    def get_tasks_parallel_task_dynamically(self, task):
        if task.is_task_dummy():
            return []
        krnl = self.get_kernel_by_task_name(task)

        phases_present = self.krnl_phase_present[krnl]
        parallel_krnls = []
        for phase_ in phases_present:
            parallel_krnls.extend(self.phase_krnl_present[phase_])

        # get_rid_of duplicates
        parallel_tasks = set([el.get_task_name() for el in set(parallel_krnls) if not(task.get_name() == el.get_task_name())])

        return list(parallel_tasks)


    def get_tasks_using_the_different_pipe_cluster(self, task, block):
        task_pipe_clusters = block.get_pipe_clusters_of_task(task)
        tasks_of_block = block.get_tasks_of_block()
        results = []
        for task_ in tasks_of_block:
            if task == task_:
               continue
            task__pipe_clusters = block.get_pipe_clusters_of_task(task_)
            if len(list(set(task_pipe_clusters) - set(task__pipe_clusters))) == len(task_pipe_clusters):
                results.append(task_.get_name())
        return results


    # Log the BW data about all the connections it the system
    def dump_mem_bus_connection_bw(self, result_folder):  # batch means that all the blocks of similar type have similar props
        file_name = "bus_mem_connection_max_bw.txt"
        file_addr = os.path.join(result_folder, file_name)
        buses = self.get_blocks_by_type("ic")

        with open(file_addr, "a+") as output:
            output.write("MasterInstance" +"," + "SlaveInstance" + ","+ "bus_bandwidth" + "," + "mode" + "\n")
            for bus in buses:
                connectd_pes = [block_ for block_ in bus.get_neighs() if block_.type =="pe" ]  # PEs connected to bus
                connectd_mems = [block_ for block_ in bus.get_neighs() if block_.type =="mem" ]  # memories connected to the bus
                connectd_ics = [block_ for block_ in bus.get_neighs() if block_.type =="ic"]
                for ic in connectd_ics:
                    for mode in ["read", "write"]:
                        output.write(ic.instance_name + "," + bus.instance_name + "," +
                                     str(ic.peak_work_rate) + "," + mode + "\n")
                for pe in connectd_pes:
                    for mode in ["read", "write"]:
                        output.write(pe.instance_name + "," + bus.instance_name + "," +
                                     str(bus.peak_work_rate) + "," + mode + "\n")
                for mem in connectd_mems:
                    for mode in ["read", "write"]:
                        output.write(bus.instance_name +  "," + mem.instance_name + ","+
                                     str(mem.peak_work_rate) + "," + mode + "\n")

    # -----------------------------------------
    # -----------------------------------------
    #           CACTI handling functions
    # -----------------------------------------
    # -----------------------------------------


    # Conversion of  memory type (naming) from FARSI to CACTI
    def FARSI_to_cacti_mem_type_converter(self, mem_subtype):
        if mem_subtype == "dram":
            return "main memory"
        elif mem_subtype == "sram":
            return "ram"

    # Conversion of  memory type (naming) from FARSI to CACTI
    def FARSI_to_cacti_cell_type_converter(self, mem_subtype):
        if mem_subtype == "dram":
            #return "lp-dram"
            return "comm-dram"
        elif mem_subtype == "sram":
            return "itrs-lop"

    # run cacti to get results
    def run_and_collect_cacti_data(self, blk, database):
        tech_node = {}
        tech_node["energy"] = 1
        tech_node["area"] = 1
        sw_hw_database_population =  database.db_input.sw_hw_database_population
        if "misc_knobs" in sw_hw_database_population.keys():
            misc_knobs = sw_hw_database_population["misc_knobs"]
            if "tech_node_SF" in misc_knobs.keys():
                tech_node = misc_knobs["tech_node_SF"]

        if not blk.type == "mem":
            print("Only memory blocks supported in CACTI")
            exit(0)

        # prime cacti
        mem_bytes = max(blk.get_area_in_bytes(), config.cacti_min_memory_size_in_bytes)
        subtype = blk.subtype
        mem_bytes = (int(mem_bytes/config.min_mem_size[subtype])+1)*config.min_mem_size[subtype] # modulo calculation

        #subtype = "sram"  # TODO: change later to sram/dram
        mem_subtype = self.FARSI_to_cacti_mem_type_converter(subtype)
        cell_type = self.FARSI_to_cacti_cell_type_converter(subtype)
        self.cacti_hndlr.set_cur_mem_type(mem_subtype)
        self.cacti_hndlr.set_cur_mem_size(mem_bytes)
        self.cacti_hndlr.set_cur_cell_type(cell_type)

        # run cacti
        try:
            cacti_area_energy_results = self.cacti_hndlr.collect_cati_data()
        except Exception as e:
            print("Using cacti, the following memory config tried and failed")
            print(self.cacti_hndlr.get_config())
            raise e

        read_energy_per_byte = float(cacti_area_energy_results['Dynamic read energy (nJ)']) * (10 ** -9) / 16
        write_energy_per_byte = float(cacti_area_energy_results['Dynamic write energy (nJ)']) * (10 ** -9) / 16
        area = float(cacti_area_energy_results['Area (mm2)']) * (10 ** -6)

        read_energy_per_byte *= tech_node["energy"]["non_gpp"]
        write_energy_per_byte *= tech_node["energy"]["non_gpp"]
        area *= tech_node["area"]["mem"]

        # log values
        self.cacti_hndlr.cacti_data_container.insert(list(zip(config.cacti_input_col_order +
                                                              config.cacti_output_col_order,
                                                              [mem_subtype, mem_bytes, read_energy_per_byte, write_energy_per_byte, area])))

        return read_energy_per_byte, write_energy_per_byte, area

    # either run or look into the cached data (from CACTI) to get energy/area data
    def collect_cacti_data(self, blk, database):

        if blk.type == "ic" :
            return 0,0,0,1
        elif blk.type == "mem":
            mem_bytes = max(blk.get_area_in_bytes(), config.cacti_min_memory_size_in_bytes) # to make sure we don't go smaller than cacti's minimum size
            mem_subtype = self.FARSI_to_cacti_mem_type_converter(blk.subtype)
            mem_bytes = (int(mem_bytes / config.min_mem_size[blk.subtype]) + 1) * config.min_mem_size[blk.subtype]  # modulo calculation
            #mem_subtype = "ram" #choose from ["main memory", "ram"]
            found_results, read_energy_per_byte, write_energy_per_byte, area = \
                self.cacti_hndlr.cacti_data_container.find(list(zip(config.cacti_input_col_order,[mem_subtype, mem_bytes])))
            if not found_results:
                read_energy_per_byte, write_energy_per_byte, area = self.run_and_collect_cacti_data(blk, database)
                #read_energy_per_byte *= tech_node["energy"]
                #write_energy_per_byte *= tech_node["energy"]
                #area *= tech_node["area"]
            area_per_byte = area/mem_bytes
            return read_energy_per_byte, write_energy_per_byte, area, area_per_byte

    # For each kernel, update the energy and power using cacti
    def cacti_update_energy_area_of_kernel(self, krnl, database):
        # iterate through block/phases, collect data and insert them up
        blk_area_dict = {}
        for blk, phase_metric in krnl.block_phase_energy_dict.items():
            # only for memory and ic
            if blk.type not in ["mem", "ic"]:
                blk_area_dict[blk] = krnl.stats.get_block_area()[blk]
                continue
            read_energy_per_byte, write_energy_per_byte, area, area_per_byte = self.collect_cacti_data(blk, database)
            for phase, metric in phase_metric.items():
                krnl.block_phase_energy_dict[blk][phase] = krnl.block_phase_read_dict[blk][
                                                               phase] * read_energy_per_byte
                krnl.block_phase_energy_dict[blk][phase] += krnl.block_phase_write_dict[blk][
                                                               phase] * write_energy_per_byte
                krnl.block_phase_area_dict[blk][phase] = area

            blk_area_dict[blk] = area

        # apply aggregates, which is iterate through every phase, scratch their values, and aggregates all the block energies
        # areas.
        krnl.stats.phase_energy_dict = krnl.aggregate_energy_of_for_every_phase()
        krnl.stats.phase_area_dict = krnl.aggregate_area_of_for_every_phase()

        """
        # for debugging; delete later
        for el in krnl.stats.get_block_area().keys():
            if el not in blk_area_dict.keys():
                print(" for debugging now delete later")
                exit(0)
        """
        krnl.stats.set_block_area(blk_area_dict)
        krnl.stats.set_stats() # do not call it on set_stats directly, as it repopoluates without cacti

        return "_"

    # For each block, get energy area
    # at the moment, only setting up area. TODO: check whether energy matters
    def cacti_update_area_of_block(self, block, database):
        if block.type not in ["mem", "ic"]:
            return
        read_energy_per_byte, write_energy_per_byte, area, area_per_byte = self.collect_cacti_data(block, database)
        block.set_area_directly(area)
        #block.update_area_energy_power_rate(energy_per_byte, area_per_byte)

    # update the design energy (after you have already updated the kernels energy)
    def cacti_update_energy_area_of_design(self):
        # resetting all first
        for soc, phase_value in self.SOC_phase_energy_dict.items():
            for phase, value in self.SOC_phase_energy_dict[soc].items():
                self.SOC_phase_energy_dict[soc][phase] = 0

        # iterate through SOCs and update
        for soc, phase_value in self.SOC_phase_energy_dict.items():
            for phase, value in self.SOC_phase_energy_dict[soc].items():
                SOC_type = soc[0]
                SOC_id = soc[1]
                for kernel in self.get_kernels():
                    if kernel.SOC_type == SOC_type and kernel.SOC_id == SOC_id:
                        if phase in kernel.stats.phase_energy_dict.keys():
                            self.SOC_phase_energy_dict[(SOC_type, SOC_id)][phase] += kernel.stats.phase_energy_dict[phase]

    def correct_power_area_with_cacti(self, database):
        # bellow dictionaries used for debugging purposes. You can delete them later
        krnl_ratio_phase = {}  # for debugging delete later

        # update in 3 stages
        # (1) fix kernel energy first
        for krnl in self.__kernels:
            krnl_ratio_phase[krnl] = self.cacti_update_energy_area_of_kernel(krnl, database)

        # (2) fix the block's area
        for block in self.get_blocks():
            self.cacti_update_area_of_block(block, database)

        # (3) update/fix the entire design accordingly
        self.cacti_update_energy_area_of_design()

    def get_hardware_graph(self):
        return self.hardware_graph

    # collect (profile) design points stats.
    def collect_dp_stats(self, database):
        self.dp_stats = DPStats(self, database)

    def get_designs_SOCs(self):
        blocks = self.get_workload_to_hardware_map().get_blocks()
        designs_SOCs = []
        for block in blocks:
            if (block.SOC_type, block.SOC_id) not in designs_SOCs:
                designs_SOCs.append((block.SOC_type, block.SOC_id))
        return designs_SOCs

    # This is used for idle power calculations
    def get_blocks(self):
        blocks = self.get_workload_to_hardware_map().get_blocks()
        return blocks

    # It is a wrapper around reset_design that includes all the necessary work to clear the stats
    # and start the simulation again (used for changing power-knobs)
    def reset_design_wrapper(self):
        # We do not want to lose this information! Since it is going to be the same
        # and we do not have any other way to retain them
        self.SOC_phase_energy_dict = defaultdict(dict)
        self.phase_latency_dict = {}
        self.dp_stats = None

    # bootstrap the design from scratch
    def reset_design(self, workload_to_hardware_map=[], workload_to_hardware_schedule=[]):
        def update_kernels(self_):
            self_.__kernels = []
            for task_to_blocks_map in self.__workload_to_hardware_map.tasks_to_blocks_map_list:
                task = task_to_blocks_map.task
                self_.__kernels.append(Kernel(self_.__workload_to_hardware_map.get_by_task(task)))#,
#                                              self_.__workload_to_hardware_schedule.get_by_task(task)))
        if workload_to_hardware_map:
            self.__workload_to_hardware_map = workload_to_hardware_map
        if workload_to_hardware_schedule:
            self.__workload_to_hardware_schedule = workload_to_hardware_schedule
        update_kernels(self)

    def get_workload_to_hardware_map(self):
        return self.__workload_to_hardware_map

    def get_workload_to_hardware_schedule(self):
        return self.__workload_to_hardware_schedule

    def get_kernels(self):
        return self.__kernels

    def get_kernel_by_task_name(self, task:Task):
        return list(filter(lambda kernel: task.name == kernel.task_name, self.get_kernels()))[0]

    def get_kernels(self):
        return self.__kernels

    def get_workload_to_hardware_map(self):
        return self.__workload_to_hardware_map


# design point statistics (stats). This class contains the profiling information for a simulated design.
# Note that the difference between system complex and SOC is that a system complex can contain multiple SOCs.
class DPStats:
    def __init__(self, sim_dp: SimDesignPoint, database):
        self.comparison_mode = "latency"  # metric to compare designs against one another
        self.dp = sim_dp  # simulated design point object
        self.__kernels = self.dp.get_kernels()  # design kernels
        self.SOC_area_dict = defaultdict(lambda: defaultdict(dict))  # area of pes
        self.SOC_area_subtype_dict = defaultdict(lambda: defaultdict(dict))  # area of pes
        self.system_complex_area_dict = defaultdict()  # system complex area values (for memory, PEs, buses)
        self.power_duration_list = defaultdict(lambda: defaultdict(dict))  # power and duration of the power list
        self.SOC_metric_dict = defaultdict(lambda: defaultdict(dict))  # dictionary containing various metrics for the SOC
        self.system_complex_metric_dict = defaultdict(lambda: defaultdict(dict))  # dictionary containing the system complex metrics
        self.database = database
        self.pipe_cluster_pathlet_phase_work_rate_dict = {}
        for pipe_cluster in self.dp.get_hardware_graph().get_pipe_clusters():
            self.pipe_cluster_pathlet_phase_work_rate_dict[pipe_cluster] = pipe_cluster.get_pathlet_phase_work_rate()

        self.pipe_cluster_pathlet_phase_latency_dict = {}
        for pipe_cluster in self.dp.get_hardware_graph().get_pipe_clusters():
            self.pipe_cluster_pathlet_phase_latency_dict[pipe_cluster] = pipe_cluster.get_pathlet_phase_latency()

        use_slack_management_estimation = config.use_slack_management_estimation
        # collect the data
        self.collect_stats(use_slack_management_estimation)

    # write the results into a file
    def dump_stats(self,  des_folder, mode="light_weight"):
        file_name = config.verification_result_file
        file_addr = os.path.join(des_folder, file_name)

        for type, id in self.dp.get_designs_SOCs():
            ic_count = len(self.dp.get_workload_to_hardware_map().get_blocks_by_type("ic"))
            mem_count = len(self.dp.get_workload_to_hardware_map().get_blocks_by_type("mem"))
            pe_count = len(self.dp.get_workload_to_hardware_map().get_blocks_by_type("pe"))
        with open(file_addr, "w+") as output:
            routing_complexity = self.dp.get_hardware_graph().get_routing_complexity()
            simple_topology = self.dp.get_hardware_graph().get_simplified_topology_code()
            blk_cnt = sum([int(el) for el in simple_topology.split("_")])
            bus_cnt = [int(el) for el in simple_topology.split("_")][0]
            mem_cnt = [int(el) for el in simple_topology.split("_")][1]
            pe_cnt = [int(el) for el in simple_topology.split("_")][2]
            task_cnt = len(list(self.dp.krnl_phase_present.keys()))
            channel_cnt = self.dp.get_hardware_graph().get_number_of_channels()

            output.write("{\n")
            output.write("\"FARSI_predicted_latency\": "+ str(max(list(self.get_system_complex_metric("latency").values()))) +",\n")
            output.write("\"FARSI_predicted_energy\": "+ str(self.get_system_complex_metric("energy")) +",\n")
            output.write("\"FARSI_predicted_power\": "+ str(self.get_system_complex_metric("power")) +",\n")
            output.write("\"FARSI_predicted_area\": "+ str(self.get_system_complex_metric("area")) +",\n")
            #output.write("\"config_code\": "+ str(ic_count) + str(mem_count) + str(pe_count)+",\n")
            #output.write("\"config_code\": "+ self.dp.get_hardware_graph().get_config_code() +",\n")
            output.write("\"simplified_topology_code\": "+ self.dp.get_hardware_graph().get_simplified_topology_code() +",\n")
            output.write("\"blk_cnt\": "+ str(blk_cnt) +",\n")
            output.write("\"pe_cnt\": "+ str(pe_cnt) +",\n")
            output.write("\"mem_cnt\": "+ str(mem_cnt) +",\n")
            output.write("\"bus_cnt\": "+ str(bus_cnt) +",\n")
            output.write("\"task_cnt\": "+ str(task_cnt) +",\n")
            output.write("\"routing_complexity\": "+ str(routing_complexity) +",\n")
            output.write("\"channel_cnt\": "+ str(channel_cnt) +",\n")
            output.write("\"FARSI simulation time\": " + str(self.dp.get_simulation_time()) + ",\n")

    # Function: profile the simulated design, collecting information about
    #           latency, power, area, and phasal behavior
    # This is called within the constructor
    def collect_stats(self, use_slack_management_estimation=False):
        for type, id in self.dp.get_designs_SOCs():
            for metric_name in config.all_metrics:
                self.set_SOC_metric_value(metric_name, type, id) # data per SoC
                self.set_system_complex_metric(metric_name) # data per System

        # estimate the behavior if slack management applied
        for type, id in self.dp.get_designs_SOCs():
            if use_slack_management_estimation:
                values_changed = self.apply_slack_management_estimation_improved(type, id)
                if values_changed:
                    for type, id in self.dp.get_designs_SOCs():
                        for metric_name in config.all_metrics:
                            self.set_system_complex_metric(metric_name)  # data per System

    # Functionality:
    #   Hot means the bottleneck or rather the
    #   most power/energy/area/performance consuming of the system. This is determined
    #   by the input argument metric.
    def get_hot_kernel_SOC(self, SOC_type, SOC_id, metric="latency", krnel_rank=0):
        kernels_on_SOC = [kernel for kernel in self.__kernels if kernel.SOC_type == SOC_type and kernel.SOC_id == SOC_id]
        sorted_kernels_hot_to_cold = sorted(kernels_on_SOC, key=lambda kernel: kernel.stats.get_metric(metric), reverse=True)
        return sorted_kernels_hot_to_cold[krnel_rank]

    #   Hot means the bottleneck or rather the
    #   most power/energy/area/performance consuming of the system. This is determined
    #   by the input argument metric.
    def get_hot_kernel_system_complex(self, metric="latency", krnel_rank=0):
        hot_krnel_list = []
        for SOC_type, SOC_id in self.get_designs_SOCs():
            hot_krnel_list.append(self.get_hot_kernel_SOC(SOC_type, SOC_id, metric, krnel_rank))
        return sorted(hot_krnel_list, key=lambda kernel: kernel.stats.get_metric(metric), reverse=True)[0]

    #   Hot means the bottleneck or rather the
    #   most power/energy/area/performance consuming of the system. This is determined
    #   by the input argument metric.
    def get_hot_block_SOC(self, SOC_type, SOC_id, metric="latency", krnel_rank=0):
        # find the hottest kernel
        hot_krnel = self.get_hot_kernel_SOC(SOC_type, SOC_id, metric, krnel_rank)
        hot_kernel_blck_bottleneck:Block = hot_krnel.stats.get_block_bottleneck(metric)
        # corresponding block bottleneck. We need this since we make a copy of the the sim_dp,
        # and hence, sim_dp and ex_dp won't be synced any more
        return hot_kernel_blck_bottleneck

    # get hot blocks of the system
    #   Hot means the bottleneck or rather the
    #   most power/energy/area/performance consuming of the system. This is determined
    #   by the input argument metric.
    # krnel_rank is the rank of the kernel to pick from once the kernels are sorted. 0 means the highest.
    # This variable is used to unstuck the heuristic when necessary (e.g., when for example the hottest kernel
    # modification is not helping the design, we move on to the second hottest)
    def get_hot_block_system_complex(self, metric="latency", krnel_rank=0):
        hot_blck_list = []
        for SOC_type, SOC_id in self.get_designs_SOCs():
            hot_blck_list.append(self.get_hot_block_SOC(SOC_type, SOC_id, metric, krnel_rank))

        return sorted(hot_blck_list, key=lambda kernel: kernel.stats.get_metric(metric), reverse=True)[0]

    # get kernels sorted based on latency
    def get_kernels_sort(self):
        sorted_kernels_hot_to_cold = sorted(self.__kernels, key=lambda kernel: kernel.stats.latency, reverse=True)
        return sorted_kernels_hot_to_cold

    # -----------------------------------------
    # Calculate profiling information per SOC
    # -----------------------------------------
    def calc_SOC_latency(self, SOC_type, SOC_id):
        kernels_on_SOC = [kernel for kernel in self.__kernels if kernel.SOC_type == SOC_type and kernel.SOC_id == SOC_id]
        workload_latency_dict = {}
        for workload, last_task in self.database.get_workloads_last_task().items():
            kernel = self.dp.get_kernel_by_task_name(self.dp.get_task_by_name(last_task))
            workload_latency_dict[workload] = kernel.stats.latency + kernel.starting_time
        return workload_latency_dict

    # calculate SOC energy
    def calc_SOC_energy(self, SOC_type, SOC_id):
        phase_total_energy = {}
        return sum(list(self.dp.SOC_phase_energy_dict[(SOC_type, SOC_id)].values()))

    # if estimate_slack_management_effect is set to true,
    # we estimate what will happen if we can introduce slacks in order to reduce power
    def calc_SOC_power(self, SOC_type, SOC_id, estimate_slack_management_effect=False):
        self.power_duration_list[SOC_type][SOC_id] = []
        sorted_listified_phase_latency_dict = sorted(self.dp.phase_latency_dict.items(), key=operator.itemgetter(0))
        sorted_latencys = [latency for phase,latency in sorted_listified_phase_latency_dict]
        sorted_phase_latency_dict = collections.OrderedDict(sorted_listified_phase_latency_dict)

        # get the energy first
        SOC_phase_energy_dict  = self.dp.SOC_phase_energy_dict[(SOC_type, SOC_id)]
        sorted_listified_phase_energy_dict = sorted(SOC_phase_energy_dict.items(), key=operator.itemgetter(0))
        sorted_phase_energy_dict = collections.OrderedDict(sorted_listified_phase_energy_dict)

        # convert to power by slicing the time with the smallest duration within which power should be
        # calculated with (PWP)
        phase_bounds_lists = slice_phases_with_PWP(sorted_phase_latency_dict)
        power_list = [] # list of power values collected based on the power collection freq
        power_duration_list = []
        for lower_bnd, upper_bnd in phase_bounds_lists:
            if sum(sorted_latencys[lower_bnd:upper_bnd])>0:
                power_this_phase = sum(list(sorted_phase_energy_dict.values())[lower_bnd:upper_bnd])/sum(sorted_latencys[lower_bnd:upper_bnd])
                power_list.append(power_this_phase)
                self.power_duration_list[SOC_type][SOC_id].append((power_this_phase, sum(sorted_latencys[lower_bnd:upper_bnd])))
            else:
                power_list.append(0)
                power_duration_list.append((0,0))

        power = max(power_list)

        return power

    # estimate what happens if we can manage slack by optimizing kernel scheduling.
    # note that this is just a estimation. Actually scheduling needs to be applied to get exact numbers.
    # note that if we apply slack, the comparison with the higher fidelity simulation
    # will be considerably hurt (since the higher fidelity simulation doesn't typically apply the slack
    # management)
    def apply_slack_management_estimation_improved(self, SOC_type, SOC_id):
        power_duration_list = self.power_duration_list[SOC_type][SOC_id]
        # relax power if possible
        total_latency = sum([duration for power, duration in power_duration_list])
        slack = self.database.get_budget("latency", "glass") - total_latency
        power_duration_recalculated = copy.deepcopy(power_duration_list)
        values_changed = False # indicating whether slack was used to modify any value
        while slack > 0 and (self.fits_budget_per_metric(SOC_type, SOC_id, "latency", 1) and
                             not self.fits_budget_per_metric(SOC_type, SOC_id, "power", 1)):
                power_duration_sorted = sorted(power_duration_recalculated, key=lambda x: x[0])
                idx = power_duration_recalculated.index(power_duration_sorted[-1])
                power, duration = power_duration_recalculated[idx]
                slack_used = min(.0005, slack)
                slack = slack - slack_used
                duration_with_slack = duration + slack_used
                power_duration_recalculated[idx] = ((power * duration) / duration_with_slack, duration_with_slack)
                values_changed = True
        power = max([power for power, duration in power_duration_recalculated])
        self.SOC_metric_dict["power"][SOC_type][SOC_id] = power
        self.SOC_metric_dict["latency"][SOC_type][SOC_id] = sum([duration for power, duration in power_duration_recalculated])
        return values_changed

    # get total area of an soc (type is not supported yet)
    def calc_SOC_area_base_on_type(self, type_, SOC_type, SOC_id):
        blocks = self.dp.get_workload_to_hardware_map().get_blocks()
        total_area= sum([block.get_area() for block in blocks if block.SOC_type == SOC_type
                         and block.SOC_id == SOC_id and block.type == type_])
        return total_area


    # get total area of an soc (type is not supported yet)
    def calc_SOC_area_base_on_subtype(self, subtype_, SOC_type, SOC_id):
        blocks = self.dp.get_workload_to_hardware_map().get_blocks()
        total_area = 0
        for block in blocks:
            if block.SOC_type == SOC_type and block.SOC_id == SOC_id and block.subtype == subtype_:
                total_area += block.get_area()
        return total_area

    # get total area of an soc
    # Variables:
    #       SOC_type:the type of SOC you need information for
    #       SOC_id: id of the SOC you are interested in
    def calc_SOC_area(self, SOC_type, SOC_id):
        blocks = self.dp.get_workload_to_hardware_map().get_blocks()
        # note: we can't use phase_area_dict for this, since:
        #   1. we double count the statically  2. if a memory is shared, we would be double counting it
        total_area= sum([block.get_area() for block in blocks if block.SOC_type == SOC_type and  block.SOC_id == SOC_id])
        return total_area

    # the cost model associated with a PE.
    # This will help us calculate the financial cost
    # of using a specific PE
    def PE_cost_model(self, task_normalized_work, block_type, model_type="linear"):
        if model_type == "linear":
            return task_normalized_work*self.database.db_input.porting_effort[block_type]
        else:
            print("this cost model is not defined")
            exit(0)

    # the cost model associated with a MEM.
    # This will help us calculate the financial cost
    # of using a specific MEM
    def MEM_cost_model(self, task_normalized_work, block, block_type, model_type="linear"):
        if model_type == "linear":
            return task_normalized_work*self.database.db_input.porting_effort[block_type]*block.get_num_of_banks()
        else:
            print("this cost model is not defined")
            exit(0)

    # the cost model associated with a IC.
    # This will help us calculate the financial cost
    # of using a specific IC.
    def IC_cost_model(self, task_normalized_work, block, block_type, model_type="linear"):
        if model_type == "linear":
            return task_normalized_work*self.database.db_input.porting_effort[block_type]
        else:
            print("this cost model is not defined")
            exit(0)

    # calculate the development cost of an SOC
    def calc_SOC_dev_cost(self, SOC_type, SOC_id):
        blocks = self.dp.get_workload_to_hardware_map().get_blocks()
        all_kernels = self.get_kernels_sort()

        # find the simplest task's work (simple = task with the least amount of work)
        krnl_work_list = []   #contains the list of works associated with different kernels (excluding dummy tasks)
        for krnl in all_kernels:
            krnl_task = krnl.get_task()
            if not krnl_task.is_task_dummy():
                krnl_work_list.append(krnl_task.get_self_task_work())
        simplest_task_work = min(krnl_work_list)

        num_of_tasks = len(all_kernels)
        dev_cost = 0
        # iterate through each block and add the cost
        for block in blocks:
            if block.type == "pe" :
                # for IPs
                if block.subtype == "ip":
                    tasks = block.get_tasks_of_block()
                    task_work = max([task.get_self_task_work() for task in tasks])  # use max incase multiple task are mapped
                    task_normalized_work = task_work/simplest_task_work
                    dev_cost += self.PE_cost_model(task_normalized_work, "ip")
                # for GPPS
                elif block.subtype == "gpp":
                    # for DSPs
                    if "G3" in block.instance_name:
                        for task in block.get_tasks_of_block():
                            task_work = task.get_self_task_work()
                            task_normalized_work = task_work/simplest_task_work
                            dev_cost += self.PE_cost_model(task_normalized_work, "dsp")
                    # for ARM
                    elif "A53" in block.instance_name or "ARM" in block.instance_name:
                        for task in block.get_tasks_of_block():
                            task_work = task.get_self_task_work()
                            task_normalized_work = task_work/simplest_task_work
                            dev_cost += self.PE_cost_model(task_normalized_work, "arm")
                    else:
                        print("cost model for this GPP is not defined")
                        exit(0)
            elif block.type == "mem":
                task_normalized_work = 1  #  treat it as the simplest task work
                dev_cost += self.MEM_cost_model(task_normalized_work, block, "mem")
            elif block.type == "ic":
                task_normalized_work = 1 #  treat it as the simplest task work
                dev_cost += self.IC_cost_model(task_normalized_work, block, "mem")
            else:
                print("cost model for ip" + block.instance_name + " is not defined")
                exit(0)

        pes = [blk for blk in blocks if blk.type == "pe"]
        mems = [blk for blk in blocks if blk.type == "mem"]
        for pe in pes:
            pe_tasks = [el.get_name() for el in pe.get_tasks_of_block()]
            for mem in mems:
                mem_tasks = [el.get_name() for el in mem.get_tasks_of_block()]
                task_share_cnt = len(pe_tasks) - len(list(set(pe_tasks) - set(mem_tasks)))
                if task_share_cnt == 0:  # this condition to avoid finding paths between vertecies, which is pretty comp intensive
                    continue
                path_length = len(self.dp.get_hardware_graph().get_path_between_two_vertecies(pe, mem))
                #path_length = len(self.dp.get_hardware_graph().get_shortest_path(pe, mem, [], []))
                effort = self.database.db_input.porting_effort["ic"]/10
                dev_cost += (path_length*task_share_cnt)*.1


        return dev_cost

    # pb_type: processing block type
    def get_SOC_s_specific_area(self, SOC_type, SOC_id, pb_type):
        assert(pb_type in ["pe", "ic", "mem"]) , "block type " + pb_type + " is not supported"
        return self.SOC_area_dict[pb_type][SOC_type][SOC_id]

    # --------
    # setters
    # --------
    def set_SOC_metric_value(self,metric_type, SOC_type, SOC_id):
        if metric_type == "area":
            self.SOC_metric_dict[metric_type][SOC_type][SOC_id] = self.calc_SOC_area(SOC_type, SOC_id)
            for block_type in ["pe", "mem", "ic"]:
                self.SOC_area_dict[block_type][SOC_type][SOC_id] = self.calc_SOC_area_base_on_type(block_type, SOC_type, SOC_id)
            for block_subtype in ["sram", "dram", "ic", "gpp", "ip"]:
                self.SOC_area_subtype_dict[block_subtype][SOC_type][SOC_id] = self.calc_SOC_area_base_on_subtype(block_subtype, SOC_type, SOC_id)

        elif metric_type == "cost":
            #self.SOC_metric_dict[metric_type][SOC_type][SOC_id] = self.calc_SOC_area(SOC_type, SOC_id)
            self.SOC_metric_dict[metric_type][SOC_type][SOC_id] = self.calc_SOC_dev_cost(SOC_type, SOC_id)
        elif metric_type == "energy":
            self.SOC_metric_dict[metric_type][SOC_type][SOC_id] =  self.calc_SOC_energy(SOC_type, SOC_id)
        elif metric_type == "power" :
            self.SOC_metric_dict[metric_type][SOC_type][SOC_id] =  self.calc_SOC_power(SOC_type, SOC_id)
        elif metric_type == "latency":
            self.SOC_metric_dict[metric_type][SOC_type][SOC_id] = self.calc_SOC_latency(SOC_type, SOC_id)
        else:
            raise Exception("metric_type:" + metric_type + " is not supported")

    # helper function to apply an operator across two dictionaries
    def operate_on_two_dic_values(self,dict1, dict2, operator):
        dict_res = {}
        for key in list(dict2.keys()) + list(dict1.keys()):
            if key in dict1.keys() and dict2.keys():
                dict_res[key] = operator(dict2[key], dict1[key])
            else:
                if key in dict1.keys():
                    dict_res[key] = dict1[key]
                elif key in dict2.keys():
                    dict_res[key] = dict2[key]
        return dict_res

    def operate_on_dicionary_values(self, dictionaries, operator):
        res = {}
        for SOCs_latency in dictionaries:
            #res = copy.deepcopy(self.operate_on_two_dic_values(res, SOCs_latency, operator))
            #gc.disable()
            res = cPickle.loads(cPickle.dumps(self.operate_on_two_dic_values(res, SOCs_latency, operator), -1))
            #gc.enable()
        return res

    # set the metric (power, area, ...) for the entire system complex
    def set_system_complex_metric(self, metric_type):
        type_id_list = self.dp.get_designs_SOCs()
        # the only spatial scenario is area
        if metric_type == "area":
            for block_type in ["pe", "mem", "ic"]:
                for type_, id_ in type_id_list:
                    self.system_complex_area_dict[block_type] = sum([self.get_SOC_area_base_on_type(block_type, type_, id_)
                                                             for type_, id_ in type_id_list])
        if metric_type in ["area", "energy", "cost"]:
            self.system_complex_metric_dict[metric_type] = sum([self.get_SOC_metric_value(metric_type, type_, id_)
                                                     for type_, id_ in type_id_list])
        elif metric_type in ["latency"]:
            self.system_complex_metric_dict[metric_type] = self.operate_on_dicionary_values([self.get_SOC_metric_value(metric_type, type_, id_)
                                                                                             for type_, id_ in type_id_list], operator.add)

            #return res
            #self.system_complex_metric_dict[metric_type] = sum([self.get_SOC_metric_value(metric_type, type_, id_)
            #                                         for type_, id_ in type_id_list])
        elif metric_type in ["power"]:
            self.system_complex_metric_dict[metric_type] = max([self.get_SOC_metric_value(metric_type, type_, id_)
                                                         for type_, id_ in type_id_list])
        else:
            raise Exception("metric_type:" + metric_type + " is not supported")

    # --------
    # getters
    # --------
    def get_SOC_metric_value(self, metric_type, SOC_type, SOC_id):
        assert(metric_type in config.all_metrics), metric_type + " not supported"
        return self.SOC_metric_dict[metric_type][SOC_type][SOC_id]

    def get_SOC_area_base_on_type(self, block_type, SOC_type, SOC_id):
        assert(block_type in ["pe", "ic", "mem"]), "block_type" + block_type + " is not supported"
        return self.SOC_area_dict[block_type][SOC_type][SOC_id]

    def get_SOC_area_base_on_subtype(self, block_subtype, SOC_type, SOC_id):
        assert(block_subtype in ["dram", "sram", "ic", "gpp", "ip"]), "block_subtype" + block_subtype + " is not supported"
        return self.SOC_area_subtype_dict[block_subtype][SOC_type][SOC_id]


    # get the simulation progress
    def get_SOC_s_latency_sim_progress(self, SOC_type, SOC_id, progress_metrics):
        kernels_on_SOC = [kernel for kernel in self.__kernels if kernel.SOC_type == SOC_type and kernel.SOC_id == SOC_id]
        kernel_metric_value = {}
        for kernel in kernels_on_SOC:
            kernel_metric_value[kernel] = []
        for metric in progress_metrics:
            for kernel in kernels_on_SOC:
                if metric == "latency":
                    kernel_metric_value[kernel].append((kernel.starting_time*10**3, kernel.stats.latency*10**3, kernel.stats.latency*10**3, "ms"))
        return kernel_metric_value

    # get the simulation progress
    def get_SOC_s_latency_sim_progress(self, SOC_type, SOC_id, metric):
        kernels_on_SOC = [kernel for kernel in self.__kernels if kernel.SOC_type == SOC_type and kernel.SOC_id == SOC_id]
        kernel_metric_value = {}
        for kernel in kernels_on_SOC:
            kernel_metric_value[kernel] = []
        for kernel in kernels_on_SOC:
            if metric == "latency":
                kernel_metric_value[kernel].append((kernel.starting_time*10**3, kernel.stats.latency*10**3, kernel.stats.latency*10**3, "ms"))
            elif metric == "bytes":
                kernel_metric_value[kernel].append((kernel.starting_time * 10 ** 3, kernel.stats.latency * 10 ** 3,
                                                    kernel.stats.latency* 10 ** 3, "bytes"))
        return kernel_metric_value


    def get_sim_progress(self, metric="latency"):
        if metric == "latency":
            return [self.get_SOC_s_latency_sim_progress(type, id, metric) for type, id in self.dp.get_designs_SOCs()]
        if metric == "bytes":
            pass


    # returns the latency associated with the phases of the system execution
    def get_phase_latency(self, SOC_type, SOC_id):
        return self.dp.phase_latency_dict

    # get utilization associated with the phases of the execution
    def get_SOC_s_sim_utilization(self, SOC_type, SOC_id):
        return self.dp.block_phase_utilization_dict

    def get_SOC_s_pipe_cluster_pathlet_phase_work_rate(self, SOC_type, SOC_id):
        return self.pipe_cluster_pathlet_phase_work_rate_dict

    def get_SOC_s_pipe_cluster_pathlet_phase_latency(self, SOC_type, SOC_id):
        return self.pipe_cluster_pathlet_phase_latency_dict

    def get_SOC_s_pipe_cluster_path_phase_latency(self, SOC_type, SOC_id):
        return self.pipe_cluster_pathlet_phase_latency_dict

    # get work associated with the phases of the execution
    def get_SOC_s_sim_work(self, SOC_type, SOC_id):
        return self.dp.block_phase_work_dict

    # get total (consider all SoCs') system metrics
    def get_system_complex_metric(self, metric_type):
        assert(metric_type in config.all_metrics), metric_type + " not supported"
        assert(not (self.system_complex_metric_dict[metric_type] == -1)), metric_type + "not calculated"
        return self.system_complex_metric_dict[metric_type]

    # check if dp_rep is meeting the budget
    def fits_budget(self, budget_coeff):
        for type, id in self.dp.get_designs_SOCs():
            for metric_name in self.database.get_budgetted_metric_names(type):
                if not self.fits_budget_for_metric(type, id, metric_name):
                    return False
        return True

    def fits_budget_per_metric(self, metric_name, budget_coeff):
        for type, id in self.dp.get_designs_SOCs():
            if not self.fits_budget_for_metric(type, id, metric_name):
                return False
        return True


    # whether the design fits the budget for the metric argument specified
    # type, and id specify the relevant parameters of the SOC
    # ignore budget_coeff for now
    def fits_budget_for_metric(self, type, id, metric_name, budget_coeff):
        return self.normalized_distance(type, id, metric_name) < .001

    def __lt__(self, other):
        comp_list = []
        for metric in config.objectives:
            comp_list.append(self.get_system_complex_metric(metric) < other.get_system_complex_metric(metric))
        return all(comp_list)

    def __gt__(self, other):
        comp_list = []
        for metric in config.objectives:
            comp_list.append(self.get_system_complex_metric(metric) > other.get_system_complex_metric(metric))
        return all(comp_list)