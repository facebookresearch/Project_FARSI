#Copyright (c) Facebook, Inc. and its affiliates.
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.

import csv
import os
import matplotlib.pyplot as plt

# the function to plot the frequency of all comm_comp in the pie chart
def plotCommCompAll(dirName, fileName):
    with open(dirName + fileName + "/result_summary/FARSI_simple_run_0_1_all_reults.csv", newline='') as csvfile:
        resultReader = csv.reader(csvfile, delimiter=',', quotechar='|')

        commNum = 0
        compNum = 0

        for i, row in enumerate(resultReader):
            # print('"' + row[1] + '"\t"' + row[4] + '"\t"' + row[5] + '"\t"' + row[10] + '"\t"' + row[12] + '"\t"' + row[13] + '"\t"' + row[22] + '"\t"' + row[23] + '"\t"' + row[28] + '"\t' + row[29] + '"\t"' + row[30] + '"')
            if row[22] != "True":
                continue

            if i > 1:
                if row[28] == "comm":
                    commNum += 1
                elif row[28] == "comp":
                    compNum += 1
                else:
                    raise Exception("comm_comp is not giving comm or comp! The new type: " + row[28])
            
        plt.pie([commNum, compNum], labels = ["comm", "comp"])
        plt.title("comm_comp: Frequency")
        plt.savefig(dirName + fileName + "/comm-compFreq-" + fileName + ".png")
        plt.show()

# the function to plot the frequency of all high level optimizations in the pie chart
def plothighLevelOptAll(dirName, fileName):
    with open(dirName + fileName + "/result_summary/FARSI_simple_run_0_1_all_reults.csv", newline='') as csvfile:
        resultReader = csv.reader(csvfile, delimiter=',', quotechar='|')

        topoNum = 0
        tunNum = 0
        mapNum = 0
        idenOptNum = 0

        for i, row in enumerate(resultReader):
            # print('"' + row[1] + '"\t"' + row[4] + '"\t"' + row[5] + '"\t"' + row[10] + '"\t"' + row[12] + '"\t"' + row[13] + '"\t"' + row[22] + '"\t"' + row[23] + '"\t"' + row[28] + '"\t' + row[29] + '"\t"' + row[30] + '"')
            if row[22] != "True":
                continue

            if i > 1:
                if row[29] == "topology":
                    topoNum += 1
                elif row[29] == "tunning":
                    tunNum += 1
                elif row[29] == "mapping":
                    mapNum += 1
                elif row[29] == "identity":
                    idenOptNum += 1
                else:
                    raise Exception("high_level_optimization is not giving topology or tunning or mapping! The new type: " + row[29])
        
        plt.pie([topoNum, tunNum, mapNum, idenOptNum], labels = ["topology", "tunning", "mapping", "identity"])
        plt.title("High Level Optimization: Frequency")
        plt.savefig(dirName + fileName + "/highLevelOpt-" + fileName + ".png")
        plt.show()

# the function to plot the frequency of all architectural variables to improve in the pie chart
def plotArchVarImpAll(dirName, fileName):
    with open(dirName + fileName + "/result_summary/FARSI_simple_run_0_1_all_reults.csv", newline='') as csvfile:
        resultReader = csv.reader(csvfile, delimiter=',', quotechar='|')

        parazNum = 0
        custNum = 0
        idenImpNum = 0

        for i, row in enumerate(resultReader):
            # print('"' + row[1] + '"\t"' + row[4] + '"\t"' + row[5] + '"\t"' + row[10] + '"\t"' + row[12] + '"\t"' + row[13] + '"\t"' + row[22] + '"\t"' + row[23] + '"\t"' + row[28] + '"\t' + row[29] + '"\t"' + row[30] + '"')
            if row[22] != "True":
                continue

            if i > 1:
                if row[30] == "parallelization":
                    parazNum += 1
                elif row[30] == "customization":
                    custNum += 1
                elif row[30] == "identity":
                    idenImpNum += 1
                else:
                    raise Exception("architectural_variable_to_improve is not parallelization or parallelism or customization! The new type: " + row[30])

        plt.pie([parazNum, custNum, idenImpNum], labels = ["parallelization", "customization", "identity"])
        plt.title("Architectural Variables to Improve: Frequency")
        plt.savefig(dirName + fileName + "/archVarImp-" + fileName + ".png")
        plt.show()

# the function to plot simulation time vs. system block count
def plotSimTimeVSblk(dirName, fileName):
    with open(dirName + fileName + "/result_summary/FARSI_simple_run_0_1_all_reults.csv", newline='') as csvfile:
        resultReader = csv.reader(csvfile, delimiter=',', quotechar='|')

        sysBlkCount = []
        simTime = []

        for i, row in enumerate(resultReader):
            # print('"' + row[1] + '"\t"' + row[4] + '"\t"' + row[5] + '"\t"' + row[10] + '"\t"' + row[12] + '"\t"' + row[13] + '"\t"' + row[22] + '"\t"' + row[23] + '"\t"' + row[28] + '"\t' + row[29] + '"\t"' + row[30] + '"')
            if row[22] != "True":
                continue

            if i > 1:
                sysBlkCount.append(int(row[13]))
                simTime.append(float(row[4]))

        plt.plot(sysBlkCount, simTime)
        plt.xlabel("System Block Count")
        plt.ylabel("Simulation Time")
        plt.title("Simulation Time vs. Sytem Block Count")
        plt.savefig(dirName + fileName + "/simTimeVSblk-" + fileName + ".png")
        plt.show()

# the function to plot move generation time vs. system block count
def plotMoveGenTimeVSblk(dirName, fileName):
    with open(dirName + fileName + "/result_summary/FARSI_simple_run_0_1_all_reults.csv", newline='') as csvfile:
        resultReader = csv.reader(csvfile, delimiter=',', quotechar='|')

        sysBlkCount = []
        moveGenTime = []

        for i, row in enumerate(resultReader):
            # print('"' + row[1] + '"\t"' + row[4] + '"\t"' + row[5] + '"\t"' + row[10] + '"\t"' + row[12] + '"\t"' + row[13] + '"\t"' + row[22] + '"\t"' + row[23] + '"\t"' + row[28] + '"\t' + row[29] + '"\t"' + row[30] + '"')
            if row[22] != "True":
                continue

            if i > 1:
                sysBlkCount.append(int(row[13]))
                moveGenTime.append(float(row[5]))
        
        plt.plot(sysBlkCount, moveGenTime)
        plt.xlabel("System Block Count")
        plt.ylabel("Move Generation Time")
        plt.title("Move Generation Time vs. System Block Count")
        plt.savefig(dirName + fileName + "/moveGenTimeVSblk-" + fileName + ".png")
        plt.show()

# the function to plot distance to goal vs. iteration x depth
def plotDistToGoalVSitr(dirName, fileName):
    with open(dirName + fileName + "/result_summary/FARSI_simple_run_0_1_all_reults.csv", newline='') as csvfile:
        resultReader = csv.reader(csvfile, delimiter=',', quotechar='|')

        itr = []
        distToGoal = []

        for i, row in enumerate(resultReader):
            # print('"' + row[1] + '"\t"' + row[4] + '"\t"' + row[5] + '"\t"' + row[10] + '"\t"' + row[12] + '"\t"' + row[13] + '"\t"' + row[22] + '"\t"' + row[23] + '"\t"' + row[28] + '"\t' + row[29] + '"\t"' + row[30] + '"')
            if row[22] != "True":
                continue

            if i > 1:
                itr.append(int(row[1]))
                distToGoal.append(float(row[10]))
        
        plt.plot(itr, distToGoal)
        plt.xlabel("Iteration and Depth Count")
        plt.ylabel("Distance to Goal")
        plt.title("Distance to Goal vs. Iteration and Depth Count")
        plt.savefig(dirName + fileName + "/distToGoalVSitr-" + fileName + ".png")
        plt.show()

# the function to plot distance to goal vs. iteration x depth
def plotRefDistToGoalVSitr(dirName, fileName):
    with open(dirName + fileName + "/result_summary/FARSI_simple_run_0_1_all_reults.csv", newline='') as csvfile:
        resultReader = csv.reader(csvfile, delimiter=',', quotechar='|')

        itr = []
        refDistToGoal = []

        for i, row in enumerate(resultReader):
            # print('"' + row[1] + '"\t"' + row[4] + '"\t"' + row[5] + '"\t"' + row[10] + '"\t"' + row[12] + '"\t"' + row[13] + '"\t"' + row[22] + '"\t"' + row[23] + '"\t"' + row[28] + '"\t' + row[29] + '"\t"' + row[30] + '"')
            if row[22] != "True":
                continue

            if i > 1:
                itr.append(int(row[1]))
                refDistToGoal.append(float(row[12]))

        plt.plot(itr, refDistToGoal)
        plt.xlabel("Iteration and Depth Count")
        plt.ylabel("Reference Design Distance to Goal")
        plt.title("Reference Design Distance to Goal vs. Iteration and Depth Count")
        plt.savefig(dirName + fileName + "/refDistToGoalVSitr-" + fileName + ".png")
        plt.show()


# the main function. comment out the plots if you do not need them
if __name__ == "__main__":
    # change the directory name and the result folder name accordingly. the directory name is the place for all your results
    dirName = "/home/yingj4/Desktop/Project_FARSI/data_collection/data/simple_run/"
    fileName = "07-13_14-06_59"

    plotCommCompAll(dirName, fileName)
    plothighLevelOptAll(dirName, fileName)
    plotArchVarImpAll(dirName, fileName)
    plotSimTimeVSblk(dirName, fileName)
    plotMoveGenTimeVSblk(dirName, fileName)
    plotDistToGoalVSitr(dirName, fileName)
    plotRefDistToGoalVSitr(dirName, fileName)
