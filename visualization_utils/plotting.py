#Copyright (c) Facebook, Inc. and its affiliates.
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.

import csv
import os
import matplotlib.pyplot as plt
import pandas as pd

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

# the function to plot distance to goal vs. iteration x depth
def plotSimTimeVSmoveNameZoneDist(dirName, fileName):
    with open(dirName + fileName + "/result_summary/FARSI_simple_run_0_1_all_reults.csv", newline='') as csvfile:
        resultReader = csv.reader(csvfile, delimiter=',', quotechar='|')

        splitSwapSimTimeZone0 = 0
        splitSimTimeZone0 = 0
        migrateSimTimeZone0 = 0
        swapSimTimeZone0 = 0
        idenSimTimeZone0 = 0
        splitSwapSimTimeZone1 = 0
        splitSimTimeZone1 = 0
        migrateSimTimeZone1 = 0
        swapSimTimeZone1 = 0
        idenSimTimeZone1 = 0
        splitSwapSimTimeZone2 = 0
        splitSimTimeZone2 = 0
        migrateSimTimeZone2 = 0
        swapSimTimeZone2 = 0
        idenSimTimeZone2 = 0
        splitSwapSimTimeZone3 = 0
        splitSimTimeZone3 = 0
        migrateSimTimeZone3 = 0
        swapSimTimeZone3 = 0
        idenSimTimeZone3 = 0

        maxDist = 0

        for i, row in enumerate(resultReader):
            # print('"' + row[1] + '"\t"' + row[4] + '"\t"' + row[5] + '"\t"' + row[10] + '"\t"' + row[12] + '"\t"' + row[13] + '"\t"' + row[22] + '"\t"' + row[23] + '"\t"' + row[28] + '"\t' + row[29] + '"\t"' + row[30] + '"')
            if row[22] != "True":
                continue

            if i == 2:
                maxDist = float(row[10])

            if i > 1:
                if row[23] == "split_swap" and float(row[10]) > maxDist * 3 // 4:
                    splitSwapSimTimeZone0 += float(row[4])
                elif row[23] == "split" and float(row[10]) > maxDist * 3 // 4:
                    splitSimTimeZone0 += float(row[4])
                elif row[23] == "migrate" and float(row[10]) > maxDist * 3 // 4:
                    migrateSimTimeZone0 += float(row[4])
                elif row[23] == "swap" and float(row[10]) > maxDist * 3 // 4:
                    swapSimTimeZone0 += float(row[4])
                elif row[23] == "identity" and float(row[10]) > maxDist * 3 // 4:
                    idenSimTimeZone0 += float(row[4])
                elif row[23] == "split_swap" and float(row[10]) <= maxDist * 3 // 4 and float(row[10]) > maxDist // 2:
                    splitSwapSimTimeZone1 += float(row[4])
                elif row[23] == "split" and float(row[10]) <= maxDist * 3 // 4 and float(row[10]) > maxDist // 2:
                    splitSimTimeZone1 += float(row[4])
                elif row[23] == "migrate" and float(row[10]) <= maxDist * 3 // 4 and float(row[10]) > maxDist // 2:
                    migrateSimTimeZone1 += float(row[4])
                elif row[23] == "swap" and float(row[10]) <= maxDist * 3 // 4 and float(row[10]) > maxDist // 2:
                    swapSimTimeZone1 += float(row[4])
                elif row[23] == "identity" and float(row[10]) <= maxDist * 3 // 4 and float(row[10]) > maxDist // 2:
                    idenSimTimeZone1 += float(row[4])
                elif row[23] == "split_swap" and float(row[10]) <= maxDist // 2 and float(row[10]) > maxDist // 4:
                    splitSwapSimTimeZone2 += float(row[4])
                elif row[23] == "split" and float(row[10]) <= maxDist // 2 and float(row[10]) > maxDist // 4:
                    splitSimTimeZone2 += float(row[4])
                elif row[23] == "migrate" and float(row[10]) <= maxDist // 2 and float(row[10]) > maxDist // 4:
                    migrateSimTimeZone2 += float(row[4])
                elif row[23] == "swap" and float(row[10]) <= maxDist // 2 and float(row[10]) > maxDist // 4:
                    swapSimTimeZone2 += float(row[4])
                elif row[23] == "identity" and float(row[10]) <= maxDist // 2 and float(row[10]) > maxDist // 4:
                    idenSimTimeZone2 += float(row[4])
                elif row[23] == "split_swap" and float(row[10]) <= maxDist // 4:
                    splitSwapSimTimeZone3 += float(row[4])
                elif row[23] == "split" and float(row[10]) <= maxDist // 4:
                    splitSimTimeZone3 += float(row[4])
                elif row[23] == "migrate" and float(row[10]) <= maxDist // 4:
                    migrateSimTimeZone3 += float(row[4])
                elif row[23] == "swap" and float(row[10]) <= maxDist // 4:
                    swapSimTimeZone3 += float(row[4])
                elif row[23] == "identity" and float(row[10]) <= maxDist // 4:
                    idenSimTimeZone3 += float(row[4])
                else:
                    raise Exception("move name is not split_swap or split or migrate or swap or identity! The new type: " + row[23])
        
        plotdata = pd.DataFrame({
            "split_swap":[splitSwapSimTimeZone0, splitSwapSimTimeZone1, splitSwapSimTimeZone2, splitSwapSimTimeZone3],
            "split":[splitSimTimeZone0, splitSimTimeZone1, splitSimTimeZone2, splitSimTimeZone3],
            "migrate":[migrateSimTimeZone0, migrateSimTimeZone1, migrateSimTimeZone2, migrateSimTimeZone3],
            "swap":[swapSimTimeZone0, swapSimTimeZone1, swapSimTimeZone2, swapSimTimeZone3],
            "identity":[idenSimTimeZone0, idenSimTimeZone1, idenSimTimeZone2, idenSimTimeZone3]
        }, index = ["1~3/4", "3/4~1/2", "1/2~1/4", "1/4~0"]
        )
        plotdata.plot(kind = 'bar', stacked = True)
        plt.xlabel("Zone decided by the max distance to goal")
        plt.ylabel("Simulation Time")
        plt.title("Simulation Time in Each Zone based on Move Name")
        plt.savefig(dirName + fileName + "/simTimeVSmoveNameZoneDist-" + fileName + ".png")
        plt.show()

# the function to plot distance to goal vs. iteration x depth
def plotSimTimeVScommCompZoneDist(dirName, fileName):
    with open(dirName + fileName + "/result_summary/FARSI_simple_run_0_1_all_reults.csv", newline='') as csvfile:
        resultReader = csv.reader(csvfile, delimiter=',', quotechar='|')

        commSimTimeZone0 = 0
        commSimTimeZone1 = 0
        commSimTimeZone2 = 0
        commSimTimeZone3 = 0
        compSimTimeZone0 = 0
        compSimTimeZone1 = 0
        compSimTimeZone2 = 0
        compSimTimeZone3 = 0

        maxDist = 0

        for i, row in enumerate(resultReader):
            # print('"' + row[1] + '"\t"' + row[4] + '"\t"' + row[5] + '"\t"' + row[10] + '"\t"' + row[12] + '"\t"' + row[13] + '"\t"' + row[22] + '"\t"' + row[23] + '"\t"' + row[28] + '"\t' + row[29] + '"\t"' + row[30] + '"')
            if row[22] != "True":
                continue

            if i == 2:
                maxDist = float(row[10])

            if i > 1:
                if row[28] == "comm" and float(row[10]) > maxDist * 3 // 4:
                    commSimTimeZone0 += float(row[4])
                elif row[28] == "comp" and float(row[10]) > maxDist * 3 // 4:
                    compSimTimeZone0 += float(row[4])
                elif row[28] == "comm" and float(row[10]) <= maxDist * 3 // 4 and float(row[10]) > maxDist // 2:
                    commSimTimeZone1 += float(row[4])
                elif row[28] == "comp" and float(row[10]) <= maxDist * 3 // 4 and float(row[10]) > maxDist // 2:
                    compSimTimeZone1 += float(row[4])
                elif row[28] == "comm" and float(row[10]) <= maxDist // 2 and float(row[10]) > maxDist // 4:
                    commSimTimeZone2 += float(row[4])
                elif row[28] == "comp" and float(row[10]) <= maxDist // 2 and float(row[10]) > maxDist // 4:
                    compSimTimeZone2 += float(row[4])
                elif row[28] == "comm" and float(row[10]) <= maxDist // 4:
                    commSimTimeZone3 += float(row[4])
                elif row[28] == "comp" and float(row[10]) <= maxDist // 4:
                    compSimTimeZone3 += float(row[4])
                else:
                    raise Exception("comm_comp is not giving comm or comp! The new type: " + row[28])
        
        plotdata = pd.DataFrame({
            "comm":[commSimTimeZone0, commSimTimeZone1, commSimTimeZone2, commSimTimeZone3],
            "comp":[compSimTimeZone0, compSimTimeZone1, compSimTimeZone2, compSimTimeZone3]
        }, index = ["1~3/4", "3/4~1/2", "1/2~1/4", "1/4~0"]
        )
        plotdata.plot(kind = 'bar', stacked = True)
        plt.xlabel("Zone decided by the max distance to goal")
        plt.ylabel("Simulation Time")
        plt.title("Simulation Time in Each Zone based on comm_comp")
        plt.savefig(dirName + fileName + "/simTimeVScommCompZoneDist-" + fileName + ".png")
        plt.show()

# the main function. comment out the plots if you do not need them
if __name__ == "__main__":
    # change the directory name and the result folder name accordingly. the directory name is the place for all your results
    dirName = "/home/yingj4/Desktop/Project_FARSI/data_collection/data/simple_run/"
    fileName = "07-13_14-06_59"

    # plotCommCompAll(dirName, fileName)
    # plothighLevelOptAll(dirName, fileName)
    # plotArchVarImpAll(dirName, fileName)
    # plotSimTimeVSblk(dirName, fileName)
    # plotMoveGenTimeVSblk(dirName, fileName)
    # plotDistToGoalVSitr(dirName, fileName)
    # plotRefDistToGoalVSitr(dirName, fileName)
    plotSimTimeVSmoveNameZoneDist(dirName, fileName)
    plotSimTimeVScommCompZoneDist(dirName, fileName)
