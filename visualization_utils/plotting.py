#Copyright (c) Facebook, Inc. and its affiliates.
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.

import csv
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# the function to get the column information of the given category
def columnNum(dirName, fileName, cate):
    with open(dirName + fileName + "/result_summary/FARSI_simple_run_0_1_all_reults.csv", newline='') as csvfile:
        resultReader = csv.reader(csvfile, delimiter=',', quotechar='|')

        for i, row in enumerate(resultReader):
            if i == 0:
                for j in range(0, len(row)):
                    if row[j] == cate:
                        return j
                raise Exception("No such category in the list! Check the name: " + cate)
            break

# the function to plot the frequency of all comm_comp in the pie chart
def plotCommCompAll(dirName, fileName, colNum, trueNum):
    with open(dirName + fileName + "/result_summary/FARSI_simple_run_0_1_all_reults.csv", newline='') as csvfile:
        resultReader = csv.reader(csvfile, delimiter=',', quotechar='|')

        commNum = 0
        compNum = 0

        for i, row in enumerate(resultReader):
            if row[trueNum] != "True":
                continue

            if i > 1:
                if row[colNum] == "comm":
                    commNum += 1
                elif row[colNum] == "comp":
                    compNum += 1
                else:
                    raise Exception("comm_comp is not giving comm or comp! The new type: " + row[colNum])

        plt.figure()
        plt.pie([commNum, compNum], labels = ["comm", "comp"])
        plt.title("comm_comp: Frequency")
        plt.savefig(dirName + fileName + "/comm-compFreq-" + fileName + ".png")
        # plt.show()
        plt.close('all')

# the function to plot the frequency of all high level optimizations in the pie chart
def plothighLevelOptAll(dirName, fileName, colNum, trueNum):
    with open(dirName + fileName + "/result_summary/FARSI_simple_run_0_1_all_reults.csv", newline='') as csvfile:
        resultReader = csv.reader(csvfile, delimiter=',', quotechar='|')

        topoNum = 0
        tunNum = 0
        mapNum = 0
        idenOptNum = 0

        for i, row in enumerate(resultReader):
            if row[trueNum] != "True":
                continue

            if i > 1:
                if row[colNum] == "topology":
                    topoNum += 1
                elif row[colNum] == "tunning":
                    tunNum += 1
                elif row[colNum] == "mapping":
                    mapNum += 1
                elif row[colNum] == "identity":
                    idenOptNum += 1
                else:
                    raise Exception("optimization name is not giving topology or tunning or mapping or identity! The new type: " + row[colNum])
        
        plt.figure()
        plt.pie([topoNum, tunNum, mapNum, idenOptNum], labels = ["topology", "tunning", "mapping", "identity"])
        plt.title("High Level Optimization: Frequency")
        plt.savefig(dirName + fileName + "/highLevelOpt-" + fileName + ".png")
        # plt.show()
        plt.close('all')

# the function to plot the frequency of all architectural variables to improve in the pie chart
def plotArchVarImpAll(dirName, fileName, colNum, trueNum):
    with open(dirName + fileName + "/result_summary/FARSI_simple_run_0_1_all_reults.csv", newline='') as csvfile:
        resultReader = csv.reader(csvfile, delimiter=',', quotechar='|')

        parazNum = 0
        custNum = 0
        localNum = 0
        idenImpNum = 0

        for i, row in enumerate(resultReader):
            if row[trueNum] != "True":
                continue

            if i > 1:
                if row[colNum] == "parallelization":
                    parazNum += 1
                elif row[colNum] == "customization":
                    custNum += 1
                elif row[colNum] == "locality":
                    localNum += 1
                elif row[colNum] == "identity":
                    idenImpNum += 1                
                else:
                    raise Exception("architectural principle is not parallelization or customization or locality or identity! The new type: " + row[colNum])

        plt.figure()
        plt.pie([parazNum, custNum, localNum, idenImpNum], labels = ["parallelization", "customization", "locality", "identity"])
        plt.title("Architectural Principle: Frequency")
        plt.savefig(dirName + fileName + "/archVarImp-" + fileName + ".png")
        # plt.show()
        plt.close('all')

# the function to plot simulation time vs. system block count
def plotSimTimeVSblk(dirName, fileName, blkColNum, simColNum, trueNum):
    with open(dirName + fileName + "/result_summary/FARSI_simple_run_0_1_all_reults.csv", newline='') as csvfile:
        resultReader = csv.reader(csvfile, delimiter=',', quotechar='|')

        sysBlkCount = []
        simTime = []

        for i, row in enumerate(resultReader):
            if row[trueNum] != "True":
                continue

            if i > 1:
                sysBlkCount.append(int(row[blkColNum]))
                simTime.append(float(row[simColNum]))

        plt.figure()
        plt.plot(sysBlkCount, simTime)
        plt.xlabel("System Block Count")
        plt.ylabel("Simulation Time")
        plt.title("Simulation Time vs. Sytem Block Count")
        plt.savefig(dirName + fileName + "/simTimeVSblk-" + fileName + ".png")
        # plt.show()
        plt.close('all')

# the function to plot move generation time vs. system block count
def plotMoveGenTimeVSblk(dirName, fileName, blkColNum, movColNum, trueNum):
    with open(dirName + fileName + "/result_summary/FARSI_simple_run_0_1_all_reults.csv", newline='') as csvfile:
        resultReader = csv.reader(csvfile, delimiter=',', quotechar='|')

        sysBlkCount = []
        moveGenTime = []

        for i, row in enumerate(resultReader):
            if row[trueNum] != "True":
                continue

            if i > 1:
                sysBlkCount.append(int(row[blkColNum]))
                moveGenTime.append(float(row[movColNum]))
        
        plt.figure()
        plt.plot(sysBlkCount, moveGenTime)
        plt.xlabel("System Block Count")
        plt.ylabel("Move Generation Time")
        plt.title("Move Generation Time vs. System Block Count")
        plt.savefig(dirName + fileName + "/moveGenTimeVSblk-" + fileName + ".png")
        # plt.show()
        plt.close('all')

# the function to plot distance to goal vs. iteration x depth
def plotDistToGoalVSitr(dirName, fileName, itrColNum, distColNum, trueNum):
    with open(dirName + fileName + "/result_summary/FARSI_simple_run_0_1_all_reults.csv", newline='') as csvfile:
        resultReader = csv.reader(csvfile, delimiter=',', quotechar='|')

        itr = []
        distToGoal = []

        for i, row in enumerate(resultReader):
            if row[trueNum] != "True":
                continue

            if i > 1:
                itr.append(int(row[itrColNum]))
                distToGoal.append(float(row[distColNum]))
        
        plt.figure()
        plt.plot(itr, distToGoal)
        plt.xlabel("Iteration and Depth Count")
        plt.ylabel("Distance to Goal")
        plt.title("Distance to Goal vs. Iteration and Depth Count")
        plt.savefig(dirName + fileName + "/distToGoalVSitr-" + fileName + ".png")
        # plt.show()
        plt.close('all')

# the function to plot distance to goal vs. iteration x depth
def plotRefDistToGoalVSitr(dirName, fileName, itrColNum, refDistColNum, trueNum):
    with open(dirName + fileName + "/result_summary/FARSI_simple_run_0_1_all_reults.csv", newline='') as csvfile:
        resultReader = csv.reader(csvfile, delimiter=',', quotechar='|')

        itr = []
        refDistToGoal = []

        for i, row in enumerate(resultReader):
            if row[trueNum] != "True":
                continue

            if i > 1:
                itr.append(int(row[itrColNum]))
                refDistToGoal.append(float(row[refDistColNum]))

        plt.figure()
        plt.plot(itr, refDistToGoal)
        plt.xlabel("Iteration and Depth Count")
        plt.ylabel("Reference Design Distance to Goal")
        plt.title("Reference Design Distance to Goal vs. Iteration and Depth Count")
        plt.savefig(dirName + fileName + "/refDistToGoalVSitr-" + fileName + ".png")
        # plt.show()
        plt.close('all')

# the function to do the zonal partitioning
def zonalPartition(comparedValue, zoneNum, maxValue):
    unit = maxValue / zoneNum

    if comparedValue > maxValue:
        return zoneNum - 1
    
    if comparedValue < 0:
        return 0

    for i in range(0, zoneNum):
        if comparedValue <= unit * (i + 1):
            return i

    raise Exception("zonalPartition is fed by a strange value! maxValue: " + str(maxValue) + "; comparedValue: " + str(comparedValue))

# the function to plot simulation time vs. move name in a zonal format
def plotSimTimeVSmoveNameZoneDist(dirName, fileName, zoneNum, moveColNum, distColNum, simColNum, trueNum):
    with open(dirName + fileName + "/result_summary/FARSI_simple_run_0_1_all_reults.csv", newline='') as csvfile:
        resultReader = csv.reader(csvfile, delimiter=',', quotechar='|')

        splitSwapSim = np.zeros(zoneNum, dtype = float)
        splitSim = np.zeros(zoneNum, dtype = float)
        migrateSim = np.zeros(zoneNum, dtype = float)
        swapSim = np.zeros(zoneNum, dtype = float)
        tranSim = np.zeros(zoneNum, dtype = float)
        routeSim = np.zeros(zoneNum, dtype = float)
        identitySim = np.zeros(zoneNum, dtype = float)

        maxDist = 0

        index = []
        for i in range(0, zoneNum):
            index.append(i)

        for i, row in enumerate(resultReader):
            # print('"' + row[trueNum] + '"\t"' + row[moveColNum] + '"\t"' + row[distColNum] + '"\t"' + row[simColNum] + '"')
            if row[trueNum] != "True":
                continue

            if i == 2:
                maxDist = float(row[distColNum])

            if i > 1:
                if row[moveColNum] == "split_swap":
                    splitSwapSim[zonalPartition(float(row[distColNum]), zoneNum, maxDist)] += float(row[simColNum])
                elif row[moveColNum] == "split":
                    splitSim[zonalPartition(float(row[distColNum]), zoneNum, maxDist)] += float(row[simColNum])
                elif row[moveColNum] == "migrate":
                    migrateSim[zonalPartition(float(row[distColNum]), zoneNum, maxDist)] += float(row[simColNum])
                elif row[moveColNum] == "swap":
                    swapSim[zonalPartition(float(row[distColNum]), zoneNum, maxDist)] += float(row[simColNum])
                elif row[moveColNum] == "transfer":
                    tranSim[zonalPartition(float(row[distColNum]), zoneNum, maxDist)] += float(row[simColNum])
                elif row[moveColNum] == "routing":
                    routeSim[zonalPartition(float(row[distColNum]), zoneNum, maxDist)] += float(row[simColNum])
                elif row[moveColNum] == "identity":
                    identitySim[zonalPartition(float(row[distColNum]), zoneNum, maxDist)] += float(row[simColNum])
                else:
                    raise Exception("move name is not split_swap or split or migrate or swap or transfer or routing or identity! The new type: " + row[moveColNum])
        
        plt.figure()
        plotdata = pd.DataFrame({
            "split_swap":splitSwapSim,
            "split":splitSim,
            "migrate":migrateSim,
            "swap":swapSim,
            "transfer":tranSim,
            "routing":routeSim,
            "identity":identitySim
        }, index = index
        )
        plotdata.plot(kind = 'bar', stacked = True)
        plt.xlabel("Zone decided by the max distance to goal")
        plt.ylabel("Simulation Time")
        plt.title("Simulation Time in Each Zone based on Move Name")
        plt.savefig(dirName + fileName + "/simTimeVSmoveNameZoneDist-" + fileName + ".png")
        # plt.show()
        plt.close('all')

# the function to plot move generation time vs. move name in a zonal format
def plotMovGenTimeVSmoveNameZoneDist(dirName, fileName, zoneNum, moveColNum, distColNum, movGenColNum, trueNum):
    with open(dirName + fileName + "/result_summary/FARSI_simple_run_0_1_all_reults.csv", newline='') as csvfile:
        resultReader = csv.reader(csvfile, delimiter=',', quotechar='|')

        splitSwapMov = np.zeros(zoneNum, dtype = float)
        splitMov = np.zeros(zoneNum, dtype = float)
        migrateMov = np.zeros(zoneNum, dtype = float)
        swapMov = np.zeros(zoneNum, dtype = float)
        tranMov = np.zeros(zoneNum, dtype = float)
        routeMov = np.zeros(zoneNum, dtype = float)
        identityMov = np.zeros(zoneNum, dtype = float)

        maxDist = 0

        index = []
        for i in range(0, zoneNum):
            index.append(i)

        for i, row in enumerate(resultReader):
            # print('"' + row[trueNum] + '"\t"' + row[moveColNum] + '"\t"' + row[distColNum] + '"\t"' + row[movGenColNum] + '"')
            if row[trueNum] != "True":
                continue

            if i == 2:
                maxDist = float(row[distColNum])

            if i > 1:
                if row[moveColNum] == "split_swap":
                    splitSwapMov[zonalPartition(float(row[distColNum]), zoneNum, maxDist)] += float(row[movGenColNum])
                elif row[moveColNum] == "split":
                    splitMov[zonalPartition(float(row[distColNum]), zoneNum, maxDist)] += float(row[movGenColNum])
                elif row[moveColNum] == "migrate":
                    migrateMov[zonalPartition(float(row[distColNum]), zoneNum, maxDist)] += float(row[movGenColNum])
                elif row[moveColNum] == "swap":
                    swapMov[zonalPartition(float(row[distColNum]), zoneNum, maxDist)] += float(row[movGenColNum])
                elif row[moveColNum] == "transfer":
                    tranMov[zonalPartition(float(row[distColNum]), zoneNum, maxDist)] += float(row[movGenColNum])
                elif row[moveColNum] == "routing":
                    routeMov[zonalPartition(float(row[distColNum]), zoneNum, maxDist)] += float(row[movGenColNum])
                elif row[moveColNum] == "identity":
                    identityMov[zonalPartition(float(row[distColNum]), zoneNum, maxDist)] += float(row[movGenColNum])
                else:
                    raise Exception("move name is not split_swap or split or migrate or swap or transfer of routing or identity! The new type: " + row[moveColNum])
        
        plt.figure()
        plotdata = pd.DataFrame({
            "split_swap":splitSwapMov,
            "split":splitMov,
            "migrate":migrateMov,
            "swap":swapMov,
            "transfer":tranMov,
            "routing":routeMov,
            "identity":identityMov
        }, index = index
        )
        plotdata.plot(kind = 'bar', stacked = True)
        plt.xlabel("Zone decided by the max distance to goal")
        plt.ylabel("Move Generation Time")
        plt.title("Move Generation Time in Each Zone based on Move Name")
        plt.savefig(dirName + fileName + "/movGenTimeVSmoveNameZoneDist-" + fileName + ".png")
        # plt.show()
        plt.close('all')

# the function to plot simulation time vs. comm_comp in a zonal format
def plotSimTimeVScommCompZoneDist(dirName, fileName, zoneNum, commcompColNum, distColNum, simColNum, trueNum):
    with open(dirName + fileName + "/result_summary/FARSI_simple_run_0_1_all_reults.csv", newline='') as csvfile:
        resultReader = csv.reader(csvfile, delimiter=',', quotechar='|')

        commSim = np.zeros(zoneNum, dtype = float)
        compSim = np.zeros(zoneNum, dtype = float)

        maxDist = 0
        index = []
        for i in range(0, zoneNum):
            index.append(i)

        for i, row in enumerate(resultReader):
            if row[trueNum] != "True":
                continue

            if i == 2:
                maxDist = float(row[distColNum])

            if i > 1:
                if row[commcompColNum] == "comm":
                    commSim[zonalPartition(float(row[distColNum]), zoneNum, maxDist)] += float(row[simColNum])
                elif row[commcompColNum] == "comp":
                    compSim[zonalPartition(float(row[distColNum]), zoneNum, maxDist)] += float(row[simColNum])
                else:
                    raise Exception("comm_comp is not giving comm or comp! The new type: " + row[colNum])
        
        plt.figure()
        plotdata = pd.DataFrame({
            "comm":commSim,
            "comp":compSim
        }, index = index
        )
        plotdata.plot(kind = 'bar', stacked = True)
        plt.xlabel("Zone decided by the max distance to goal")
        plt.ylabel("Simulation Time")
        plt.title("Simulation Time in Each Zone based on comm_comp")
        plt.savefig(dirName + fileName + "/simTimeVScommCompZoneDist-" + fileName + ".png")
        # plt.show()
        plt.close('all')

# the function to plot simulation time vs. comm_comp in a zonal format
def plotMovGenTimeVScommCompZoneDist(dirName, fileName, zoneNum, commcompColNum, distColNum, movGenColNum, trueNum):
    with open(dirName + fileName + "/result_summary/FARSI_simple_run_0_1_all_reults.csv", newline='') as csvfile:
        resultReader = csv.reader(csvfile, delimiter=',', quotechar='|')

        commMov = np.zeros(zoneNum, dtype = float)
        compMov = np.zeros(zoneNum, dtype = float)

        maxDist = 0
        index = []
        for i in range(0, zoneNum):
            index.append(i)

        for i, row in enumerate(resultReader):
            if row[trueNum] != "True":
                continue

            if i == 2:
                maxDist = float(row[distColNum])

            if i > 1:
                if row[commcompColNum] == "comm":
                    commMov[zonalPartition(float(row[distColNum]), zoneNum, maxDist)] += float(row[movGenColNum])
                elif row[commcompColNum] == "comp":
                    compMov[zonalPartition(float(row[distColNum]), zoneNum, maxDist)] += float(row[movGenColNum])
                else:
                    raise Exception("comm_comp is not giving comm or comp! The new type: " + row[colNum])
        
        plt.figure()
        plotdata = pd.DataFrame({
            "comm":commMov,
            "comp":compMov
        }, index = index
        )
        plotdata.plot(kind = 'bar', stacked = True)
        plt.xlabel("Zone decided by the max distance to goal")
        plt.ylabel("Move Generation Time")
        plt.title("Move Generation Time in Each Zone based on comm_comp")
        plt.savefig(dirName + fileName + "/movGenTimeVScommCompZoneDist-" + fileName + ".png")
        # plt.show()
        plt.close('all')

# the function to plot simulation time vs. optimization name in a zonal format
def plotSimTimeVShighLevelOptZoneDist(dirName, fileName, zoneNum, optColNum, distColNum, simColNum, trueNum):
    with open(dirName + fileName + "/result_summary/FARSI_simple_run_0_1_all_reults.csv", newline='') as csvfile:
        resultReader = csv.reader(csvfile, delimiter=',', quotechar='|')

        topoSim = np.zeros(zoneNum, dtype = float)
        tunSim = np.zeros(zoneNum, dtype = float)
        mapSim = np.zeros(zoneNum, dtype = float)
        idenOptSim = np.zeros(zoneNum, dtype = float)

        maxDist = 0
        index = []
        for i in range(0, zoneNum):
            index.append(i)

        for i, row in enumerate(resultReader):
            if row[trueNum] != "True":
                continue

            if i == 2:
                maxDist = float(row[distColNum])

            if i > 1:
                if row[optColNum] == "topology":
                    topoSim[zonalPartition(float(row[distColNum]), zoneNum, maxDist)] += float(row[simColNum])
                elif row[optColNum] == "tunning":
                    tunSim[zonalPartition(float(row[distColNum]), zoneNum, maxDist)] += float(row[simColNum])
                elif row[optColNum] == "mapping":
                    mapSim[zonalPartition(float(row[distColNum]), zoneNum, maxDist)] += float(row[simColNum])
                elif row[optColNum] == "identity":
                    idenOptSim[zonalPartition(float(row[distColNum]), zoneNum, maxDist)] += float(row[simColNum])
                else:
                    raise Exception("optimization name is not giving topology or tunning or mapping or identity! The new type: " + row[optColNum])
        
        plt.figure()
        plotdata = pd.DataFrame({
            "topology":topoSim,
            "tunning":tunSim,
            "mapping":mapSim,
            "identity":idenOptSim
        }, index = index
        )
        plotdata.plot(kind = 'bar', stacked = True)
        plt.xlabel("Zone decided by the max distance to goal")
        plt.ylabel("Simulation Time")
        plt.title("Simulation Time in Each Zone based on Optimation Name")
        plt.savefig(dirName + fileName + "/simTimeVShighLevelOptZoneDist-" + fileName + ".png")
        # plt.show()
        plt.close('all')

# the function to plot simulation time vs. optimization name in a zonal format
def plotMovGenTimeVShighLevelOptZoneDist(dirName, fileName, zoneNum, optColNum, distColNum, movGenColNum, trueNum):
    with open(dirName + fileName + "/result_summary/FARSI_simple_run_0_1_all_reults.csv", newline='') as csvfile:
        resultReader = csv.reader(csvfile, delimiter=',', quotechar='|')

        topoMov = np.zeros(zoneNum, dtype = float)
        tunMov = np.zeros(zoneNum, dtype = float)
        mapMov = np.zeros(zoneNum, dtype = float)
        idenOptMov = np.zeros(zoneNum, dtype = float)

        maxDist = 0
        index = []
        for i in range(0, zoneNum):
            index.append(i)

        for i, row in enumerate(resultReader):
            if row[trueNum] != "True":
                continue

            if i == 2:
                maxDist = float(row[distColNum])

            if i > 1:
                if row[optColNum] == "topology":
                    topoMov[zonalPartition(float(row[distColNum]), zoneNum, maxDist)] += float(row[movGenColNum])
                elif row[optColNum] == "tunning":
                    tunMov[zonalPartition(float(row[distColNum]), zoneNum, maxDist)] += float(row[movGenColNum])
                elif row[optColNum] == "mapping":
                    mapMov[zonalPartition(float(row[distColNum]), zoneNum, maxDist)] += float(row[movGenColNum])
                elif row[optColNum] == "identity":
                    idenOptMov[zonalPartition(float(row[distColNum]), zoneNum, maxDist)] += float(row[movGenColNum])
                else:
                    raise Exception("optimization name is not giving topology or tunning or mapping or identity! The new type: " + row[optColNum])
        
        plt.figure()
        plotdata = pd.DataFrame({
            "topology":topoMov,
            "tunning":tunMov,
            "mapping":mapMov,
            "identity":idenOptMov
        }, index = index
        )
        plotdata.plot(kind = 'bar', stacked = True)
        plt.xlabel("Zone decided by the max distance to goal")
        plt.ylabel("Move Generation Time")
        plt.title("Move Generation Time in Each Zone based on Optimization Name")
        plt.savefig(dirName + fileName + "/movGenTimeVShighLevelOptZoneDist-" + fileName + ".png")
        # plt.show()
        plt.close('all')

# the function to plot simulation time vs. architectural principle in a zonal format
def plotSimTimeVSarchVarImpZoneDist(dirName, fileName, zoneNum, archColNum, distColNum, simColNum, trueNum):
    with open(dirName + fileName + "/result_summary/FARSI_simple_run_0_1_all_reults.csv", newline='') as csvfile:
        resultReader = csv.reader(csvfile, delimiter=',', quotechar='|')

        paraSim = np.zeros(zoneNum, dtype = float)
        custSim = np.zeros(zoneNum, dtype = float)
        localSim = np.zeros(zoneNum, dtype = float)
        idenImpSim = np.zeros(zoneNum, dtype = float)

        maxDist = 0
        index = []
        for i in range(0, zoneNum):
            index.append(i)

        for i, row in enumerate(resultReader):
            if row[trueNum] != "True":
                continue

            if i == 2:
                maxDist = float(row[distColNum])

            if i > 1:
                if row[archColNum] == "parallelization":
                    paraSim[zonalPartition(float(row[distColNum]), zoneNum, maxDist)] += float(row[simColNum])
                elif row[archColNum] == "customization":
                    custSim[zonalPartition(float(row[distColNum]), zoneNum, maxDist)] += float(row[simColNum])
                elif row[archColNum] == "locality":
                    localSim[zonalPartition(float(row[distColNum]), zoneNum, maxDist)] += float(row[simColNum])
                elif row[archColNum] == "identity":
                    idenImpSim[zonalPartition(float(row[distColNum]), zoneNum, maxDist)] += float(row[simColNum])
                else:
                    raise Exception("architectural principle is not giving parallelization or customization or locality or identity! The new type: " + row[archColNum])
        
        plt.figure()
        plotdata = pd.DataFrame({
            "parallelization":paraSim,
            "customization":custSim,
            "locality":localSim,
            "identity":idenImpSim
        }, index = index
        )
        plotdata.plot(kind = 'bar', stacked = True)
        plt.xlabel("Zone decided by the max distance to goal")
        plt.ylabel("Simulation Time")
        plt.title("Simulation Time in Each Zone based on Architectural Principle")
        plt.savefig(dirName + fileName + "/simTimeVSarchVarImpZoneDist-" + fileName + ".png")
        # plt.show()
        plt.close('all')

# the function to plot simulation time vs. architectural principle in a zonal format
def plotMovGenTimeVSarchVarImpZoneDist(dirName, fileName, zoneNum, archColNum, distColNum, movGenColNum, trueNum):
    with open(dirName + fileName + "/result_summary/FARSI_simple_run_0_1_all_reults.csv", newline='') as csvfile:
        resultReader = csv.reader(csvfile, delimiter=',', quotechar='|')

        paraMov = np.zeros(zoneNum, dtype = float)
        custMov = np.zeros(zoneNum, dtype = float)
        localMov = np.zeros(zoneNum, dtype = float)
        idenImpMov = np.zeros(zoneNum, dtype = float)

        maxDist = 0
        index = []
        for i in range(0, zoneNum):
            index.append(i)

        for i, row in enumerate(resultReader):
            if row[trueNum] != "True":
                continue

            if i == 2:
                maxDist = float(row[distColNum])

            if i > 1:
                if row[archColNum] == "parallelization":
                    paraMov[zonalPartition(float(row[distColNum]), zoneNum, maxDist)] += float(row[movGenColNum])
                elif row[archColNum] == "customization":
                    custMov[zonalPartition(float(row[distColNum]), zoneNum, maxDist)] += float(row[movGenColNum])
                elif row[archColNum] == "locality":
                    localMov[zonalPartition(float(row[distColNum]), zoneNum, maxDist)] += float(row[movGenColNum])
                elif row[archColNum] == "identity":
                    idenImpMov[zonalPartition(float(row[distColNum]), zoneNum, maxDist)] += float(row[movGenColNum])
                else:
                    raise Exception("architectural principle is not giving parallelization or customization or locality or identity! The new type: " + row[archColNum])
        
        plt.figure()
        plotdata = pd.DataFrame({
            "parallelization":paraMov,
            "customization":custMov,
            "locality":localMov,
            "identity":idenImpMov
        }, index = index
        )
        plotdata.plot(kind = 'bar', stacked = True)
        plt.xlabel("Zone decided by the max distance to goal")
        plt.ylabel("Move Generation Time")
        plt.title("Move Generation Time in Each Zone based on Architectural Principle")
        plt.savefig(dirName + fileName + "/movGenTimeVSarchVarImpZoneZoneDist-" + fileName + ".png")
        # plt.show()
        plt.close('all')

# the function to plot convergence vs. iterationxdepth
def plotConvergeVSitr3d(dirName, fileName, latNum, powNum, areaNum, trueNum):
    with open(dirName + fileName + "/result_summary/FARSI_simple_run_0_1_all_reults.csv", newline='') as csvfile:
        resultReader = csv.reader(csvfile, delimiter=',', quotechar='|')

        for i, row in enumerate(resultReader):
            if row[trueNum] != "True":
                continue

            if i > 1:
                latDict = eval(row[latNum])
                keys = list(latDict.keys())
                latList = list(latDict.values())
                # print(keys)
        
        plt.figure()
        ax = plt.axes(projection = '3d')
        plt.show()

# the main function. comment out the plots if you do not need them
if __name__ == "__main__":
    # change the directory name and the result folder name accordingly. the directory name is the place for all your results
    dirName = "/home/yingj4/Desktop/Project_FARSI/data_collection/data/simple_run/07-15-2021/"
    # if you want to generate the figures for a single result folder, change the fileName variable on the next line (and of course, move the functions outside the loop below). otherwise, it will do an automatic sweep
    fileName = "07-14_20-32_39"

    # change the number of zones to suit for your analysis
    zoneNum = 4

    fileList = os.listdir(dirName)

    for fileName in fileList:
        print(fileName)

        # comment and uncomment the following functions for your plottings
        
        # plotCommCompAll(dirName, fileName, columnNum(dirName, fileName, "comm_comp"), columnNum(dirName, fileName, "move validity"))
        # plothighLevelOptAll(dirName, fileName, columnNum(dirName, fileName, "optimization name"), columnNum(dirName, fileName, "move validity"))
        # plotArchVarImpAll(dirName, fileName, columnNum(dirName, fileName, "architectural principle"), columnNum(dirName, fileName, "move validity"))
        # plotSimTimeVSblk(dirName, fileName, columnNum(dirName, fileName, "system block count"), columnNum(dirName, fileName, "simulation time"), columnNum(dirName, fileName, "move validity"))
        # plotMoveGenTimeVSblk(dirName, fileName, columnNum(dirName, fileName, "system block count"), columnNum(dirName, fileName, "move generation time"), columnNum(dirName, fileName, "move validity"))
        # plotDistToGoalVSitr(dirName, fileName, columnNum(dirName, fileName, "iterationxdepth number"), columnNum(dirName, fileName, "dist_to_goal_non_cost"), columnNum(dirName, fileName, "move validity"))
        # plotRefDistToGoalVSitr(dirName, fileName, columnNum(dirName, fileName, "iterationxdepth number"), columnNum(dirName, fileName, "ref_des_dist_to_goal_non_cost"), columnNum(dirName, fileName, "move validity"))
        # plotSimTimeVSmoveNameZoneDist(dirName, fileName, zoneNum, columnNum(dirName, fileName, "move name"), columnNum(dirName, fileName, "dist_to_goal_non_cost"), columnNum(dirName, fileName, "simulation time"), columnNum(dirName, fileName, "move validity"))
        # plotMovGenTimeVSmoveNameZoneDist(dirName, fileName, zoneNum, columnNum(dirName, fileName, "move name"), columnNum(dirName, fileName, "dist_to_goal_non_cost"), columnNum(dirName, fileName, "move generation time"), columnNum(dirName, fileName, "move validity"))
        # plotSimTimeVScommCompZoneDist(dirName, fileName, zoneNum, columnNum(dirName, fileName, "comm_comp"), columnNum(dirName, fileName, "dist_to_goal_non_cost"), columnNum(dirName, fileName, "simulation time"), columnNum(dirName, fileName, "move validity"))
        # plotMovGenTimeVScommCompZoneDist(dirName, fileName, zoneNum, columnNum(dirName, fileName, "comm_comp"), columnNum(dirName, fileName, "dist_to_goal_non_cost"), columnNum(dirName, fileName, "move generation time"), columnNum(dirName, fileName, "move validity"))
        # plotSimTimeVShighLevelOptZoneDist(dirName, fileName, zoneNum, columnNum(dirName, fileName, "optimization name"), columnNum(dirName, fileName, "dist_to_goal_non_cost"), columnNum(dirName, fileName, "simulation time"), columnNum(dirName, fileName, "move validity"))
        # plotMovGenTimeVShighLevelOptZoneDist(dirName, fileName, zoneNum, columnNum(dirName, fileName, "optimization name"), columnNum(dirName, fileName, "dist_to_goal_non_cost"), columnNum(dirName, fileName, "move generation time"), columnNum(dirName, fileName, "move validity"))
        # plotSimTimeVSarchVarImpZoneDist(dirName, fileName, zoneNum, columnNum(dirName, fileName, "architectural principle"), columnNum(dirName, fileName, "dist_to_goal_non_cost"), columnNum(dirName, fileName, "simulation time"), columnNum(dirName, fileName, "move validity"))
        # plotMovGenTimeVSarchVarImpZoneDist(dirName, fileName, zoneNum, columnNum(dirName, fileName, "architectural principle"), columnNum(dirName, fileName, "dist_to_goal_non_cost"), columnNum(dirName, fileName, "move generation time"), columnNum(dirName, fileName, "move validity"))
        plotConvergeVSitr3d(dirName, fileName, columnNum(dirName, fileName, "latency"), columnNum(dirName, fileName, "power"), columnNum(dirName, fileName, "area"), columnNum(dirName, fileName, "move validity"))
        # break
