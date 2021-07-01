# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 12:32:07 2021
Script used to compare time series from multiple configurations for 
forces, angles, moments, and powers
@author: Daniel.Feeney
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.signal as sig

# Define constants and options
fileToLoad = 24
fileToLoad2 = 26
fileToLoad3 = 27
runTrial = 1 #set to 1 for running 
fThresh = 50; #below this value will be set to 0.
writeData = 0; #will write to spreadsheet if 1 entered
plottingEnabled = 0 #plots the bottom if 1. No plots if 0
stepLen = 50
manualTrim = 0
x = np.linspace(0,stepLen,stepLen)
pd.options.mode.chained_assignment = None  # default='warn' set to warn for a lot of warnings

# Read in balance file
#fPath = 'C:\\Users\\Daniel.Feeney\\Dropbox (Boa)\\Hike Work Research\\Hike Pilot 2021\\TM\Kinetics\\'
fPath = 'C:\\Users\\Daniel.Feeney\\Dropbox (Boa)\\Endurance Health Validation\\DU_Running_Summer_2021\\Data\\KineticsKinematics\\'
fileExt = r".txt"
entries = [fName for fName in os.listdir(fPath) if fName.endswith(fileExt)]


# list of functions 
# finding landings on the force plate once the filtered force exceeds the force threshold
def findLandings(force, fThresh):
    lic = []
    for step in range(len(force)-1):
        if force[step] == 0 and force[step + 1] >= fThresh:
            lic.append(step)
    return lic

#Find takeoff from FP when force goes from above thresh to 0
def findTakeoffs(force, fThresh):
    lto = []
    for step in range(len(force)-1):
        if force[step] >= fThresh and force[step + 1] == 0:
            lto.append(step + 1)
    return lto

def calcVLR(force, startVal, lengthFwd):
    # function to calculate VLR from 80 and 20% of the max value observed in the first n
    # indices (n defined by lengthFwd). 
    maxF = np.max(force[startVal:startVal+lengthFwd])
    eightyPctMax = 0.8 * maxF
    twentyPctMax = 0.2 * maxF
    # find indices of 80 and 20 and calc loading rate as diff in force / diff in time (N/s)
    eightyIndex = next(x for x, val in enumerate(force[startVal:startVal+lengthFwd]) 
                      if val > eightyPctMax) 
    twentyIndex = next(x for x, val in enumerate(force[startVal:startVal+lengthFwd]) 
                      if val > twentyPctMax) 
    VLR = ((eightyPctMax - twentyPctMax) / ((eightyIndex/1000) - (twentyIndex/1000)))
    return(VLR)
    
#Find max braking force moving forward
def calcPeakBrake(force, landing, length):
    newForce = np.array(force)
    return min(newForce[landing:landing+length])

def findNextZero(force, length):
    # Starting at a landing, look forward (after first 15 indices)
    # to the find the next time the signal goes from - to +
    for step in range(length):
        if force[step] <= 0 and force[step + 1] >= 0 and step > 15:
            break
    return step


def delimitTrial(inputDF):
    # generic function to plot and start/end trial #
    fig, ax = plt.subplots()
    ax.plot(inputDF.ForcesZ, label = 'Left Force')
    fig.legend()
    pts = np.asarray(plt.ginput(2, timeout=-1))
    plt.close()
    outputDat = inputDF.iloc[int(np.floor(pts[0,0])) : int(np.floor(pts[1,0])),:]
    outputDat = outputDat.reset_index()
    return(outputDat)

def filterForce(inputForce, sampFrq, cutoffFrq):
        # low-pass filter the input force signals
        #t = np.arange(len(inputForce)) / sampFrq
        w = cutoffFrq / (sampFrq / 2) # Normalize the frequency
        b, a = sig.butter(4, w, 'low')
        filtForce = sig.filtfilt(b, a, inputForce)
        return(filtForce)
    
def thresholdForce(inputDF, threshForce):
    forceTot = inputDF.ForcesZ
    forceTot[forceTot<threshForce] = 0
    forceTot = np.array(forceTot)
    return(forceTot)
    
# preallocate matrix for force and fill in with force data
def forceMatrix(inputForce, landings, noSteps, stepLength):
    #input a force signal, return matrix with n rows (for each landing) by m col
    #for each point in stepLen.
    #skip every other landing for TM FP data
    preForce = np.zeros((noSteps,stepLength))
    
    for iterVar, landing in enumerate(landings):
        try:
            preForce[iterVar,] = inputForce[landing:landing+stepLength]
        except:
            print(landing)
            
    return preForce


def defThreshold(inputDF):
        # find threshold force
        fig, ax = plt.subplots()
        ax.plot(inputDF.ForcesZ, label = 'Right Foot Force')
        print('Select a point to represent 0 in the trial')
        pts = np.asarray(plt.ginput(1, timeout=-1))
        plt.close()
        fThresh = pts[0][1]
        return(fThresh)
    
def makeCompFig(avgVal1, sdVal1, avgVal2, sdVal2, avgVal3, sdVal3, avgVal4, sdVal4, 
                avgVal5, sdVal5,avgVal6, sdVal6, avgVal7, sdVal7, avgVal8, sdVal8,
                avgVal9, sdVal9, Ylabel1, Ylabel2, Ylabel3):
        # plot ensemble average values from landings defined in a
        # a different function above. Takes the avg and std of the columns
        # as inputs
        fig, (ax1, ax2, ax3) = plt.subplots(1,3)
        ax1.plot(x, avgVal1, 'k', color='#000000', label = '{}'.format(config1))
        ax1.plot(x, avgVal4, 'k', color = '#DC582A', label = '{}'.format(config2))
        ax1.plot(x, avgVal7, 'k', color = '#CAF0E4', label = '{}'.format(config3))
        ax1.set_xlabel('Time')
        ax1.set_ylabel(f"{Ylabel1}")
        ax1.fill_between(x, avgVal1-sdVal1, avgVal1+sdVal1,
            alpha=0.5, edgecolor='#000000', facecolor='#000000')
        ax1.fill_between(x, avgVal4-sdVal4, avgVal4+sdVal4,
            alpha=0.5, edgecolor='#DC582A', facecolor='#DC582A')
        ax1.fill_between(x, avgVal7-sdVal7, avgVal7+sdVal7,
            alpha=0.5, edgecolor='#CAF0E4', facecolor='#CAF0E4')
        
        ax2.plot(x, avgVal2, 'k', color='#000000')
        ax2.plot(x, avgVal5, 'k', color='#DC582A')
        ax2.plot(x, avgVal8, 'k', color = '#CAF0E4')
        ax2.set_xlabel('Time')
        ax2.set_ylabel(f"{Ylabel2}")
        ax2.fill_between(x, avgVal2-sdVal2, avgVal2+sdVal2,
            alpha=0.5, edgecolor='#000000', facecolor='#000000')
        ax2.fill_between(x, avgVal5-sdVal5, avgVal5+sdVal5,
            alpha=0.5, edgecolor='#DC582A', facecolor='#DC582A')
        ax2.fill_between(x, avgVal8-sdVal8, avgVal8+sdVal8,
            alpha=0.5, edgecolor='#CAF0E4', facecolor='#CAF0E4')

        ax3.plot(x, avgVal3, 'k', color='#000000', label = 'Lace')
        ax3.plot(x, avgVal6, 'k', color='#DC582A', label = 'Dual Dial')
        ax3.plot(x, avgVal9, 'k', color='#CAF0E4', label = 'Single Dial')
        ax3.set_xlabel('Time')
        ax3.set_ylabel(f"{Ylabel3}")
        ax3.fill_between(x, avgVal3-sdVal3, avgVal3+sdVal3,
            alpha=0.5, edgecolor='#000000', facecolor='#000000')
        ax3.fill_between(x, avgVal6-sdVal6, avgVal6+sdVal6,
            alpha=0.5, edgecolor='#DC582A', facecolor='#DC582A')
        ax3.fill_between(x, avgVal9-sdVal9, avgVal9+sdVal9,
            alpha=0.5, edgecolor='#CAF0E4', facecolor='#CAF0E4')
        ax3.legend()
        plt.legend()
        plt.suptitle('{}'.format(subName))
        plt.tight_layout()
    
# determine if first step is left or right then delete every other
def toTrimOrNot(data,landings, takeoffs, runTrial):
    if runTrial == 1:
        if (np.mean(dat.LCOPx[landings[0]:takeoffs[0]]) < np.mean(dat.LCOPx[landings[1]:takeoffs[1]])):
            trimmedLandings = [i for a, i in enumerate(landings) if  a%2 == 0]
            trimmedTakeoffs = [i for a, i in enumerate(takeoffs) if  a%2 == 0]
        else:
            trimmedLandings = [i for a, i in enumerate(landings) if  a%2 != 0]
            trimmedTakeoffs = [i for a, i in enumerate(takeoffs) if  a%2 != 0]
    else:
        trimmedLandings = landings
        trimmedTakeoffs = takeoffs
    return(trimmedLandings, trimmedTakeoffs)

def trimTakeoffs(landingVec, takeoffVec):
    if landingVec[0] > takeoffVec[0]:
        takeoffVec.pop(0)
        return(takeoffVec)
    else:
        return(takeoffVec)

### load file in
fName = entries[fileToLoad] #Load one file at a time
fName2 = entries[fileToLoad2]
fName3 = entries[fileToLoad3]

config1 = fName.split('_')[1]
config2 = fName2.split('_')[1]
config3 = fName3.split('_')[1]
subName = fName.split('_')[0]

dat = pd.read_csv(fPath+fName,sep='\t', skiprows = 8, header = 0)  
dat2 = pd.read_csv(fPath+fName2,sep='\t', skiprows = 8, header = 0)
dat3 = pd.read_csv(fPath+fName3,sep='\t', skiprows = 8, header = 0)

dat.ForcesZ = dat.ForcesZ * -1
dat2.ForcesZ = dat2.ForcesZ * -1
dat3.ForcesZ = dat3.ForcesZ * -1

# define a ton and unpack variables
trimmedForce = thresholdForce(dat, fThresh)
trimmedForce2 = thresholdForce(dat2, fThresh)
trimmedForce3 = thresholdForce(dat3, fThresh)

#find the landings and offs of the FP as vectors from function above
landings = findLandings(trimmedForce, fThresh)
takeoffs = findTakeoffs(trimmedForce, fThresh)
takeoffs = trimTakeoffs(landings, takeoffs)
# two
landings2 = findLandings(trimmedForce2, fThresh)
takeoffs2 = findTakeoffs(trimmedForce2, fThresh)
takeoffs2 = trimTakeoffs(landings2, takeoffs2)
# three
landings3 = findLandings(trimmedForce3, fThresh)
takeoffs3 = findTakeoffs(trimmedForce3, fThresh)
takeoffs3 = trimTakeoffs(landings3, takeoffs3)

# Trim the end and skip every other value for running data
(trimmedLandings,trimmedTakeoffs) = toTrimOrNot(dat, landings, takeoffs, runTrial)
#two
(trimmedLandings2,trimmedTakeoffs2) = toTrimOrNot(dat2, landings2, takeoffs2, runTrial)
#three
(trimmedLandings3,trimmedTakeoffs3) = toTrimOrNot(dat3, landings3, takeoffs3, runTrial)


class ensembleMeanData:
    ### This will be filled in with the stacked dataframes using the forceMatrix function ###
    ### plot from these stacked dataframes for each configuration ###
    def __init__(self, Fz, Fx, Fy, AP, KP, HP, AF, AIV, AABD,AMx, AMy, AMz, KF, KR, KMx, KMy, KMz, HF, HI, HA, HMx, HMy, HMz):
        self.Forcez = Fz
        self.Forcey = Fy
        self.Forcex = Fx
        self.AnklePower = AP
        self.KneePower = KP
        self.HipPower = HP
        self.AnkleFlex = AF
        self.AnkleInv = AIV
        self.AnkleAbd = AABD
        self.AnkleMomX = AMx
        self.AnkleMomY = AMy
        self.AnkleMomZ = AMz
        self.KneeFlex = KF
        self.KneeRot = KR
        self.KneeMomX = KMx
        self.KneeMomY = KMy
        self.KneeMomZ = KMz
        self.HipFlex = HF
        self.HipInt = HI
        self.HipAbd = HA
        self.HipMomX = HMx
        self.HipMomY = HMy
        self.HipMomZ = HMz

meanDat1 = ensembleMeanData(np.mean(forceMatrix(trimmedForce, trimmedLandings, 10, stepLen), axis = 0), np.mean(forceMatrix(dat.ForcesX, trimmedLandings, 10, stepLen), axis = 0), 
                np.mean(forceMatrix(dat.ForcesY*-1, trimmedLandings, 10, stepLen),axis = 0), np.mean(forceMatrix(dat.LAnklePower, trimmedLandings, 10, stepLen), axis = 0), 
                np.mean(forceMatrix(dat.LKneePower, trimmedLandings, 10, stepLen),axis=0), np.mean(forceMatrix(dat.LHipPower, trimmedLandings, 10, stepLen),axis=0),
                np.mean(forceMatrix(dat.LAnkleFlex, trimmedLandings, 10, stepLen),axis=0), np.mean(forceMatrix(dat.LAnkleInv, trimmedLandings, 10, stepLen),axis=0),
                np.mean(forceMatrix(dat.LAnkleAbd, trimmedLandings, 10, stepLen),axis=0), np.mean(forceMatrix(dat.LAnkleMomentx, trimmedLandings, 10, stepLen),axis=0), 
                np.mean(forceMatrix(dat.LAnkleMomenty, trimmedLandings, 10, stepLen),axis=0), np.mean(forceMatrix(dat.LAnkleMomentz, trimmedLandings, 10, stepLen),axis=0),
                np.mean(forceMatrix(dat.LKneeFlex, trimmedLandings, 10, stepLen),axis=0), np.mean(forceMatrix(dat.LKneeRot, trimmedLandings, 10, stepLen),axis=0), 
                np.mean(forceMatrix(dat.LKneeMomentX, trimmedLandings, 10, stepLen),axis=0), 
                np.mean(forceMatrix(dat.LKneeMomentY, trimmedLandings, 10, stepLen),axis=0), np.mean(forceMatrix(dat.LKneeMomentZ, trimmedLandings, 10, stepLen),axis=0),
                np.mean(forceMatrix(dat.LHipFlex, trimmedLandings, 10, stepLen),axis=0), np.mean(forceMatrix(dat.LHipInt, trimmedLandings, 10, stepLen),axis=0),
                np.mean(forceMatrix(dat.LHipAbd, trimmedLandings, 10, stepLen),axis=0), np.mean(forceMatrix(dat.LHipMomentx, trimmedLandings, 10, stepLen),axis=0), 
                np.mean(forceMatrix(dat.LHipMomenty, trimmedLandings, 10, stepLen),axis=0), np.mean(forceMatrix(dat.LHipMomentz, trimmedLandings, 10, stepLen),axis=0))

meanDat2 = ensembleMeanData(np.mean(forceMatrix(trimmedForce2, trimmedLandings2, 10, stepLen), axis = 0), np.mean(forceMatrix(dat2.ForcesX, trimmedLandings2, 10, stepLen), axis = 0), 
                np.mean(forceMatrix(dat2.ForcesY*-1, trimmedLandings2, 10, stepLen),axis = 0), np.mean(forceMatrix(dat2.LAnklePower, trimmedLandings2, 10, stepLen), axis = 0), 
                np.mean(forceMatrix(dat2.LKneePower, trimmedLandings2, 10, stepLen),axis=0), np.mean(forceMatrix(dat2.LHipPower, trimmedLandings2, 10, stepLen),axis=0),
                np.mean(forceMatrix(dat2.LAnkleFlex, trimmedLandings2, 10, stepLen),axis=0), np.mean(forceMatrix(dat2.LAnkleInv, trimmedLandings2, 10, stepLen),axis=0),
                np.mean(forceMatrix(dat2.LAnkleAbd, trimmedLandings2, 10, stepLen),axis=0), np.mean(forceMatrix(dat2.LAnkleMomentx, trimmedLandings2, 10, stepLen),axis=0), 
                np.mean(forceMatrix(dat2.LAnkleMomenty, trimmedLandings2, 10, stepLen),axis=0), np.mean(forceMatrix(dat2.LAnkleMomentz, trimmedLandings2, 10, stepLen),axis=0),
                np.mean(forceMatrix(dat2.LKneeFlex, trimmedLandings2, 10, stepLen),axis=0), np.mean(forceMatrix(dat2.LKneeRot, trimmedLandings2, 10, stepLen),axis=0), 
                np.mean(forceMatrix(dat2.LKneeMomentX, trimmedLandings2, 10, stepLen),axis=0), 
                np.mean(forceMatrix(dat2.LKneeMomentY, trimmedLandings2, 10, stepLen),axis=0), np.mean(forceMatrix(dat2.LKneeMomentZ, trimmedLandings2, 10, stepLen),axis=0),
                np.mean(forceMatrix(dat2.LHipFlex, trimmedLandings2, 10, stepLen),axis=0), np.mean(forceMatrix(dat2.LHipInt, trimmedLandings2, 10, stepLen),axis=0),
                np.mean(forceMatrix(dat2.LHipAbd, trimmedLandings2, 10, stepLen),axis=0), np.mean(forceMatrix(dat2.LHipMomentx, trimmedLandings2, 10, stepLen),axis=0), 
                np.mean(forceMatrix(dat2.LHipMomenty, trimmedLandings2, 10, stepLen),axis=0), np.mean(forceMatrix(dat2.LHipMomentz, trimmedLandings2, 10, stepLen),axis=0))

meanDat3 = ensembleMeanData(np.mean(forceMatrix(trimmedForce3, trimmedLandings3, 10, stepLen), axis = 0), np.mean(forceMatrix(dat3.ForcesX, trimmedLandings3, 10, stepLen), axis = 0), 
                np.mean(forceMatrix(dat3.ForcesY*-1, trimmedLandings3, 10, stepLen),axis = 0), np.mean(forceMatrix(dat3.LAnklePower, trimmedLandings3, 10, stepLen), axis = 0), 
                np.mean(forceMatrix(dat3.LKneePower, trimmedLandings3, 10, stepLen),axis=0), np.mean(forceMatrix(dat3.LHipPower, trimmedLandings3, 10, stepLen),axis=0),
                np.mean(forceMatrix(dat3.LAnkleFlex, trimmedLandings3, 10, stepLen),axis=0), np.mean(forceMatrix(dat3.LAnkleInv, trimmedLandings3, 10, stepLen),axis=0),
                np.mean(forceMatrix(dat3.LAnkleAbd, trimmedLandings3, 10, stepLen),axis=0), np.mean(forceMatrix(dat3.LAnkleMomentx, trimmedLandings3,10, stepLen),axis=0), 
                np.mean(forceMatrix(dat3.LAnkleMomenty, trimmedLandings3, 10, stepLen),axis=0), np.mean(forceMatrix(dat3.LAnkleMomentz, trimmedLandings3, 10, stepLen),axis=0),
                np.mean(forceMatrix(dat3.LKneeFlex, trimmedLandings3, 10, stepLen),axis=0), np.mean(forceMatrix(dat3.LKneeRot, trimmedLandings3, 10, stepLen),axis=0), 
                np.mean(forceMatrix(dat3.LKneeMomentX, trimmedLandings3, 10, stepLen),axis=0), 
                np.mean(forceMatrix(dat3.LKneeMomentY, trimmedLandings3, 10, stepLen),axis=0), np.mean(forceMatrix(dat3.LKneeMomentZ, trimmedLandings3, 10, stepLen),axis=0),
                np.mean(forceMatrix(dat3.LHipFlex, trimmedLandings3, 10, stepLen),axis=0), np.mean(forceMatrix(dat3.LHipInt, trimmedLandings3, 10, stepLen),axis=0),
                np.mean(forceMatrix(dat3.LHipAbd, trimmedLandings3, 10, stepLen),axis=0), np.mean(forceMatrix(dat3.LHipMomentx, trimmedLandings3, 10, stepLen),axis=0), 
                np.mean(forceMatrix(dat3.LHipMomenty, trimmedLandings3, 10, stepLen),axis=0), np.mean(forceMatrix(dat3.LHipMomentz, trimmedLandings3, 10, stepLen),axis=0))

class ensembleSDData:
    ### This will be filled in with the sd of stacked dataframes using the forceMatrix function ###
    ### plot from these stacked dataframes for each configuration ###
    def __init__(self, Fz, Fx, Fy, AP, KP, HP, AF, AIV, AABD,AMx, AMy, AMz, KF, KR, KMx, KMy, KMz, HF, HI, HA, HMx, HMy, HMz):
        self.sdForcez = Fz
        self.sdForcey = Fy
        self.sdForcex = Fx
        self.sdAnklePower = AP
        self.sdKneePower = KP
        self.sdHipPower = HP
        self.sdAnkleFlex = AF
        self.sdAnkleInv = AIV
        self.sdAnkleAbd = AABD
        self.sdAnkleMomX = AMx
        self.sdAnkleMomY = AMy
        self.sdAnkleMomZ = AMz
        self.sdKneeFlex = KF
        self.sdKneeRot = KR
        self.sdKneeMomX = KMx
        self.sdKneeMomY = KMy
        self.sdKneeMomZ = KMz
        self.sdHipFlex = HF
        self.sdHipInt = HI
        self.sdHipAbd = HA
        self.sdHipMomX = HMx
        self.sdHipMomY = HMy
        self.sdHipMomZ = HMz

sdDat1 = ensembleSDData(np.std(forceMatrix(trimmedForce, trimmedLandings, 10, stepLen), axis = 0), np.std(forceMatrix(dat.ForcesX, trimmedLandings, 10, stepLen), axis = 0), 
                np.std(forceMatrix(dat.ForcesY, trimmedLandings, 10, stepLen),axis = 0), np.std(forceMatrix(dat.LAnklePower, trimmedLandings, 10, stepLen), axis = 0), 
                np.std(forceMatrix(dat.LKneePower, trimmedLandings, 10, stepLen),axis=0), np.std(forceMatrix(dat.LHipPower, trimmedLandings, 10, stepLen),axis=0),
                np.std(forceMatrix(dat.LAnkleFlex, trimmedLandings, 10, stepLen),axis=0), np.std(forceMatrix(dat.LAnkleInv, trimmedLandings, 10, stepLen),axis=0),
                np.std(forceMatrix(dat.LAnkleAbd, trimmedLandings, 10, stepLen),axis=0), np.std(forceMatrix(dat.LAnkleMomentx, trimmedLandings, 10, stepLen),axis=0), 
                np.std(forceMatrix(dat.LAnkleMomenty, trimmedLandings, 10, stepLen),axis=0), np.std(forceMatrix(dat.LAnkleMomentz, trimmedLandings, 10, stepLen),axis=0),
                np.std(forceMatrix(dat.LKneeFlex, trimmedLandings, 10, stepLen),axis=0), np.std(forceMatrix(dat.LKneeRot, trimmedLandings, 10, stepLen),axis=0), 
                np.std(forceMatrix(dat.LKneeMomentX, trimmedLandings, 10, stepLen),axis=0), 
                np.std(forceMatrix(dat.LKneeMomentY, trimmedLandings, 10, stepLen),axis=0), np.std(forceMatrix(dat.LKneeMomentZ, trimmedLandings, 10, stepLen),axis=0),
                np.std(forceMatrix(dat.LHipFlex, trimmedLandings, 10, stepLen),axis=0), np.std(forceMatrix(dat.LHipInt, trimmedLandings, 10, stepLen),axis=0),
                np.std(forceMatrix(dat.LHipAbd, trimmedLandings, 10, stepLen),axis=0), np.std(forceMatrix(dat.LHipMomentx, trimmedLandings, 10, stepLen),axis=0), 
                np.std(forceMatrix(dat.LHipMomenty, trimmedLandings, 10, stepLen),axis=0), np.std(forceMatrix(dat.LHipMomentz, trimmedLandings, 10, stepLen),axis=0))

sdDat2 = ensembleSDData(np.std(forceMatrix(trimmedForce2, trimmedLandings2, 10, stepLen), axis = 0), np.std(forceMatrix(dat2.ForcesX, trimmedLandings2, 10, stepLen), axis = 0), 
                np.std(forceMatrix(dat2.ForcesY, trimmedLandings2, 10, stepLen),axis = 0), np.std(forceMatrix(dat2.LAnklePower, trimmedLandings2, 10, stepLen), axis = 0), 
                np.std(forceMatrix(dat2.LKneePower, trimmedLandings2, 10, stepLen),axis=0), np.std(forceMatrix(dat2.LHipPower, trimmedLandings2, 10, stepLen),axis=0),
                np.std(forceMatrix(dat2.LAnkleFlex, trimmedLandings2, 10, stepLen),axis=0), np.std(forceMatrix(dat2.LAnkleInv, trimmedLandings2, 10, stepLen),axis=0),
                np.std(forceMatrix(dat2.LAnkleAbd, trimmedLandings2, 10, stepLen),axis=0), np.std(forceMatrix(dat2.LAnkleMomentx, trimmedLandings2, 10, stepLen),axis=0), 
                np.std(forceMatrix(dat2.LAnkleMomenty, trimmedLandings2, 10, stepLen),axis=0), np.std(forceMatrix(dat2.LAnkleMomentz, trimmedLandings2, 10, stepLen),axis=0),
                np.std(forceMatrix(dat2.LKneeFlex, trimmedLandings2, 10, stepLen),axis=0), np.std(forceMatrix(dat2.LKneeRot, trimmedLandings2, 10, stepLen),axis=0), 
                np.std(forceMatrix(dat2.LKneeMomentX, trimmedLandings2, 10, stepLen),axis=0), 
                np.std(forceMatrix(dat2.LKneeMomentY, trimmedLandings2, 10, stepLen),axis=0), np.std(forceMatrix(dat2.LKneeMomentZ, trimmedLandings2, 10, stepLen),axis=0),
                np.std(forceMatrix(dat2.LHipFlex, trimmedLandings2, 10, stepLen),axis=0), np.std(forceMatrix(dat2.LHipInt, trimmedLandings2, 10, stepLen),axis=0),
                np.std(forceMatrix(dat2.LHipAbd, trimmedLandings2, 10, stepLen),axis=0), np.std(forceMatrix(dat2.LHipMomentx, trimmedLandings2, 10, stepLen),axis=0), 
                np.std(forceMatrix(dat2.LHipMomenty, trimmedLandings2, 10, stepLen),axis=0), np.std(forceMatrix(dat2.LHipMomentz, trimmedLandings2, 10, stepLen),axis=0))

sdDat3 = ensembleSDData(np.std(forceMatrix(trimmedForce3, trimmedLandings3, 10, stepLen), axis = 0), np.std(forceMatrix(dat3.ForcesX, trimmedLandings3, 10, stepLen), axis = 0), 
                np.std(forceMatrix(dat3.ForcesY, trimmedLandings3, 10, stepLen),axis = 0), np.std(forceMatrix(dat3.LAnklePower, trimmedLandings3, 10, stepLen), axis = 0), 
                np.std(forceMatrix(dat3.LKneePower, trimmedLandings3, 10, stepLen),axis=0), np.std(forceMatrix(dat3.LHipPower, trimmedLandings3, 10, stepLen),axis=0),
                np.std(forceMatrix(dat3.LAnkleFlex, trimmedLandings3, 10, stepLen),axis=0), np.std(forceMatrix(dat3.LAnkleInv, trimmedLandings3, 10, stepLen),axis=0),
                np.std(forceMatrix(dat3.LAnkleAbd, trimmedLandings3, 10, stepLen),axis=0), np.std(forceMatrix(dat3.LAnkleMomentx, trimmedLandings3, 10, stepLen),axis=0), 
                np.std(forceMatrix(dat3.LAnkleMomenty, trimmedLandings3, 10, stepLen),axis=0), np.std(forceMatrix(dat3.LAnkleMomentz, trimmedLandings3, 10, stepLen),axis=0),
                np.std(forceMatrix(dat3.LKneeFlex, trimmedLandings3, 10, stepLen),axis=0), np.std(forceMatrix(dat3.LKneeRot, trimmedLandings3, 10, stepLen),axis=0), 
                np.std(forceMatrix(dat3.LKneeMomentX, trimmedLandings3, 10, stepLen),axis=0), 
                np.std(forceMatrix(dat3.LKneeMomentY, trimmedLandings3, 10, stepLen),axis=0), np.std(forceMatrix(dat3.LKneeMomentZ, trimmedLandings3, 10, stepLen),axis=0),
                np.std(forceMatrix(dat3.LHipFlex, trimmedLandings3, 10, stepLen),axis=0), np.std(forceMatrix(dat3.LHipInt, trimmedLandings3, 10, stepLen),axis=0),
                np.std(forceMatrix(dat3.LHipAbd, trimmedLandings3, 10, stepLen),axis=0), np.std(forceMatrix(dat3.LHipMomentx, trimmedLandings3, 10, stepLen),axis=0), 
                np.std(forceMatrix(dat3.LHipMomenty, trimmedLandings3, 10, stepLen),axis=0), np.std(forceMatrix(dat3.LHipMomentz, trimmedLandings3, 10, stepLen),axis=0))

### end ###


# make shaded error bar figures from std and mean defined above 
makeCompFig(meanDat1.Forcez, sdDat1.sdForcez, meanDat1.Forcex, sdDat1.sdForcex, meanDat1.Forcey, sdDat1.sdForcey, 
            meanDat2.Forcez, sdDat2.sdForcez, meanDat2.Forcex, sdDat2.sdForcex, meanDat2.Forcey, sdDat2.sdForcey,
            meanDat3.Forcez, sdDat3.sdForcez, meanDat3.Forcex, sdDat3.sdForcex, meanDat3.Forcey, sdDat3.sdForcey,
            'Z Force','X Force','Y Force')

makeCompFig(meanDat1.AnklePower, sdDat1.sdAnklePower, meanDat1.KneePower, sdDat1.sdKneePower, meanDat1.HipPower, sdDat1.sdHipPower, 
            meanDat2.AnklePower, sdDat2.sdAnklePower, meanDat2.KneePower, sdDat2.sdKneePower, meanDat2.HipPower, sdDat2.sdHipPower,
            meanDat3.AnklePower, sdDat3.sdAnklePower, meanDat3.KneePower, sdDat3.sdKneePower, meanDat3.HipPower, sdDat3.sdHipPower,
            'Ankle Power','Knee Power','Hip Power')

makeCompFig(meanDat1.AnkleFlex, sdDat1.sdAnkleFlex, meanDat1.KneeFlex, sdDat1.sdKneeFlex, meanDat1.HipFlex, sdDat1.sdHipFlex, 
            meanDat2.AnkleFlex, sdDat2.sdAnkleFlex, meanDat2.KneeFlex, sdDat2.sdKneeFlex, meanDat2.HipFlex, sdDat2.sdHipFlex,
            meanDat3.AnkleFlex, sdDat3.sdAnkleFlex, meanDat3.KneeFlex, sdDat3.sdKneeFlex, meanDat3.HipFlex, sdDat3.sdHipFlex,
            'Ankle Flexion','Knee Flexion','Hip Flexion')

