# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 08:47:20 2021

@author: Daniel.Feeney
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.signal as sig

# Define constants and options
fileToLoad = 12
fThresh = 50; #below this value will be set to 0.
writeData = 0; #will write to spreadsheet if 1 entered
plottingEnabled = 0 #plots the bottom if 1. No plots if 0
stepLen = 50
x = np.linspace(0,stepLen,stepLen)

# Read in balance file
fPath = 'C:\\Users\\Daniel.Feeney\\Dropbox (Boa)\\Hike Work Research\\Hike Pilot 2021\\TM\Kinetics\\'
fPath = 'C:\\Users\\Daniel.Feeney\\Dropbox (Boa)\\Endurance Health Validation\\DU_Running_Summer_2021\\Data\\KineticsKinematics\\'
#fPath = 'C:\\Users\\Daniel.Feeney\\Dropbox (Boa)\\EndurancePerformance\\Altra_MontBlanc_June2021\\Kinetics\\'
#fPath = 'C:\\Users\\Daniel.Feeney\\Dropbox (Boa)\\Endurance Health Validation\\DU_Running_Summer_2021\\Data\\Sub 01\\'
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
    
def trimForce(inputDF, threshForce):
    forceTot = inputDF.ForcesZ
    forceTot[forceTot<threshForce] = 0
    forceTot = np.array(forceTot)
    return(forceTot)

def makeFig(inputDF, forceCol, Xcol, Ycol, Zcol, title):
    # plot aligned time series data for first-look at the data
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    
    color = 'tab:red'
    ax1.set_xlabel('time')
    ax1.set_ylabel('TotalForce(N)', color=color)
    ax1.plot(inputDF[forceCol], color=color, label = 'Total Force')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    
    ax2.set_ylabel(title) 
    ax2.plot(inputDF[Xcol], label = Xcol)
    ax2.plot(inputDF[Ycol], label = Ycol)
    ax2.plot(inputDF[Zcol], label = Zcol)
    # ask matplotlib for the plotted objects and their labels
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc=2)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
    
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
    
def makeNewFig(avgVal1, sdVal1, avgVal2, sdVal2, avgVal3, sdVal3, Ylabel1, Ylabel2, Ylabel3):
    
        # plot ensemble average values from landings defined in a
        # a different function above. Takes the avg and std of the columns
        # as inputs
        fig, (ax1, ax2, ax3) = plt.subplots(1,3)
        
        ax1.plot(x, avgVal1, 'k', color='#00966C')
        ax1.set_xlabel('Time')
        ax1.set_ylabel(f"{Ylabel1}")
        ax1.fill_between(x, avgVal1-sdVal1, avgVal1+sdVal1,
            alpha=0.5, edgecolor='#00966C', facecolor='#00966C')

        ax2.plot(x, avgVal2, 'k', color='#00966C')
        ax2.set_xlabel('Time')
        ax2.set_ylabel(f"{Ylabel2}")
        ax2.fill_between(x, avgVal2-sdVal2, avgVal2+sdVal2,
            alpha=0.5, edgecolor='#00966C', facecolor='#00966C')

        ax3.plot(x, avgVal3, 'k', color='#00966C')
        ax3.set_xlabel('Time')
        ax3.set_ylabel(f"{Ylabel3}")
        ax3.fill_between(x, avgVal3-sdVal3, avgVal3+sdVal3,
            alpha=0.5, edgecolor='#00966C', facecolor='#00966C')

        plt.tight_layout()

### load file in
fName = entries[fileToLoad] #Load one file at a time

dat = pd.read_csv(fPath+fName,sep='\t', skiprows = 8, header = 0)  

dat.ForcesZ = dat.ForcesZ * -1

#### Trim data to begin and end in a flight phase
print('Select start and end of analysis trial')
forceDat = delimitTrial(dat)
#forceThresh = defThreshold(forceDat)
# when COPx is more negative, that is left foot strike

# define a ton and unpack variables
trimmedForce = trimForce(forceDat, fThresh)
XtotalForce = forceDat.ForcesX
YtotalForce = forceDat.ForcesY
AnklePower = forceDat.LAnklePower
KneePower = forceDat.LKneePower
HipPower = forceDat.LHipPower

AnkleFlexion = forceDat.LAnkleFlex
AnkleInversion = forceDat.LAnkleInv
AnkleAbd = forceDat.LAnkleAbd
#AnkleFlexion = dat.LAnkleAngleX
#AnkleInversion = dat.LAnkleAngleY
#AnkleAbd = dat.LAnkleZAngle

KneeFlex = forceDat.LKneeFlex
KneeRot = forceDat.LKneeRot
#KneeFlex = dat.LKneeYAngle
#KneeRot = dat.LKneeXAngle

HipFlex = forceDat.LHipFlex
HipAbd = forceDat.LHipAbd
HipInt = forceDat.LHipInt
#HipInt = forceDat.LHipRot

#HipFlex = dat.LHipXAngle
#HipAbd = dat.LHipYAngle
#HipInt = dat.LHipZAngle


AnkleMomX = forceDat.LAnkleMomentx
AnkleMomY = forceDat.LAnkleMomenty
AnkleMomZ = forceDat.LAnkleMomentz

KneeMomX = forceDat.LKneeMomentX
KneeMomY = forceDat.LKneeMomentY
KneeMomZ = forceDat.LKneeMomentZ

HipMomX = forceDat.LHipMomentx
HipMomY = forceDat.LHipMomenty
HipMomZ = forceDat.LHipMomentz

#find the landings and offs of the FP as vectors from function above
landings = findLandings(trimmedForce, fThresh)
takeoffs = findTakeoffs(trimmedForce, fThresh)
# determine if first step is left or right then delete every other
# landing and takeoff. MORE NEGATIVE IS LEFT
if (np.mean(dat.LCOPx[landings[0]:takeoffs[0]]) < np.mean(dat.LCOPx[landings[1]:takeoffs[1]])):
    trimmedLandings = [i for a, i in enumerate(landings) if  a%2 == 0]
    trimmedTakeoffs = [i for a, i in enumerate(takeoffs) if  a%2 == 0]
else:
    trimmedLandings = [i for a, i in enumerate(landings) if  a%2 != 0]
    trimmedTakeoffs = [i for a, i in enumerate(takeoffs) if  a%2 != 0]

# create an x-axis
x = np.linspace(0,stepLen,stepLen)

# stack data in order to allow ensemble averaging
stackedF = forceMatrix(trimmedForce, trimmedLandings, 10, stepLen)
XforceOut = forceMatrix(XtotalForce, trimmedLandings, 10, stepLen)
YforceOut = forceMatrix(YtotalForce, trimmedLandings, 10, stepLen)

AnklePowerOut = forceMatrix(AnklePower, trimmedLandings, 10, stepLen)
KneePowerOut = forceMatrix(KneePower, trimmedLandings, 10, stepLen)
HipPowerOut = forceMatrix(HipPower, trimmedLandings, 10, stepLen)

#Ankle 
AnkleFlexionOut = forceMatrix(AnkleFlexion, trimmedLandings, 10, stepLen)
AnkleInversionOut = forceMatrix(AnkleInversion, trimmedLandings, 10, stepLen)
AnkleAbdOut = forceMatrix(AnkleAbd, trimmedLandings, 10, stepLen)

AnkleMomXOut = forceMatrix(AnkleMomX, trimmedLandings, 10, stepLen)
AnkleMomYOut = forceMatrix(AnkleMomY, trimmedLandings, 10, stepLen)
AnkleMomZOut = forceMatrix(AnkleMomZ, trimmedLandings, 10, stepLen)

# Knee
KneeFlexionOut = forceMatrix(KneeFlex, trimmedLandings, 10, stepLen)
KneeRotationnOut = forceMatrix(KneeRot, trimmedLandings, 10, stepLen)

KneeMomXOut = forceMatrix(KneeMomX, trimmedLandings, 10, stepLen)
KneeMomYOut = forceMatrix(KneeMomY, trimmedLandings, 10, stepLen)
KneeMomZOut = forceMatrix(KneeMomZ, trimmedLandings, 10, stepLen)

#Hip 
HipFlexionOut = forceMatrix(HipFlex, trimmedLandings, 10, stepLen)
HipIntOut = forceMatrix(HipInt, trimmedLandings, 10, stepLen)
HipAbdOut = forceMatrix(HipAbd, trimmedLandings, 10, stepLen)

HipMomXOut = forceMatrix(HipMomX, trimmedLandings, 10, stepLen)
HipMomYOut = forceMatrix(HipMomY, trimmedLandings, 10, stepLen)
HipMomZOut = forceMatrix(HipMomZ, trimmedLandings, 10, stepLen)

# create mean and SD for each variable
#forces
avgF = np.mean(stackedF, axis = 0)
sdF = np.std(stackedF, axis = 0)

avgFX = np.mean(XforceOut, axis = 0)
sdFX = np.std(XforceOut, axis = 0)

avgFY = np.mean(YforceOut, axis = 0)
sdFY = np.std(YforceOut, axis = 0)
avgFY = avgFY * -1

#Ankle powers
avgAP = np.nanmean(AnklePowerOut, axis = 0)
sdAP = np.nanstd(AnklePowerOut, axis = 0)
#Knee powers
avgKP = np.nanmean(KneePowerOut, axis = 0)
sdKP = np.nanstd(KneePowerOut, axis = 0)
#Hip powers
avgHP = np.nanmean(HipPowerOut, axis = 0) 
sdHP = np.nanstd(HipPowerOut, axis = 0)

#AnkleAngles
avgAnkleFlex = np.mean(AnkleFlexionOut, axis = 0)
sdAnkleFlex = np.std(AnkleFlexionOut, axis = 0)
avgAnkleInv = np.mean(AnkleInversionOut, axis = 0)
sdAnkleInv = np.std(AnkleInversionOut, axis = 0)
avgAnkleAbd = np.mean(AnkleAbdOut, axis = 0)
sdAnkleAbd = np.std(AnkleAbdOut, axis = 0)

#AnkleMoments
avgAnkMomX = np.mean(AnkleMomXOut, axis = 0)
sdAnkleMomX = np.std(AnkleMomXOut, axis = 0)
avgAnkleMomY = np.mean(AnkleMomYOut, axis = 0)
sdAnkleMomY = np.std(AnkleMomYOut, axis = 0)
avgAnkleMomZ = np.mean(AnkleMomZOut, axis = 0)
sdAnkleMomZ = np.std(AnkleMomZOut, axis = 0)

#Knee Angles
avgKneeFlex = np.mean(KneeFlexionOut, axis = 0)
sdKneeFlex = np.std(KneeFlexionOut, axis = 0)
avgKneeInt = np.mean(KneeRotationnOut, axis = 0)
sdKneeInt = np.std(KneeRotationnOut, axis = 0)

#Knee moments
avgKneeMomX = np.mean(KneeMomXOut, axis = 0)
sdKneeMomX = np.std(KneeMomXOut, axis = 0)
avgKneeMomY = np.mean(KneeMomYOut, axis = 0)
sdKneeMomY = np.std(KneeMomYOut, axis = 0)
avgKneeMomZ = np.mean(KneeMomZOut, axis = 0)
sdKneeMomZ = np.std(KneeMomZOut, axis = 0)

# Hip Angles
avgHipFlex = np.mean(HipFlexionOut, axis = 0)
sdHipFlex = np.std(HipFlexionOut, axis = 0)
avgHipInv = np.mean(HipIntOut, axis = 0)
sdHipInv = np.std(HipIntOut, axis = 0)
avgHipAbd = np.mean(HipAbdOut, axis = 0)
sdHipAbd = np.std(HipAbdOut, axis = 0)

#Hip moments
avgHipMomX = np.mean(HipMomXOut, axis = 0)
sdHipMomX = np.std(HipMomXOut, axis = 0)
avgHipMomY = np.mean(HipMomYOut, axis = 0)
sdHipMomY = np.std(HipMomYOut, axis = 0)
avgHipMomZ = np.mean(HipMomZOut, axis = 0)
sdHipMomZ = np.std(HipMomZOut, axis = 0)

# make shaded error bar figures from std and mean defined above 
makeNewFig(avgF, sdF, avgFX, sdFX, avgFY, sdFY, 'Z Force','X Force','Y Force')

makeNewFig(avgAP, sdAP, avgKP, sdKP, avgHP, sdHP,'Ankle Power','Knee Power','Hip Power')

makeNewFig(avgAnkleFlex, sdAnkleFlex, avgAnkleInv,  sdAnkleInv, avgAnkleAbd, sdAnkleAbd, 'Anlke Flexion', 'Ankle Inversion','AnkleAbduction')
makeNewFig(avgAnkMomX, sdAnkleMomX, avgAnkleMomY,  sdAnkleMomY, avgAnkleMomZ, sdAnkleMomZ, 'Anlke X Moment', 'Ankle Y Moment','Ankle Z Moment')

makeNewFig(avgKneeFlex, sdKneeFlex, avgKneeInt,  sdKneeInt, avgAnkleAbd, sdAnkleAbd, 'Knee Flexion', 'Knee Rotation','AnkleAbduction')
makeNewFig(avgKneeMomX, sdKneeMomX, avgKneeMomY, sdKneeMomY, avgKneeMomZ, sdKneeMomZ, 'Knee X Moment', 'Knee Y Moment', 'Knee Z Moment')

makeNewFig(avgHipFlex, sdHipFlex, avgHipInv,  sdHipInv, avgHipAbd, sdHipAbd, 'Hip Flexion', 'Hip Rotation','Hip Abduction')
makeNewFig(avgHipMomX, sdHipMomX, avgHipMomY,  sdHipMomY, avgHipMomZ, sdHipMomZ, 'Hip X Moment', 'Hip Y Moment','Hip Z Moment')


# Full time series data for investigations of raw data if needed 
#makeFig(dat, 'ForcesZ', 'LAnklePower', 'LKneePower', 'LHipPower', 'Powers')

#makeFig(dat, 'ForcesZ', 'LAnkleMomentx', 'LAnkleMomenty', 'LAnkleMomentz', 'Ankle Moments')
# makeFig(dat, 'ForcesZ', 'LAnkleAngleX', 'LAnkleAngleY', 'LAnkleZAngle', 'Ankle Angles')

#makeFig(dat, 'ForcesZ', 'LKneeMomentX', 'LKneeMomentY', 'LKneeMomentZ', 'Knee Moments')
# makeFig(dat, 'ForcesZ', 'LKneeXAngle', 'LKneeYAngle', 'LKneeZAngle', 'Knee Angles')

#makeFig(dat, 'ForcesZ', 'LHipMomentx', 'LHipMomenty', 'LHipMomentz', 'Hip Moments')
#makeFig(dat, 'ForcesZ', 'LHipXAngle', 'LHipYAngle', 'LHipZAngle', 'Hip Angles')

# # Hip X moment: int/ext rotation, Hip Y: Hip Z
# Knee X moment: int/ext rotation, Hip Y moment: ab/adduction, Hip Z moment: flex/extension
# Ankle Y: flex/extension
