# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 11:33:33 2021

@author: Daniel.Feeney
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.signal as sig
import seaborn as sns

# Define constants and options
fThresh = 50; #below this value will be set to 0.
writeData = 0; #will write to spreadsheet if 1 entered
plottingEnabled = 0 #plots the bottom if 1. No plots if 0

# Read in balance file
fPath = 'C:\\Users\\Daniel.Feeney\\Boa Technology Inc\\PFL - General\\HikePilot_2021\\Hike Pilot 2021\Data\\'
entries = os.listdir(fPath)


# list of functions 
# finding landings on the force plate once the filtered force exceeds the force threshold
def findLandings(force):
    lic = []
    for step in range(len(force)-1):
        if force[step] == 0 and force[step + 1] >= fThresh:
            lic.append(step)
    return lic

#Find takeoff from FP when force goes from above thresh to 0
def findTakeoffs(force):
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


#Preallocation
loadingRate = []
peakBrakeF = []
brakeImpulse = []
VLR = []
VLRtwo = []
sName = []
tmpConfig = []
timeP = []
NL = []
PkMed = []
PkLat = []

# Save Time series data in separate DF #
vertForce = []
longConfig = []
longSub = []
timeIndex = []


## Y is ankle sagittal moment
## loop through the selected files
for file in entries:
    try:
        
        fName = file #Load one file at a time
        
        dat = pd.read_csv(fPath+fName,sep='\t', skiprows = 8, header = 0)
        
        fig, ax = plt.subplots()
        ax.plot(dat.ForcesZ, label = 'Right Total Force')
        fig.legend()
        print('Select start and end of analysis trial')
        pts = np.asarray(plt.ginput(2, timeout=-1))
        plt.close()
        # downselect the region of the dataframe you'd like
        dat = dat.iloc[int(np.floor(pts[0,0])) : int(np.floor(pts[1,0])),:]
        dat = dat.reset_index()
        
        #Parse file name into subject and configuration 
        subName = fName.split(sep = "_")[0]
        config = fName.split(sep = "_")[1]
        config = config.split(sep = ' - ')[0]
        #timePoint = fName.split(sep = "_")[3]
        dat.ForcesY = dat.ForcesY.fillna(0) #removing the often NA first 3-10 entries
        
        # Filter force
        forceZ = dat.ForcesZ * -1
        forceZ[forceZ<fThresh] = 0
        brakeForce = dat.ForcesY[0:len(dat)] * -1
        MForce = dat.ForcesX[0:len(dat)] 
        
        
        fs = 200 #Sampling rate
        t = np.arange(len(dat)) / fs
        fc = 20  # Cut-off frequency of the filter
        w = fc / (fs / 2) # Normalize the frequency
        b, a = sig.butter(4, w, 'low')
        brakeForce[0] = 0
        brakeFilt = sig.filtfilt(b, a, brakeForce)
        
        #find the landings and offs of the FP as vectors
        landings = findLandings(forceZ)
        takeoffs = findTakeoffs(forceZ)
                
        
        #For each landing, calculate rolling averages and time to stabilize
    
        for countVar, landing in enumerate(landings):
            try:
               # Define where next zero is
                VLR.append(calcVLR(forceZ, landing, 200))
                nextLanding = findNextZero(brakeFilt[landing:landing+600],600)
                NL.append(nextLanding)
                #stepLen.append(findStepLen(forceZ[landing:landing+800],800))
                brakeImpulse.append(sum(brakeFilt[landing:takeoffs[countVar]]))
                sName.append(subName)
                tmpConfig.append(config)
                #timeP.append(timePoint)
                peakBrakeF.append(calcPeakBrake(brakeFilt,landing, 600))
                PkMed.append(np.max(MForce[landing:takeoffs[countVar]]))
                PkLat.append(np.min(MForce[landing:takeoffs[countVar]]))
                
            except:
                print(landing)
        
    except:
            print(file)

outcomes = pd.DataFrame({'Subject':list(sName), 'Config': list(tmpConfig),'NL':list(NL),'peakBrake': list(peakBrakeF),
                         'brakeImpulse': list(brakeImpulse), 'VLR': list(VLR), 'PkMed':list(PkMed), 'PkLat':list(PkLat)})

#outcomes.to_csv("C:\\Users\\Daniel.Feeney\\Dropbox (Boa)\\Hike Work Research\\Work Pilot 2021/WalkForceComb.csv",mode='a',header=False)


def makeFig(inputDF, forceCol, Xcol, Ycol, Zcol, title):
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    
    color = 'tab:red'
    ax1.set_xlabel('time')
    ax1.set_ylabel('TotalForce(N)', color=color)
    ax1.plot(inputDF.forceCol, color=color, label = 'Total Force')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    
    ax2.set_ylabel(title) 
    ax2.plot(inputDF.Xcol, label = 'X')
    ax2.plot(inputDF.Ycol, label = 'Y')
    ax2.plot(inputDF.Zcol, label = 'Z')
    # ask matplotlib for the plotted objects and their labels
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc=2)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
    

fig2 = plt.figure()
ax1 = fig2.add_subplot(111)

color = 'tab:red'
ax1.set_xlabel('time')
ax1.set_ylabel('TotalForce(N)', color=color)
ax1.plot(dat.ForcesZ, color=color, label = 'Total Force')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

ax2.set_ylabel('Knee Moments') 
ax2.plot(dat.LKneeMomentX, label = 'X')
ax2.plot(dat.LKneeMomentY, label = 'Y')
ax2.plot(dat.LKneeMomentZ, label = 'Z')
# ask matplotlib for the plotted objects and their labels
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2, loc=2)
fig2.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()