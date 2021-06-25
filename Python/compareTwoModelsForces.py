# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 11:09:55 2021

@author: Daniel.Feeney
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.signal as sig

# Define constants and options

# Read in balance file
fPath = 'C:\\Users\\Daniel.Feeney\\Dropbox (Boa)\\EndurancePerformance\\Altra_MontBlanc_June2021\\'
fPath = 'C:\\Users\\Daniel.Feeney\\Dropbox (Boa)\\Endurance Health Validation\\DU_Running_Summer_2021\\Data\\Forces\\'

entries = os.listdir(fPath)

# load data in
fName = entries[1] #Load one file at a time
fName2 = entries[5]
config1 = fName.split('_')[1]
config2 = fName2.split('_')[1]

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

def delimitTrial(inputDF):
    # generic function to plot and start/end trial #
    fig, ax = plt.subplots()
    ax.plot(inputDF.LForceZ, label = 'Left Force')
    fig.legend()
    pts = np.asarray(plt.ginput(2, timeout=-1))
    plt.close()
    outputDat = inputDF.iloc[int(np.floor(pts[0,0])) : int(np.floor(pts[1,0])),:]
    outputDat = outputDat.reset_index()
    return(outputDat)

def defThreshold(inputDF):
        # find threshold force
        fig, ax = plt.subplots()
        ax.plot(inputDF.LForceZ, label = 'Right Foot Force')
        print('Select a point to represent 0 in the trial')
        pts = np.asarray(plt.ginput(1, timeout=-1))
        plt.close()
        fThresh = pts[0][1]
        return(fThresh)
    
def trimForce(inputDF, threshForce):
    forceTot = inputDF.LForceZ
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

def filterForce(inputForce, sampFrq, cutoffFrq):
        # low-pass filter the input force signals
        #t = np.arange(len(inputForce)) / sampFrq
        w = cutoffFrq / (sampFrq / 2) # Normalize the frequency
        b, a = sig.butter(4, w, 'low')
        filtForce = sig.filtfilt(b, a, inputForce)
        return(filtForce)
        


dat = pd.read_csv(fPath+fName,sep='\t', skiprows = 8, header = 0)
dat = dat.fillna(0)
dat.LForceZ = dat.LForceZ * -1
dat.LForceZ = filterForce(dat.LForceZ, 1000, 20)
dat.LForceY = filterForce(dat.LForceY, 1000, 20)
dat.LForceX = filterForce(dat.LForceX, 1000, 20)

dat2 = pd.read_csv(fPath+fName2,sep='\t', skiprows = 8, header = 0) 
dat2 = dat2.fillna(0)
dat2.LForceZ = dat2.LForceZ * -1
dat2.LForceZ = filterForce(dat2.LForceZ, 1000, 20)
dat2.LForceY = filterForce(dat2.LForceY, 1000, 20)
dat2.LForceX = filterForce(dat2.LForceX, 1000, 20)

# Trim the trials to a smaller section # 
print('Select start and end of analysis trial 1')
forceDat = delimitTrial(dat)
forceThresh = defThreshold(forceDat)
trimmedForce = trimForce(forceDat, forceThresh)
XtotalForce = forceDat.LForceX
YtotalForce = forceDat.LForceY

# Trim the trials to a smaller section # 
print('Select start and end of analysis trial 2')
forceDat2 = delimitTrial(dat2)
forceThresh2 = defThreshold(forceDat2)
trimmedForce2 = trimForce(forceDat2, forceThresh2)
XtotalForce2 = forceDat2.LForceX
YtotalForce2 = forceDat2.LForceY

#find the landings and offs of the FP as vectors from function above
landings = findLandings(trimmedForce, forceThresh)
takeoffs = findTakeoffs(trimmedForce, forceThresh)
#find the landings and offs of the FP as vectors from function above
landings2 = findLandings(trimmedForce2, forceThresh2)
takeoffs2 = findTakeoffs(trimmedForce2, forceThresh2)

stepLen = 250
x = np.linspace(0,stepLen,stepLen)
stackedF = forceMatrix(trimmedForce, landings, 10, stepLen)
XforceOut = forceMatrix(XtotalForce, landings, 10, stepLen)
YforceOut = forceMatrix(YtotalForce, landings, 10, stepLen)

stackedF2 = forceMatrix(trimmedForce2, landings2, 10, stepLen)
XforceOut2 = forceMatrix(XtotalForce2, landings2, 10, stepLen)
YforceOut2 = forceMatrix(YtotalForce2, landings2, 10, stepLen)

# create matrices with average and SD of force trajectories
x = np.linspace(0,stepLen,stepLen)
avgF = np.mean(stackedF, axis = 0)
sdF = np.std(stackedF, axis = 0)
avgF2 = np.mean(stackedF2, axis = 0)
sdF2 = np.std(stackedF2, axis = 0)

avgFX = np.mean(XforceOut, axis = 0)
sdFX = np.std(XforceOut, axis = 0)
avgFX2 = np.mean(XforceOut2, axis = 0)
sdFX2 = np.std(XforceOut2, axis = 0)

avgFY = np.mean(YforceOut, axis = 0)
sdFY = np.std(YforceOut, axis = 0)
avgFY = avgFY * -1
avgFY2 = np.mean(YforceOut2, axis = 0)
sdFY2 = np.std(YforceOut2, axis = 0)
avgFY2 = avgFY2 * -1

#####
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
ax1.plot(x, avgF, 'k', color='#00966C', label = '{}'.format(config1))
ax1.plot(x, avgF2, 'k', color='#000000', label = '{}'.format(config2))
ax1.set_xlabel('Time')
ax1.set_ylabel('Force (N)')
ax1.set_title('Average Vertical Force')
ax1.fill_between(x, avgF-sdF, avgF+sdF,
    alpha=0.5, edgecolor='#00966C', facecolor='#00966C')
ax1.fill_between(x, avgF2-sdF2, avgF2+sdF2,
   alpha=0.5, edgecolor='#000000', facecolor='#000000')
ax2.plot(x, avgFX, 'k', color='#00966C', label = '{}'.format(config1))
ax2.plot(x, avgFX2, 'k', color='#000000', label = '{}'.format(config2))
ax2.set_xlabel('Time')
ax2.set_ylabel('Force (N)')
ax2.set_title('Average M-L Force')
ax2.fill_between(x, avgFX2-sdFX, avgFX+sdFX,
    alpha=0.5, edgecolor='#00966C', facecolor='#00966C')
ax2.fill_between(x, avgFX2-sdFX2, avgFX2+sdFX2,
    alpha=0.5, edgecolor='#000000', facecolor='#000000')
ax2.legend(loc = 'right')
ax3.plot(x, avgFY, 'k', color='#00966C', label = '{}'.format(config1))
ax3.plot(x, avgFY2, 'k', color='#000000', label = '{}'.format(config2))
ax3.set_xlabel('Time')
ax3.set_ylabel('Force (N)')
ax3.set_title('Average A-P Force')
ax3.fill_between(x, avgFY-sdFY, avgFY+sdFY,
    alpha=0.5, edgecolor='#00966C', facecolor='#00966C')
ax3.fill_between(x, avgFY2-sdFY2, avgFY2+sdFY2,
    alpha=0.5, edgecolor='#000000', facecolor='#000000')
plt.tight_layout()