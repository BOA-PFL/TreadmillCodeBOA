# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 11:09:55 2021
Use this to compare the time series in X,Y,and Z directions from treadmill
running. 
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
fName = entries[148] #Load one file at a time
fName2 = entries[150]
fName3 = entries[152]
config1 = fName.split('_')[1]
config2 = fName2.split('_')[1]
config3 = fName3.split('_')[1]
fThresh = 50
run = 1
stepLen = 200

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

dat3 = pd.read_csv(fPath+fName2,sep='\t', skiprows = 8, header = 0) 
dat3 = dat3.fillna(0)
dat3.LForceZ = dat2.LForceZ * -1
dat3.LForceZ = filterForce(dat2.LForceZ, 1000, 20)
dat3.LForceY = filterForce(dat2.LForceY, 1000, 20)
dat3.LForceX = filterForce(dat2.LForceX, 1000, 20)

# Trim the trials to a smaller section # 
#print('Select start and end of analysis trial 1')
forceDat = dat#delimitTrial(dat)
trimmedForce = trimForce(forceDat, fThresh)
XtotalForce = forceDat.LForceX
YtotalForce = forceDat.LForceY

# Trim the trials to a smaller section # 
#print('Select start and end of analysis trial 2')
forceDat2 = dat2#delimitTrial(dat2)
trimmedForce2 = trimForce(forceDat2, fThresh)
XtotalForce2 = forceDat2.LForceX
YtotalForce2 = forceDat2.LForceY

forceDat3 = dat3
trimmedForce3 = trimForce(forceDat3, fThresh)
XtotalForce3 = forceDat3.LForceX
YtotalForce3 = forceDat3.LForceY

#find the landings and offs of the FP as vectors from function above
landings = findLandings(trimmedForce, fThresh)
takeoffs = findTakeoffs(trimmedForce, fThresh)
#find the landings and offs of the FP as vectors from function above
landings2 = findLandings(trimmedForce2, fThresh)
takeoffs2 = findTakeoffs(trimmedForce2, fThresh)

landings3 = findLandings(trimmedForce3, fThresh)
takeoffs3 = findTakeoffs(trimmedForce3, fThresh)

#### start
if run == 1:
    if ( np.mean( abs(dat.LCOPx[landings[0]:takeoffs[0]]) ) > np.mean( abs(dat.LCOPx[landings[1]:takeoffs[1]]) ) ): #if landing 0 is left, keep all evens
        trimmedLandings = [i for a, i in enumerate(landings) if  a%2 == 0]
        trimmedTakeoffs = [i for a, i in enumerate(takeoffs) if  a%2 == 0]
    else: #keep all odds
        trimmedLandings = [i for a, i in enumerate(landings) if  a%2 != 0]
        trimmedTakeoffs = [i for a, i in enumerate(takeoffs) if  a%2 != 0]
else:
    trimmedLandings2 = landings
    trimmedTakesoffs2 = takeoffs
    
if run == 1:
    if ( np.mean( abs(dat2.LCOPx[landings2[0]:takeoffs2[0]]) ) > np.mean( abs(dat2.LCOPx[landings2[1]:takeoffs2[1]]) ) ): #if landing 0 is left, keep all evens
        trimmedLandings2 = [i for a, i in enumerate(landings2) if  a%2 == 0]
        trimmedTakeoffs2 = [i for a, i in enumerate(takeoffs2) if  a%2 == 0]
    else: #keep all odds
        trimmedLandings2 = [i for a, i in enumerate(landings2) if  a%2 != 0]
        trimmedTakeoffs2 = [i for a, i in enumerate(takeoffs2) if  a%2 != 0]
else:
    trimmedLandings2 = landings2
    trimmedTakesoffs2 = takeoffs2
    
if run == 1:
    if ( np.mean( abs(dat3.LCOPx[landings3[0]:takeoffs3[0]]) ) > np.mean( abs(dat3.LCOPx[landings3[1]:takeoffs3[1]]) ) ): #if landing 0 is left, keep all evens
        trimmedLandings3 = [i for a, i in enumerate(landings3) if  a%2 == 0]
        trimmedTakeoffs3 = [i for a, i in enumerate(takeoffs3) if  a%2 == 0]
    else: #keep all odds
        trimmedLandings3 = [i for a, i in enumerate(landings3) if  a%2 != 0]
        trimmedTakeoffs3 = [i for a, i in enumerate(takeoffs3) if  a%2 != 0]
else:
    trimmedLandings2 = landings3
    trimmedTakesoffs2 = takeoffs3

#### end

x = np.linspace(0,stepLen,stepLen)
stackedF = forceMatrix(trimmedForce, trimmedLandings, 20, stepLen)
XforceOut = forceMatrix(XtotalForce, trimmedLandings, 20, stepLen)
YforceOut = forceMatrix(YtotalForce, trimmedLandings, 20, stepLen)

stackedF2 = forceMatrix(trimmedForce2, trimmedLandings2, 20, stepLen)
XforceOut2 = forceMatrix(XtotalForce2, trimmedLandings2, 20, stepLen)
YforceOut2 = forceMatrix(YtotalForce2, trimmedLandings2, 20, stepLen)

stackedF3 = forceMatrix(trimmedForce3, trimmedLandings3, 20, stepLen)
XforceOut3 = forceMatrix(XtotalForce3, trimmedLandings3, 20, stepLen)
YforceOut3 = forceMatrix(YtotalForce3, trimmedLandings3, 20, stepLen)

# create matrices with average and SD of force trajectories
x = np.linspace(0,stepLen,stepLen)
avgF = np.mean(stackedF, axis = 0)
sdF = np.std(stackedF, axis = 0)
avgF2 = np.mean(stackedF2, axis = 0)
sdF2 = np.std(stackedF2, axis = 0)
avgF3 = np.mean(stackedF3, axis = 0)
sdF3 = np.std(stackedF3, axis = 0)

avgFX = np.mean(XforceOut, axis = 0)
sdFX = np.std(XforceOut, axis = 0)
avgFX2 = np.mean(XforceOut2, axis = 0)
sdFX2 = np.std(XforceOut2, axis = 0)
avgFX3 = np.mean(XforceOut3, axis = 0)
sdFX3 = np.std(XforceOut3, axis = 0)

avgFY = np.mean(YforceOut, axis = 0)
sdFY = np.std(YforceOut, axis = 0)
avgFY = avgFY * -1
avgFY2 = np.mean(YforceOut2, axis = 0)
sdFY2 = np.std(YforceOut2, axis = 0)
avgFY2 = avgFY2 * -1
avgFY3 = np.mean(YforceOut3, axis = 0)
sdFY3 = np.std(YforceOut3, axis = 0)
avgFY3 = avgFY3 * -1

#####
SMALL_SIZE = 18
MEDIUM_SIZE = 20
BIGGER_SIZE = 22

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(20,11))
ax1.plot(x, avgF, 'k', color='#000000', label = '{}'.format(config1))
ax1.plot(x, avgF2, 'k', color='#DC582A', label = '{}'.format(config2))
ax1.plot(x, avgF3, 'k', color='#CAF0E4', label = '{}'.format(config3))
ax1.set_xlabel('Time')
ax1.set_ylabel('Force (N)')
ax1.set_title('Average Vertical Force')
ax1.fill_between(x, avgF-sdF, avgF+sdF,
    alpha=0.5, edgecolor='#000000', facecolor='#000000')
ax1.fill_between(x, avgF2-sdF2, avgF2+sdF2,
   alpha=0.5, edgecolor='#DC582A', facecolor='#DC582A')
ax1.fill_between(x, avgF3-sdF3, avgF2+sdF2,
   alpha=0.5, edgecolor='#CAF0E4', facecolor='#CAF0E4')
ax2.plot(x, avgFX, 'k', color='#000000', label = '{}'.format(config1))
ax2.plot(x, avgFX2, 'k', color='#DC582A', label = '{}'.format(config2))
ax2.plot(x, avgFX3, 'k', color='#CAF0E4', label = '{}'.format(config3))
ax2.set_xlabel('Time')
ax2.set_ylabel('Force (N)')
ax2.set_title('Average M-L Force')
ax2.fill_between(x, avgFX2-sdFX, avgFX+sdFX,
    alpha=0.5, edgecolor='#000000', facecolor='#000000')
ax2.fill_between(x, avgFX2-sdFX2, avgFX2+sdFX2,
    alpha=0.5, edgecolor='#DC582A', facecolor='#DC582A')
ax2.fill_between(x, avgFX3-sdFX3, avgFX3+sdFX3,
    alpha=0.5, edgecolor='#CAF0E4', facecolor='#CAF0E4')
ax2.legend(loc = 'upper right')
ax3.plot(x, avgFY, 'k', color='#000000', label = '{}'.format(config1))
ax3.plot(x, avgFY2, 'k', color='#DC582A', label = '{}'.format(config2))
ax3.plot(x, avgFY3, 'k', color='#CAF0E4', label = '{}'.format(config3))
ax3.set_xlabel('Time')
ax3.set_ylabel('Force (N)')
ax3.set_title('Average A-P Force')
ax3.fill_between(x, avgFY-sdFY, avgFY+sdFY,
    alpha=0.5, edgecolor='#000000', facecolor='#000000')
ax3.fill_between(x, avgFY2-sdFY2, avgFY2+sdFY2,
    alpha=0.5, edgecolor='#DC582A', facecolor='#DC582A')
ax3.fill_between(x, avgFY3-sdFY3, avgFY3+sdFY3,
    alpha=0.5, edgecolor='#CAF0E4', facecolor='#CAF0E4')
plt.tight_layout()
