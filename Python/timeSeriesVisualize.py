# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 09:45:18 2020
Visualize shadded errorbars
@author: Daniel.Feeney
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.signal as sig
import seaborn as sns

# Define constants and options

# Read in balance file
fPath = 'C:\\Users\\Daniel.Feeney\\Dropbox (Boa)\\EndurancePerformance\\Altra_MontBlanc_Jan2021\\TMdata\\'
entries = os.listdir(fPath)
fThresh = 80

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
        if force[step] >= 0 and force[step + 1] == 0:
            lto.append(step + 1)
    return lto

def delimitTrial(inputDF):
    # generic function to plot and start/end trial #
    fig, ax = plt.subplots()
    ax.plot(inputDF.LForceZ, label = 'Left Force')
    fig.legend()
    pts = np.asarray(plt.ginput(2, timeout=-1))
    plt.close()
    outputDat = dat.iloc[int(np.floor(pts[0,0])) : int(np.floor(pts[1,0])),:]
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

## Load file in
fName = entries[1] #Load one file at a time
        
dat = pd.read_csv(fPath+fName,sep='\t', skiprows = 8, header = 0)
dat.LForceZ = dat.LForceZ * -1

print('Select start and end of analysis trial')
forceDat = delimitTrial(dat)
forceThresh = defThreshold(forceDat)
trimmedForce = trimForce(forceDat, forceThresh)
XtotalForce = dat.LForceX
YtotalForce = dat.LForceY

#find the landings and offs of the FP as vectors from function above
landings = findLandings(trimmedForce)
takeoffs = findTakeoffs(trimmedForce)

stepLen = 30
x = np.linspace(0,stepLen,stepLen)
stackedF = forceMatrix(trimmedForce, landings, 10, stepLen)
XforceOut = forceMatrix(XtotalForce, landings, 10, stepLen)
YforceOut = forceMatrix(YtotalForce, landings, 10, stepLen)

# create matrices with average and SD of force trajectories
x = np.linspace(0,stepLen,stepLen)
avgF = np.mean(stackedF, axis = 0)
sdF = np.std(stackedF, axis = 0)

avgFX = np.mean(XforceOut, axis = 0)
sdFX = np.std(XforceOut, axis = 0)

avgFY = np.mean(YforceOut, axis = 0)
sdFY = np.std(YforceOut, axis = 0)
avgFY = avgFY * -1

#####
fig, (ax1, ax2, ax3) = plt.subplots(3)
ax1.plot(x, avgF, 'k', color='#00966C', label = 'Z Force')
#ax1.plot(x, avgF3, 'k', color='#000000', label = '{}'.format(config3))
ax1.set_xlabel('Time')
ax1.set_ylabel('Force (N)')
ax1.set_title('Average Vertical Force')
ax1.fill_between(x, avgF-sdF, avgF+sdF,
    alpha=0.5, edgecolor='#00966C', facecolor='#00966C')
#ax1.fill_between(x, avgF3-sdF3, avgF3+sdF3,
#    alpha=0.5, edgecolor='#000000', facecolor='#000000')

ax2.plot(x, avgFX, 'k', color='#00966C', label = 'X Force')
#ax2.plot(x, avgXF3, 'k', color='#000000', label = '{}'.format(config3))
ax2.set_xlabel('Time')
ax2.set_ylabel('Force (N)')
ax2.set_title('Average A-P Force')
ax2.fill_between(x, avgFX-sdFX, avgFX+sdFX,
    alpha=0.5, edgecolor='#00966C', facecolor='#00966C')
#ax2.fill_between(x, avgXF3-sdXF3, avgXF3+sdXF3,
#    alpha=0.5, edgecolor='#000000', facecolor='#000000')
ax2.legend(loc = 'right')
ax3.plot(x, avgFY, 'k', color='#00966C', label = 'Y Force')
#ax3.plot(x, avgYF3, 'k', color='#000000', label = '{}'.format(config3))
ax3.set_xlabel('Time')
ax3.set_ylabel('Force (N)')
ax3.set_title('Average A-P Force')
ax3.fill_between(x, avgFY-sdFY, avgFY+sdFY,
    alpha=0.5, edgecolor='#00966C', facecolor='#00966C')
#ax3.fill_between(x, avgYF3-sdYF3, avgYF3+sdYF3,
#    alpha=0.5, edgecolor='#000000', facecolor='#000000')
#plt.suptitle('{} Skater Jump'.format(subName))
plt.tight_layout()