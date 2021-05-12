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

# list of functions 
# finding landings on the force plate once the filtered force exceeds the force threshold
def findLandings(force):
    lic = []
    for step in range(len(force)-1):
        if force[step] == 0 and force[step + 1] >= 0:
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
    #for each point in stepLen
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


#find the landings and offs of the FP as vectors from function above
landings = findLandings(trimmedForce)
takeoffs = findTakeoffs(trimmedForce)

stepLen = 40
x = np.linspace(0,stepLen,stepLen)
stackedF = forceMatrix(trimmedForce, landings, 5, stepLen)

# create matrices with average and SD of force trajectories
x = np.linspace(0,stepLen,stepLen)
avgF = np.mean(stackedF, axis = 0)
sdF = np.std(stackedF, axis = 0)
#Plot force
plt.plot(x, avgF, 'k', color='#CC4F1B')
plt.xlabel('Time')
plt.ylabel('Force (N)')
plt.title('Treadmill vertical force')
plt.fill_between(x, avgF-sdF, avgF+sdF,
    alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
plt.ylim([0,2200])
   