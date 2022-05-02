# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 11:38:57 2020

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
lookFwd = 1000
pd.options.mode.chained_assignment = None  # default='warn' set to warn for a lot of warnings
manualTrim = 0 #set to 1 to use ginput

# Read in balance file
fPath = 'C:\\Users\\Daniel.Feeney\\Dropbox (Boa)\\Hike Work Research\\Work Pilot 2021\\WalkForces\\'
fPath = 'C:\\Users\\Daniel.Feeney\\Boa Technology Inc\\PFL - General\\HikePilot_2021\\Hike Pilot 2021\\Data\\Forces TM\\'
fPath = 'C:\\Users\\Daniel.Feeney\\Dropbox (Boa)\\Hike Work Research\\Hike Pilot 2021\\TM\Forces\\'
entries = os.listdir(fPath)


# list of functions 
def findLandings(force):
    """
    The purpose of this function is to determine the landings (foot contacts)
    events on the force plate when the filtered vertical ground reaction force
    exceeds the force threshold

    Parameters
    ----------
    force : list
        vertical ground reaction force. 

    Returns
    -------
    lic : list
        indices of the landings (foot contacts)

    """
    lic = []
    for step in range(len(force)-1):
        if force[step] == 0 and force[step + 1] >= fThresh:
            lic.append(step)
    return lic

def findTakeoffs(force):
    """
    Find takeoff from FP when force goes from above thresh to 0

    Parameters
    ----------
    force : list
        vertical ground reaction force

    Returns
    -------
    lto : list
        indices of the take-offs

    """
    lto = []
    for step in range(len(force)-1):
        if force[step] >= fThresh and force[step + 1] == 0:
            lto.append(step + 1)
    return lto


def calcVLR(force, startVal, lengthFwd, endLoading):
    """
    Function to calculate VLR from 80 and 20% of the max value observed in the 
    first n indices (n defined by lengthFwd).

    Parameters
    ----------
    force : list
        vertical ground reaction force
    startVal : int
        The value to start computing the loading rate from. Typically the first
        index after the landing (foot contact) detection
    lengthFwd : int
        Number of indices to examine forward to compute the loading rate
    endLoading : int
        set to where an impact peak should have occured if there is one and can 
        be biased longer so the for loop doesn't error out
    sampFrq : int
        sample frequency

    Returns
    -------
    VLR
        vertical loading rate

    """ 
    tmpDiff = np.diff(force[startVal:startVal+500])
    
    if next(x for x, val in enumerate( tmpDiff ) 
                      if val < 0) < endLoading:
        maxFindex = next(x for x, val in enumerate( tmpDiff ) 
                      if val < 0)
        maxF = force[startVal + maxFindex]
        eightyPctMax = 0.8 * maxF
        twentyPctMax = 0.2 * maxF
            # find indices of 80 and 20 and calc loading rate as diff in force / diff in time (N/s)
        eightyIndex = next(x for x, val in enumerate(force[startVal:startVal+lengthFwd]) 
                      if val > eightyPctMax) 
        twentyIndex = next(x for x, val in enumerate(force[startVal:startVal+lengthFwd]) 
                      if val > twentyPctMax) 
        VLR = ((eightyPctMax - twentyPctMax) / ((eightyIndex/1000) - (twentyIndex/1000)))
    
    else:
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
    
def calcPeakBrake(force, landing, length):
    """
    Compute the maximum breaking (A/P) force during locomotion.

    Parameters
    ----------
    force : list
        A/P ground reaction force
    landing : int
        the landing (foot contact) of the step/stride of interest
    length : int
        number of indices to look forward for computing the maximum breaking
        force

    Returns
    -------
    minimum of the force (in a function): list

    """
    newForce = np.array(force)
    return min(newForce[landing:landing+length])

def findNextZero(force, length):
    """
    Find the zero-crossing in the A/P ground reaction force that indicates the 
    transition from breaking to propulsion

    Parameters
    ----------
    force : list
        A/P ground reaction force that is already segmented from initial
        contact to take-off
    length : int
        number of indices to look forward for the zero-crossing

    Returns
    -------
    step : int
        number of indicies that the zero crossing occurs

    """
    # Starting at a landing, look forward (after first 45 indices)
    # to the find the next time the signal goes from - to +
    for step in range(length):
        if force[step] <= 0 and force[step + 1] >= 0 and step > 45:
            break
    return step


def delimitTrial(inputDF):
    """
    Function to crop the data

    Parameters
    ----------
    inputDF : dataframe
        Original dataframe

    Returns
    -------
    outputDat: dataframe
        Dataframe that has been cropped based on selection

    """
    print('Select 2 points: the start and end of the trial')#
    fig, ax = plt.subplots()
    ax.plot(inputDF.LForceZ, label = 'Left Force')
    fig.legend()
    pts = np.asarray(plt.ginput(2, timeout=-1))
    plt.close()
    outputDat = inputDF.iloc[int(np.floor(pts[0,0])) : int(np.floor(pts[1,0])),:]
    outputDat = outputDat.reset_index()
    return(outputDat)

def filterForce(inputForce, sampFrq, cutoffFrq):
    """
    Low pass filter the input signal

    Parameters
    ----------
    inputForce : list
        Input signal (e.g. vertical force)
    sampFrq : int
        sampling frequency
    cutoffFrq : int
        low-pass filtering cut-off frequency

    Returns
    -------
    filtForce: list
        the low-pass filtered signal of the "inputForce"

    """
    # low-pass filter the input force signals
    #t = np.arange(len(inputForce)) / sampFrq
    w = cutoffFrq / (sampFrq / 2) # Normalize the frequency
    b, a = sig.butter(4, w, 'low')
    filtForce = sig.filtfilt(b, a, inputForce)
    return(filtForce)
    
def trimForce(inputDF, threshForce):
    """
    Function to zero the vertical force below a threshold

    Parameters
    ----------
    ForceVert : list
        Vertical ground reaction force
    threshForce : float
        Zeroing threshold

    Returns
    -------
    ForceVert: numpy array
        Vertical ground reaction force that has been zeroed below a threshold

    """
    forceTot = inputDF.LForceZ
    forceTot[forceTot<threshForce] = 0
    forceTot = np.array(forceTot)
    return(forceTot)

def trimTakeoffs(landingVec, takeoffVec):
    """
    Function to ensure that the first take-off index is greater than the first 
    landing index

    Parameters
    ----------
    landingVec : list
        indices of the landings
    takeoffVec : list
        indices of the take-offs

    Returns
    -------
    takeoffVec

    """
    if landingVec[0] > takeoffVec[0]:
        takeoffVec.pop(0)
        return(takeoffVec)
    else:
        return(takeoffVec)

#Preallocation
CT = []
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
level = []

## loop through the selected files
for file in entries:
    try:
        
        fName = file #Load one file at a time
        
        #Parse file name into subject and configuration 
        subName = fName.split(sep = "_")[0]
        config = fName.split(sep = "_")[1]
        upDown = fName.split(sep='_')[2]
        upDown = upDown.split(sep = ' - ')[0]
        
        dat = pd.read_csv(fPath+fName,sep='\t', skiprows = 8, header = 0)  
        dat = dat.fillna(0)
        dat.LForceZ = -1 * dat.LForceZ
        dat.LForceZ = filterForce(dat.LForceZ, 1000, 20)
        dat.LForceY = filterForce(dat.LForceY, 1000, 20)
        dat.LForceX = filterForce(dat.LForceX, 1000, 20)
        
        # Trim the trials to a smaller section and threshold force
        if manualTrim == 1:
            print('Select start and end of analysis trial 1')
            forceDat = delimitTrial(dat)
        else: 
            forceDat = dat
            
        forceZ = trimForce(forceDat, fThresh)
        MForce = forceDat.LForceX
        
        # if uphill, make braking force negative, if DH, keep sign
        if upDown == 'UH':
            brakeFilt = forceDat.LForceY * -1
        else:
            brakeFilt = forceDat.LForceY
                
        #find the landings and offs of the FP as vectors
        landings = findLandings(forceZ)
        takeoffs = findTakeoffs(forceZ)
        takeoffs = trimTakeoffs(landings, takeoffs)
        
        #For each landing, calculate rolling averages and time to stabilize
    
        for countVar, landing in enumerate(landings):
            try:
               # Define where next zero is
                VLR.append(calcVLR(forceZ, landing, 1000,600))
                VLRtwo.append( (np.max( np.diff(forceZ[landing+5:landing+50]) )/(1/1000) ) )
                try:
                    CT.append(takeoffs[countVar] - landing)
                except:
                    CT.append(0)
                try:
                    nextLanding = findNextZero( np.array(brakeFilt[landing:landing+lookFwd]),lookFwd )
                    brakeImpulse.append(np.nansum( (brakeFilt[landing:landing+nextLanding]) ))
                except:
                    brakeImpulse.append(0)
                #stepLen.append(findStepLen(forceZ[landing:landing+800],800))
                sName.append(subName)
                tmpConfig.append(config)
                peakBrakeF.append(calcPeakBrake(brakeFilt,landing, lookFwd))
                level.append(upDown)
                try:
                    PkMed.append(np.max(MForce[landing:takeoffs[countVar]]))
                except:
                    PkMed.append(0)
                try:
                    PkLat.append(np.min(MForce[landing:takeoffs[countVar]]))
                except:
                    PkLat.append(0)
                
            except:
                print(landing)
        
    except:
            print(file)

# Place outcome variables in a single variable
outcomes = pd.DataFrame({'Subject':list(sName), 'Config': list(tmpConfig),'Level':list(level),'CT':list(CT),'peakBrake': list(peakBrakeF),
                         'brakeImpulse': list(brakeImpulse), 'VLR': list(VLR), 'PkMed':list(PkMed), 'PkLat':list(PkLat)})

# Save variable to csv
outcomes.to_csv("C:\\Users\\Daniel.Feeney\\Boa Technology Inc\\PFL - General\\HikePilot_2021\\Hike Pilot 2021\\Data\\WalkForceComb.csv")#,mode='a',header=False)

#df2 = pd.DataFrame(pd.concat(vertForce), np.concatenate(longConfig))    

#longDat = pd.concat(vertForce, ignore_index = True)
#longDat2['Config'] = pd.DataFrame(np.concatenate(longConfig))
#longDat['Sub'] = pd.DataFrame(np.concatenate(longSub))
#longDat['TimePoint'] = pd.DataFrame(np.concatenate(timeIndex))

# If outcome plots are desired:
if plottingEnabled == 1:
    outcomes[['peakBrake']] = -1 * outcomes[['peakBrake']]
    outcomes[['PkLat']] = -1 * outcomes[['PkLat']]
    cleanedOutcomes = outcomes[outcomes['brakeImpulse'] <= -1000]
    cleanedOutcomes[['brakeImpulse']] = -1 * cleanedOutcomes[['brakeImpulse']]
        
    
    f, axes = plt.subplots(1,2)
    sns.boxplot(y='peakBrake', x='Subject', hue="Config",
                     data=cleanedOutcomes, 
                     palette="colorblind", ax=axes[0])
    
    sns.boxplot(y='VLR', x='Subject', hue = "Config", 
                     data=cleanedOutcomes, 
                     palette="colorblind", ax=axes[1])
    
    f, axes = plt.subplots(1,2)
    sns.boxplot(y='brakeImpulse', x='Subject', hue = "Config", 
                     data=cleanedOutcomes, 
                     palette="colorblind", ax=axes[0])
    
    sns.boxplot(y='NL', x='Subject', hue = "Config", 
                     data=cleanedOutcomes, 
                     palette="colorblind", ax=axes[1])
    plt.tight_layout()
    
    f, axes = plt.subplots(1,2)
    sns.boxplot(y='PkMed', x='Subject', hue="Config",
                     data=cleanedOutcomes, 
                     palette="colorblind", ax=axes[0])
    sns.boxplot(y='PkLat', x='Subject', hue="Config",
                     data=cleanedOutcomes, 
                     palette="colorblind", ax=axes[1])

#
#newForce = pd.concat(vertForce)
#newSub = pd.concat(longSub)
#newConfig = pd.concat(longConfig)
#newTry = {'vertForce': newForce,
#        'Sub': newSub,
#        'Config': newConfig
#        }
#
#df = pd.DataFrame(newTry, columns = ['Force', 'Sub','Config'])
#### 
#fmri = sns.load_dataset("fmri")
#sns.relplot(
#    data=fmri, kind="line",
#    x="timepoint", y="signal", col="region",
#    hue="event", style="event",
#)

###
