# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 11:38:57 2020
Analyzing the force data from our Bertec duel belt Treadmill
Calculates relevent metrics for running and walking with optional parameters 
at the top. 
manualTrim = 1 means you want to plot each force time series 
and select when the trial starts and ends
plottingEnabled will show a plot for each iteration (not recommended)
fThresh is the force threshold to set force to 0 
@author: Daniel.Feeney
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.signal as sig


# Define constants and options
run = 0 # Set this to 1 where participant is running on one belt so only the left are detected. 0 for dual belt
manualTrim = 0  #set this to 1 if you want to manually trim trials with ginput, 0 if you want it auto trimmed (start and end of trial)
fThresh = 50 #below this value will be set to 0.
writeData = 0 #will write to spreadsheet if 1 entered
plottingEnabled = 1 #plots the bottom if 1. No plots if 0
lookFwd = 50
timeToLoad = 150 #length to look forward for an impact peak
pd.options.mode.chained_assignment = None  # default='warn' set to warn for a lot of warnings
bigData = 0
# Read in balance file
fPath = 'C:\\Users\\eric.honert\\Boa Technology Inc\\PFL Team - General\\Testing Segments\\WorkWear_Performance\\Elten_Jan2022\\Treadmill\\Forces\\'
entries = os.listdir(fPath)

if bigData == 1:
    ## need to be modified for each test!
    Shoe = 'MontBlanc'
    Brand = 'Altra'
    Year = '2021'
    Month = 'June'
    ##
#______________________________________________________________________________
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
            
    if lic[0] == 0:
        lic = lic[1:]
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
    lto = lto[1:]
    return lto


def calcVLR(force, startVal, lengthFwd, endLoading, sampFrq):
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
    
    tmpDiff = np.diff(force[startVal:startVal+500])*sampFrq
    
    # If there is an impact peak, utilize it to compute the loading rate
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
        VLR = ((eightyPctMax - twentyPctMax) / ((eightyIndex/sampFrq) - (twentyIndex/sampFrq)))
    
    # If there is no impact peak, utilize the endLoading to compute the loading rate
    else:
        maxF = np.max(force[startVal:startVal+endLoading])
        eightyPctMax = 0.8 * maxF
        twentyPctMax = 0.2 * maxF
        # find indices of 80 and 20 and calc loading rate as diff in force / diff in time (N/s)
        eightyIndex = next(x for x, val in enumerate(force[startVal:startVal+endLoading]) 
                          if val > eightyPctMax) 
        twentyIndex = next(x for x, val in enumerate(force[startVal:startVal+endLoading]) 
                          if val > twentyPctMax) 
        VLR = ((eightyPctMax - twentyPctMax) / ((eightyIndex/sampFrq) - (twentyIndex/sampFrq)))
        
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
    print('Select 2 points: the start and end of the trial')
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
    
def trimForce(ForceVert, threshForce):
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
    ForceVert[ForceVert<threshForce] = 0
    ForceVert = np.array(ForceVert)
    return(ForceVert)

def trimLandings(landingVec, takeoffVec):
    """
    Function to ensure that the first landing index is greater than the first 
    take-off index

    Parameters
    ----------
    landingVec : list
        indices of the landings
    takeoffVec : list
        indices of the take-offs

    Returns
    -------
    landingVec: list
        updated indices of the landings

    """
    if landingVec[0] > takeoffVec[0]:
        landingVec.pop(0)
        return(landingVec)
    else:
        return(landingVec)
    
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
    
    # preallocate matrix for force and fill in with force data
def forceMatrix(inputForce, landings, noSteps, stepLength):
    """
    Create a matrix that contains the clipped ground reaction force based on
    initial contact (or could be another variable)

    Parameters
    ----------
    inputForce : list
        Input variable that is of interest
    landings : list
        initial contact indices
    noSteps : int
        desired number of steps to examine
    stepLength : int
        Length (of frames) of interest

    Returns
    -------
    preForce : numpy array
        Input data segmented into similar length steps

    """
    #input a force signal, return matrix with n rows (for each landing) by m col
    #for each point in stepLen.
    preForce = np.zeros((noSteps,stepLength))
    
    for iterVar, landing in enumerate(landings):
        try:
            preForce[iterVar,] = inputForce[landing:landing+stepLength]
        except:
            print(landing)
            
    return preForce
#______________________________________________________________________________
#Preallocation
loadingRate = []
peakBrakeF = []
brakeImpulse = []
VALRs = []
VLRtwo = []
pkForce = []
sName = []
tmpConfig = []
timeP = []
NL = []
PkMed = []
PkLat = []
CTs = []
meanForce = []
propImpulse = []
shoes = []
months = []
brands = []
years = []
segment = []
shoeCondition = []

forceTime = np.zeros((len(entries),210))
subShort = []
configShort = []

## loop through the selected files
for loop, file in enumerate(entries):
    try:
        fName = file #Load one file at a time
        
        #Parse file name into subject and configuration 
        subName = fName.split(sep = "_")[0]
        config = fName.split(sep = "_")[2]
        if bigData == 1:
            config = fName.split(sep = "_")[2].split(' - ')[0]
        
        # Extract data frequency
        freq = pd.read_csv(fPath+fName,sep='\t',usecols=[0], nrows=1, skiprows=[0,1], header = 0)
        freq = freq.values.tolist()
        freq = freq[0][0]
            
        dat = pd.read_csv(fPath+fName,sep='\t', skiprows = 8, header = 0)
        dat = dat.fillna(0)
        dat.LForceZ = -1 * dat.LForceZ
        
        # Trim the trials to a smaller section and threshold force
        if manualTrim == 1:
            print('Select start and end of analysis trial 1')
            forceDat = delimitTrial(dat)
        else: 
            forceDat = dat
        
        MForce = forceDat.LForceX
        brakeFilt = forceDat.LForceY * -1        
        forceZ = trimForce(forceDat.LForceZ, fThresh)
        
                
        #find the landings and takeoffs of the FP as vectors
        landings = findLandings(forceZ)
        takeoffs = findTakeoffs(forceZ)

        takeoffs = trimTakeoffs(landings, takeoffs)
        # determine if first step is left or right then delete every other
        # landing and takeoff. MORE NEGATIVE IS LEFT
        if run == 1:
            if (np.mean( abs(dat.LCOPx[landings[0]:takeoffs[0]]) ) > np.mean( abs(dat.LCOPx[landings[1]:takeoffs[1]])) ): #if landing 0 is left, keep all evens
                trimmedLandings = [i for a, i in enumerate(landings) if  a%2 == 0]
                trimmedTakeoffs = [i for a, i in enumerate(takeoffs) if  a%2 == 0]
            else: #keep all odds
                trimmedLandings = [i for a, i in enumerate(landings) if  a%2 != 0]
                trimmedTakeoffs = [i for a, i in enumerate(takeoffs) if  a%2 != 0]
        else:
            trimmedTakeoffs = takeoffs
            if landings[-1] > trimmedTakeoffs[-1]:
                trimmedLandings = landings[0:-2]
            else:
                trimmedLandings = landings
        
        ## check to make sure brake force is applied in the correct direction ##
        if np.mean(brakeFilt[landings[1]:landings[1]+100]) > 0:
            brakeFilt = -1 * brakeFilt
        #For each landing, calculate rolling averages and time to stabilize
        # ### Start SPM test ###
        # if int(fName.split(' - ')[0].split('_')[2]) == 1:
        #     stackedF = forceMatrix(forceZ, trimmedLandings, len(trimmedLandings), 210)
        #     forceTime[loop,:] = np.mean(stackedF, axis = 0)
        #     subShort.append(int(subName.split('S')[1]))
        #     if config == 'SL':
        #         configShort.append(0)
        #     elif config == 'SD':
        #         configShort.append(1)
        #     else:
        #         configShort.append(2)
            
        
        ### end test ###
        for countVar, landing in enumerate(trimmedLandings):
            try:
               # Define where next zero is
                VALRs.append(calcVLR(forceZ, landing+1, 150, timeToLoad, freq))
                VLRtwo.append( (np.max( np.diff(forceZ[landing+5:landing+150]) )/(1/freq) ) )
                try:
                    CTs.append(trimmedTakeoffs[countVar] - landing)
                except:
                    CTs.append(0)
                try:
                    brakeImpulse.append( sum(i for i in brakeFilt[landing:trimmedTakeoffs[countVar]] if i < 0)/freq ) #sum all negative brake force vals
                    propImpulse.append( sum(i for i in brakeFilt[landing:trimmedTakeoffs[countVar]] if i > 0)/freq ) #sum all positive values
                except:
                    brakeImpulse.append(0)
                    propImpulse.append(0)
                #stepLen.append(findStepLen(forceZ[landing:landing+800],800))
                sName.append(subName)
                tmpConfig.append(config)
                
                peakBrakeF.append(calcPeakBrake(brakeFilt,landing, lookFwd))
                try:
                    meanForce.append( np.mean(forceZ[landing:trimmedTakeoffs[countVar]]))
                except:
                    meanForce.append(0)
                try:
                    pkForce.append( np.max(forceZ[landing:trimmedTakeoffs[countVar]]) )
                except:
                    pkForce.append(0)
                try:
                    PkMed.append(np.max(MForce[landing:trimmedTakeoffs[countVar]]))
                except:
                    PkMed.append(0)
                try:
                    PkLat.append(np.min(MForce[landing:trimmedTakeoffs[countVar]]))
                except:
                    PkLat.append(0)
                if bigData:
                    shoes.append(Shoe)
                    months.append(Month)
                    years.append(Year)
                    brands.append(Brand)
                    segment.append('trail')
                    if config == 'Lace':
                        shoeCondition.append('Lace')
                    else:
                        shoeCondition.append('BOA')
                            
            except:
                print(landing)
        
    except:
            print(file)

outcomes = pd.DataFrame({'Subject':list(sName), 'Config': list(tmpConfig),'pBF': list(peakBrakeF),
                         'brakeImpulse': list(brakeImpulse), 'VALR': list(VALRs), 'VILR':list(VLRtwo),'pMF':list(PkMed),
                         'pLF':list(PkLat), 'CT':list(CTs),'pVGRF':list(pkForce), 'meanForce':list(meanForce),
                         'PropImp':list(propImpulse)})

outcomes.to_csv("C:\\Users\\eric.honert\\Boa Technology Inc\\PFL Team - General\\WorkWear_Performance\\Elten_Jan2022\\Treadmill\\Forces\\Forces2.csv",mode='a',header=True)

if bigData == 1:
    outcomes = pd.DataFrame({'Subject':list(sName), 'ShoeCondition':list(shoeCondition),'Config': list(tmpConfig),
                             'Segment':list(segment), 'Shoe':list(shoes), 'Brand':list(brands),'year':list(years),
                             'month':list(months), 'VALR': list(VALRs), 'VILR':list(VLRtwo),'pBF': list(peakBrakeF),
                         'pMF':list(PkMed),'pLF':list(PkLat), 'CT':list(CTs)})
    
    outcomes.to_csv('C:\\Users\\daniel.feeney\\Boa Technology Inc\\PFL - General\\BigData2021\\BigDataRun.csv',mode='a',header=False)
# not currently working. Need to make this a 2 way Anova (subject and config). Getting close
# import spm1d
# forceTime = forceTime[~np.all(forceTime == 0, axis=1)]
# forceTime = forceTime[:,1:]
# F = spm1d.stats.anova1rm(forceTime, np.array(configShort), np.array(subShort))
# Fi = F.inference(alpha=0.05)
# Fi.plot()
