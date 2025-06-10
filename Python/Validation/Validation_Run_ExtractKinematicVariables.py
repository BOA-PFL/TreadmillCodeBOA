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

# Define constants and options
manualTrim = 0
runTrial = 1
fThresh = 80
writeData = 0; #will write to spreadsheet if 1 entered
stepLen = 80
x = np.linspace(0,stepLen,stepLen)
pd.options.mode.chained_assignment = None  # default='warn' set to warn for a lot of warnings

# Read in balance file
#fPath = 'C:\\Users\\Daniel.Feeney\\Dropbox (Boa)\\Hike Work Research\\Hike Pilot 2021\\TM\Kinetics\\'
fPath = 'C:\\Users\\Eric.Honert\\Dropbox (Boa)\\Endurance Health Validation\\DU_Running_Summer_2021\\Data\\KineticsKinematics\\'
#fPath = fPath = 'C:\\Users\\daniel.feeney\\Boa Technology Inc\\PFL - General\\AgilityPerformanceData\\BOA_InternalStrap_July2021\\KineticsKinematics\\TM\\'

fileExt = r".txt"
entries = [fName for fName in os.listdir(fPath) if fName.endswith(fileExt)]


# list of functions 
def findLandings(force, fThresh):
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

def findTakeoffs(force, fThresh):
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
        if force[step] <= 0 and force[step + 1] >= 0 and step > 15:
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
    
    ax1 = fig.add_subplot(111)
    
    color = 'tab:red'
    ax1.set_xlabel('time')
    ax1.set_ylabel('TotalForce(N)', color=color)
    ax1.plot(inputDF.ForcesZ, color=color, label = 'Total Force')
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.plot(inputDF.LAnklePower, label = 'Ankle Power')
    pts = np.asarray(plt.ginput(1, timeout=-1))
    plt.close()
    outputDat = inputDF.iloc[int(np.floor(pts[0,0])) : int(np.floor(pts[0,0])+2000),:]
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
    inputDF : dataframe
        dataframe containing all variables, including vertical (Z) GRF
    threshForce : float
        Zeroing threshold

    Returns
    -------
    ForceVert: numpy array
        Vertical ground reaction force that has been zeroed below a threshold

    """
    forceTot = inputDF.ForcesZ
    forceTot[forceTot<threshForce] = 0
    forceTot = np.array(forceTot)
    return(forceTot)

def defThreshold(inputDF):
    """
    Function to define a force threshold, below which the
    forces can be set to zero using "trimForce"

    Parameters
    ----------
    inputDF : dataframe
        Dataframe containing all collected data for a trial

    Returns
    -------
    fThresh : list(float)
        Selected threshold from the interactive plot

    """
    # find threshold force
    fig, ax = plt.subplots()
    ax.plot(inputDF.ForcesZ, label = 'Right Foot Force')
    print('Select a point to represent 0 in the trial')
    pts = np.asarray(plt.ginput(1, timeout=-1))
    plt.close()
    fThresh = pts[0][1]
    return(fThresh)

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
    if landingVec[len(landingVec)-1] > takeoffVec[len(landingVec)-1]:
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

## Variable extraction
loadingRate = []
peakBrakeF = []
brakeImpulse = []
VLR = []
VLRtwo = []

pkAnklePw = []
ankleNetWork = []
anklePosWork = []
ankleNegWork = []
pkKneePw = []
kneeNetWork = []
kneeNegWork = []
kneePosWork = []
pkHipPw = []
hipNetWork = []
hipNegWork = []
hipPosWork = []
hipNetWork = []

pkAnkleInv = []
pkAnkleAbd = []
pkAnkleFlex = []
ankleFlexROM = []
ankleAbdROM = []
ankleInvROM = []

pkKneeFlex = []
pkKneeRot = []

pkHipFlex = []
pkHipAbd = []
pkHipRot = []

pkAnkleMomX = []
pkAnkleMomY = []
pkAnkleMomZ = []
minAnkleMomX = []
minAnkleMomY = []
minAnkleMomZ = []

pkKneeMomX = []
pkKneeMomY = []
pkKneeMomZ = []
minKneeMomX = []
minKneeMomY = []
minKneeMomZ = []

pkHipMomX = []
pkHipMomY = []
pkHipMomZ = []
minHipMomX = []
minHipMomY = []
minHipMomZ = []

hipRotROM= []
hipAbdROM = []
hipFlexROM = []
kneeRotROM = []
kneeFlexROM = []

sName = []
tmpConfig = []
timeP = []
NL = []
PkMed = []
PkLat = []

kneeXROM = []
kneeZROM = []

# Index through all files within the selected folder
for fName in entries:
    try:
        # Load the data        
        dat = pd.read_csv(fPath+fName,sep='\t', skiprows = 8, header = 0)  
        
        dat.ForcesZ = dat.ForcesZ * -1
        
        #### Trim data to begin and end in a flight phase
                # Trim the trials to a smaller section and threshold force
        if manualTrim == 1:
            print('Select start and end of analysis trial 1')
            forceDat = delimitTrial(dat)
        else: 
            forceDat = dat

        trimmedForce = trimForce(forceDat, fThresh)
        MForce = forceDat.ForcesX
        brakeFilt = forceDat.ForcesY * -1
        
        #Parse file name into subject and configuration 
        subName = fName.split(sep = "_")[0]
        config = fName.split(sep = "_")[1]
        config = config.split(sep = ' - ')[0]
        
        # define a ton and unpack variables

        XtotalForce = forceDat.ForcesX
        YtotalForce = forceDat.ForcesY
        AnklePower = forceDat.LAnklePower
        KneePower = forceDat.LKneePower
        HipPower = forceDat.LHipPower
        
        AnkleFlexion = forceDat.LAnkleFlex
        AnkleInversion = forceDat.LAnkleInv
        AnkleAbd = forceDat.LAnkleAbd
        
        KneeFlex = forceDat.LKneeFlex
        KneeRot = forceDat.LKneeRot
        
        HipFlex = forceDat.LHipFlex
        HipAbd = forceDat.LHipAbd
        try:
            HipInt = forceDat.LHipInt
        except:
            HipInt = forceDat.LHipRot
            
        KneeXAng = forceDat.LKneeXAngle #X is Transverse
        KneeYAng = forceDat.LKneeYAngle #Y is saggital
        KneeZAng = forceDat.LKneeZAngle #Z is Frontal
        
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
        ##
        #landings = trimLandings(landings, takeoffs)
        takeoffs = trimTakeoffs(landings, takeoffs)
        
        
        if runTrial == 1:
            #Find the first step with a true ankle power

            if (np.max(dat.LAnklePower[landings[0]:takeoffs[0]]) > np.max(dat.LAnklePower[landings[1]:takeoffs[1]])):
                trimmedLandings = [i for a, i in enumerate(landings) if  a%2 == 0]
                trimmedTakesoffs = [i for a, i in enumerate(takeoffs) if  a%2 == 0]
            else:
                trimmedLandings = [i for a, i in enumerate(landings) if  a%2 != 0]
                trimmedTakesoffs = [i for a, i in enumerate(takeoffs) if  a%2 != 0]

        else:
            trimmedLandings = landings
            trimemdTakesoffs = takeoffs
        
        # if dat.COP_X[landings[0]] < dat.COP_X[landings[1]]:
        #     trimmedLandings = landings[0:-1:2]
        #     trimmedTakesoffs = takeoffs[0:-1:2]
        # else:
        #     trimmedLandings = landings[1:-1:2]
        #     trimmedTakesoffs = takeoffs[1:-1:2]
        
        # Index through all the landings        
        for countVar, landing in enumerate(trimmedLandings):
            try:
                # separate into positive and negatiave work
                pkAnklePw.append( np.max(AnklePower[landing:trimmedTakesoffs[countVar]]) )
                ankleNetWork.append( np.sum(AnklePower[landing:trimmedTakesoffs[countVar]] ))
                ankleNegWork.append( sum(i for i in AnklePower[landing:trimmedTakesoffs[countVar]] if i < 0) )
                anklePosWork.append( sum(i for i in AnklePower[landing:trimmedTakesoffs[countVar]] if i > 0) )
                
                kneeXROM.append( np.max(KneeXAng[landing:trimmedTakesoffs[countVar]]) - np.min(KneeXAng[landing:trimmedTakesoffs[countVar]]) )
                kneeZROM.append( np.max(KneeZAng[landing:trimmedTakesoffs[countVar]]) - np.min(KneeZAng[landing:trimmedTakesoffs[countVar]]) )
                
                pkKneePw.append( np.max( KneePower[landing:trimmedTakesoffs[countVar]]) )
                kneeNetWork.append( np.sum( KneePower[landing:trimmedTakesoffs[countVar]]) )
                kneeNegWork.append(sum(i for i in KneePower[landing:trimmedTakesoffs[countVar]] if i < 0) )
                kneePosWork.append( sum(i for i in KneePower[landing:trimmedTakesoffs[countVar]] if i > 0) )
                
                pkHipPw.append( np.max(HipPower[landing:trimmedTakesoffs[countVar]]) )
                hipNetWork.append( np.sum(HipPower[landing:trimmedTakesoffs[countVar]]) )
                hipNegWork.append( sum(i for i in HipPower[landing:trimmedTakesoffs[countVar]] if i < 0) )
                hipPosWork.append( sum(i for i in HipPower[landing:trimmedTakesoffs[countVar]] if i > 0) )
                
                pkAnkleInv.append( np.max(AnkleInversion[landing:trimmedTakesoffs[countVar]]) )
                pkAnkleFlex.append( np.max(AnkleFlexion[landing:trimmedTakesoffs[countVar]]) )
                ankleFlexROM.append( np.max(AnkleFlexion[landing:trimmedTakesoffs[countVar]]) - np.min(AnkleFlexion[landing:trimmedTakesoffs[countVar]]) )
                ankleInvROM.append( np.max(AnkleInversion[landing:trimmedTakesoffs[countVar]]) - np.min(AnkleInversion[landing:trimmedTakesoffs[countVar]]) )
                ankleAbdROM.append( np.max(AnkleAbd[landing:trimmedTakesoffs[countVar]]) - np.min(AnkleAbd[landing:trimmedTakesoffs[countVar]]) )           
                
                pkKneeFlex.append( np.max(KneeFlex[landing:trimmedTakesoffs[countVar]]) ) 
                pkKneeRot.append( np.max(KneeRot[landing:trimmedTakesoffs[countVar]]) )
                
                pkHipFlex.append( np.max(HipFlex[landing:trimmedTakesoffs[countVar]]) )
                pkHipRot.append( np.max(HipInt[landing:trimmedTakesoffs[countVar]]) )
                pkHipAbd.append( np.max(HipAbd[landing:trimmedTakesoffs[countVar]]) )
                hipRotROM.append( np.max(HipInt[landing:trimmedTakesoffs[countVar]]) - np.min(HipInt[landing:trimmedTakesoffs[countVar]]))
                hipAbdROM.append( np.max(HipAbd[landing:trimmedTakesoffs[countVar]]) - np.min(HipAbd[landing:trimmedTakesoffs[countVar]]))
                kneeRotROM.append( np.max(KneeRot[landing:trimmedTakesoffs[countVar]]) - np.min(KneeRot[landing:trimmedTakesoffs[countVar]]))
                hipFlexROM.append( np.max(HipFlex[landing:trimmedTakesoffs[countVar]]) - np.min(HipFlex[landing:trimmedTakesoffs[countVar]]) )
                kneeFlexROM.append( np.max(KneeFlex[landing:trimmedTakesoffs[countVar]]) - np.min(KneeFlex[landing:trimmedTakesoffs[countVar]]) )
                
                pkAnkleMomX.append( np.max(AnkleMomX[landing:trimmedTakesoffs[countVar]]) )
                pkAnkleMomY.append( np.max(AnkleMomY[landing:trimmedTakesoffs[countVar]]) )
                pkAnkleMomZ.append( np.max(AnkleMomZ[landing:trimmedTakesoffs[countVar]]) )
                minAnkleMomX.append( np.min(AnkleMomX[landing:trimmedTakesoffs[countVar]]) )
                minAnkleMomY.append( np.min(AnkleMomY[landing:trimmedTakesoffs[countVar]]) )
                minAnkleMomZ.append( np.min(AnkleMomZ[landing:trimmedTakesoffs[countVar]]) )
                
                pkKneeMomX.append( np.max(KneeMomX[landing:trimmedTakesoffs[countVar]]) )
                pkKneeMomY.append( np.max(KneeMomY[landing:trimmedTakesoffs[countVar]]) )
                pkKneeMomZ.append( np.max(KneeMomZ[landing:trimmedTakesoffs[countVar]]) )
                minKneeMomX.append( np.min(KneeMomX[landing:trimmedTakesoffs[countVar]]) )
                minKneeMomY.append( np.min(KneeMomY[landing:trimmedTakesoffs[countVar]]) )
                minKneeMomZ.append( np.min(KneeMomZ[landing:trimmedTakesoffs[countVar]]) )        
                  
                pkHipMomX.append( np.max(HipMomX[landing:trimmedTakesoffs[countVar]]) )
                pkHipMomY.append( np.max(HipMomY[landing:trimmedTakesoffs[countVar]]) )
                pkHipMomZ.append( np.max(HipMomZ[landing:trimmedTakesoffs[countVar]]) )
                minHipMomX.append( np.min(HipMomX[landing:trimmedTakesoffs[countVar]]) )
                minHipMomY.append( np.min(HipMomY[landing:trimmedTakesoffs[countVar]]) )
                minHipMomZ.append( np.min(HipMomZ[landing:trimmedTakesoffs[countVar]]) )      
                
                sName.append(subName)
                tmpConfig.append(config)
                
            except:
                print(landing)
                
        
    except:
        print(fName)

# Put all variables of interest into a single variable
outcomes = pd.DataFrame({'Subject':list(sName), 'Config': list(tmpConfig),'PkAnklePw':list(pkAnklePw), 'AnkleNetWork':list(ankleNetWork),
                         'AnkleNegWork':list(ankleNegWork), 'AnklePosWork':list(anklePosWork),
                         'PkKneePw':list(pkKneePw), 'KneeNetWork':list(kneeNetWork),'KneeNegWork':list(kneeNegWork),'KneePosWork':list(kneePosWork),
                         'PkHipPw':list(pkHipPw), 'HipNetWork':list(hipNetWork), 'hipNegWork':list(hipNegWork),'hipPosWork':list(hipPosWork),
                         'pkAnkleMomX':list(pkAnkleMomX),'pkAnkleMomY':list(pkAnkleMomY),'HipRotROM':list(hipRotROM),'HipAbdROM':list(hipAbdROM),
                         'pkAnkleMomZ':list(pkAnkleMomZ), 'pkKneeMomX':list(pkKneeMomX), 'pkKneeMomY':list(pkKneeMomY),'pkKneeMomZ':list(pkKneeMomZ),
                         'pkAnkleFlex':list(pkAnkleFlex), 'pkAnkleInv':list(pkAnkleInv), 'pkKneeFlex':list(pkKneeFlex),
                         'pkKneeRot':list(pkKneeRot), 'pkHipAbd':list(pkHipAbd), 'pkHipFlex':list(pkHipFlex),'pkHipInt':list(pkHipRot),
                         'AnkleAbdROM':list(ankleAbdROM), 'AnkleInvROM':list(ankleInvROM), 'AnkleFlexROM':list(ankleFlexROM),
                         'minHipMomZ':list(minHipMomZ), 'minHipMomY':list(minHipMomY), 'minKneeMomZ':list(minKneeMomZ), 'MinKneeMomY':list(minKneeMomY),
                         'minAnkleMomZ':list(minAnkleMomZ), 'hipFlexROM':list(hipFlexROM), 'kneeRotROM':list(kneeRotROM), 'kneeFlexROM':list(kneeFlexROM),
                         'pkHipMomX':list(pkHipMomX),  'pkHipMomY':list(pkHipMomY), 'pkHipMomZ':list(pkHipMomZ), 'kneeXROM':list(kneeXROM),
                         'kneeZROM':list(kneeZROM)})

# Create csv with all outcome variables
outcomes.to_csv('C:\\Users\\Daniel.Feeney\\Dropbox (Boa)\\Endurance Health Validation\\DU_Running_Summer_2021\\Data\\KinematicsKineticsROM2.csv')#, mode ='a',header = False)
#outcomes.to_csv('C:\\Users\\daniel.feeney\\Boa Technology Inc\\PFL - General\\AgilityPerformanceData\\BOA_InternalStrap_July2021\\KineticsKinematics\\TM\\KinKinematics.csv')#, mode ='a',header = False)


def makeFig(inputDF, forceCol, Xcol, Ycol, Zcol, title):
    """
    Plot aligned time series data for first-look at the data

    Parameters
    ----------
    inputDF : dataframe
        Dataframe containing all data
    forceCol : str
        Column header for GRF in string format (e.g. 'ForcesZ')
    Xcol : str
        Column header for the intended variable in string format (e.g. 'ForcesZ')
    Ycol : str
        Column header for the intended variable in string format (e.g. 'ForcesZ')
    Zcol : str
        Column header for the intended variable in string format (e.g. 'ForcesZ')
    title : str
        title of the plot

    Returns
    -------
    None.

    """
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

# Full time series data for investigations of raw data if needed 
#makeFig(dat, 'ForcesZ', 'LAnklePower', 'LKneePower', 'LHipPower', 'Powers')

#makeFig(dat, 'ForcesZ', 'LAnkleMomentx', 'LAnkleMomenty', 'LAnkleMomentz', 'Ankle Moments')
#makeFig(dat, 'ForcesZ', 'LAnkleFlex', 'LAnkleInv', 'LAnkleAbd', 'Ankle Angles')

#makeFig(dat, 'ForcesZ', 'LKneeMomentX', 'LKneeMomentY', 'LKneeMomentZ', 'Knee Moments')
# makeFig(dat, 'ForcesZ', 'LKneeXAngle', 'LKneeYAngle', 'LKneeZAngle', 'Knee Angles')

#makeFig(dat, 'ForcesZ', 'LHipMomentx', 'LHipMomenty', 'LHipMomentz', 'Hip Moments')
#makeFig(dat, 'ForcesZ', 'LHipXAngle', 'LHipYAngle', 'LHipZAngle', 'Hip Angles')
