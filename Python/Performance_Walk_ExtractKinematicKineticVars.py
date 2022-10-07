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
from numpy import cos,sin
import matplotlib.pyplot as plt
import os
import scipy
import scipy.signal as sig
from scipy.integrate import cumtrapz
import addcopyfighandler


# Define constants and options
fThresh = 50 #below this value will be set to 0.
lookFwd = 50
timeToLoad = 150 #length to look forward for an impact peak
pd.options.mode.chained_assignment = None  # default='warn' set to warn for a lot of warnings




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

def intp_strides(var,landings,GS):
    """
    Function to interpolate the variable of interest across a stride
    (from foot contact to subsiquent foot contact) in order to plot the 
    variable of interest over top each other

    Parameters
    ----------
    var : list or numpy array
        Variable of interest. Can be taken from a dataframe or from a numpy array
    landings : list
        Foot contact indicies

    Returns
    -------
    intp_var : numpy array
        Interpolated variable to 101 points with the number of columns dictated
        by the number of strides.

    """
    # Preallocate
    intp_var = np.zeros((101,len(GS)-1))
    # Index through the strides
    for ii in range(len(GS)-1):
        dum = var[landings[GS[ii]]:landings[GS[ii]+1]]
        f = scipy.interpolate.interp1d(np.arange(0,len(dum)),dum)
        intp_var[:,ii] = f(np.linspace(0,len(dum)-1,101))
        
    return intp_var

def findPosNegWork(var,freq):
    """
    Function to compute the positive and negative work for a particular variable
    that is already segmented across a step or stride. 

    Parameters
    ----------
    var : list or numpy array
        Variable of interest that has been segmented for a step/stride
    freq : int
        frequency

    Returns
    -------
    pos_work : float
        positive work
    neg_work : float
        negative work

    """
    
    # Define positive and negative portions for the variable of interest
    pos_var = np.array(var); pos_var = pos_var[pos_var > 0]
    neg_var = np.array(var); neg_var = neg_var[neg_var < 0]
    
    pos_work = scipy.integrate.trapezoid(pos_var,dx = 1/freq)
    neg_work = scipy.integrate.trapezoid(neg_var,dx = 1/freq)
    
    return [pos_work,neg_work]

def COMPower_Work_walking(LGRF,RGRF,slope,HS,GoodStrides,freq):
    """
    This function computes the center-of-mass power and work for the "leading"
    limb (ie the limb that is used to segment the GRFs)

    Parameters
    ----------
    LGRF : numpy array (Nx3)
        Left foot GRF
    RGRF : numpy array (Nx3)
        Right foot GRF
    slope : float or int
        slope of the treadmill
    HS : numpy array (Nx1)
        Heel strike (foot contact) array
    GoodStrides : numpy array (Nx1)
        Array of good strides
    freq : foat or int
        Data collection frequency

    Returns
    -------
    CW_pos : list
        Positive COM work [J]
    CW_neg : list
        Negative COM work [J]

    """
    # Compute the COM power using the individual limbs method                
    
    # Debugging tool: Showing the time-continuous COM power
    show_COMpower = 0

    # First compute the approximate body weight: will need to rotate the
    # ground reaction forces into the inertial coordinate system
    slope = slope*np.pi/180
    BM = np.nanmean(LGRF[:,1]*np.sin(slope)+LGRF[:,2]*np.cos(slope)+RGRF[:,1]*np.sin(slope)+RGRF[:,2]*np.cos(slope))/9.81
    # Compute the COM acceleration
    acc = (LGRF+RGRF)/BM - [0,9.81*np.sin(slope),9.81*np.cos(slope)]
    
    # Pre-allocate variable space
    CW_pos = []; CW_neg = []
    
    COM_power_store = np.zeros((101,len(GoodStrides)-1))
    # Index through the good strides for computing COM Power + Work
    for cc,jj in enumerate(GoodStrides[:-1]):
        acc_stride = acc[HS[jj]:HS[jj+1],:]
        time_stride = np.array(range(len(acc_stride)))/freq
        com_vel = cumtrapz(acc_stride,time_stride,initial=0,axis=0)
        com_vel = com_vel - np.mean(com_vel,axis=0) + [0,speed,0]
        # COM Power
        com_power_lead = np.sum(com_vel*LGRF[HS[jj]:HS[jj+1],:],axis=1)
        # Compute the positive/negative work
        # Note: this may need to be updated for level ground for collision/push-off work
        [pos_tmp,neg_temp] = findPosNegWork(com_power_lead,freq)

        CW_pos.append(pos_tmp)
        CW_neg.append(neg_temp)
        # Store the time-continous COM curve
        f = scipy.interpolate.interp1d(np.arange(0,len(com_power_lead)),com_power_lead)
        COM_power_store[:,cc] = f(np.linspace(0,len(com_power_lead)-1,101))
    
    # Debugging tool: examine the time-continous curves    
    if show_COMpower == 1:
        plt.plot(COM_power_store)
        plt.close() # create a breakpoint here for visualizing plots
    
    return(CW_pos,CW_neg)

#______________________________________________________________________________
# Read in balance file
fPath = 'C:\\Users\\eric.honert\\Boa Technology Inc\\PFL Team - General\\Testing Segments\\Hike\\ZonalFit_Midcut_Aug2022\\Treadmill\\'
entries = [fName for fName in os.listdir(fPath) if fName.endswith('PerformanceTestData_V2.txt')]

save_on = 0
#______________________________________________________________________________
#Preallocation
oSub = []

kinSub = []
kinCond = []
kinName = []
kinConfig = []
pAnkEvVel = []

forSub = []
forCond = []
forName = []
forConfig = []
loadingRate = []
peakBrakeF = []
brakeImpulse = []
brakeImpulse_rot = []
VALRs = []
VLRtwo = []
pkForce = []
COMWork_pos = []
COMWork_neg = []


timeP = []
NL = []
PkMed = []
PkLat = []
CTs = []
meanForce = []
propImpulse = []
propImpulse_rot = []


slope6 = ['GregMullen','TJ']

## loop through the selected files
for ii in range(0,len(entries)):
    # try:
        fName = entries[ii] #Load one file at a time
        print(fName)
        
        #Parse file name into subject and configuration 
        subName = fName.split(sep = "_")[0]
        oSub.append(subName)
        # config = fName.split(sep = "_")[1]
        
        if '2D' in fName:
            config = 'DD'
        elif 'TD' in fName:
            config = 'UZ'
        elif 'BD' in fName:
            config = 'LZ'
        
        print(config)
        
        # Treadmill speed
        if fName.count('ownhill'):
            speed = -1.2
            tmpCond = 'Downhill'
        else:
            speed = 1.2
            tmpCond = 'Uphill'
            
        # Set the angle of the treadmill
        if subName in slope6:
            ang = 6*np.pi/180
        else:
            ang = 10*np.pi/180  
        
        # Extract data frequency
        freq = pd.read_csv(fPath+fName,sep='\t',usecols=[0], nrows=1, skiprows=[0,1], header = 0)
        freq = freq.values.tolist()
        freq = freq[0][0]
            
        dat = pd.read_csv(fPath+fName,sep='\t', skiprows = 8, header = 0)
        
        # Always check force directions
        if np.mean(dat.Left_GRF_Z) < 0:
            dat.Left_GRF_X = -1 * dat.Left_GRF_X
            dat.Left_GRF_Y = 1 * dat.Left_GRF_Y
            dat.Left_GRF_Z = -1 * dat.Left_GRF_Z
            dat.Right_GRF_X = -1 * dat.Right_GRF_X
            dat.Right_GRF_Y = 1 * dat.Right_GRF_Y
            dat.Right_GRF_Z = -1 * dat.Right_GRF_Z
        else:
            dat.Left_GRF_Y = -1 * dat.Left_GRF_Y
            dat.Right_GRF_Y = -1 * dat.Right_GRF_Y
            
        LGRF = np.array(list(zip(dat.Left_GRF_X,dat.Left_GRF_Y,dat.Left_GRF_Z)))
        RGRF = np.array(list(zip(dat.Right_GRF_X,dat.Right_GRF_Y,dat.Right_GRF_Z)))
        # Trim the trials to a smaller section and threshold force
        forceDat = dat
        
        MForce = dat.Left_GRF_X
        brakeFilt = dat.Left_GRF_Y      
        forceZ = trimForce(dat.Left_GRF_Z, fThresh)        
                
        #find the landings and takeoffs of the FP as vectors
        landings = findLandings(forceZ)
        takeoffs = findTakeoffs(forceZ)

        trimmedTakeoffs = trimTakeoffs(landings, takeoffs)
        # determine if first step is left or right then delete every other
        # landing and takeoff. MORE NEGATIVE IS LEFT
        if landings[-1] > trimmedTakeoffs[-1]:
            trimmedLandings = landings[0:-1]
        else:
            trimmedLandings = landings
        
        ## check to make sure brake force is applied in the correct direction ##
        if np.mean(brakeFilt[landings[1]:landings[1]+100]) > 0:
            brakeFilt = -1 * brakeFilt
            
        # Need to eliminate bad strides
        # Assume: that most of the strides are good
        step_time = []
        for jj in range(len(trimmedLandings)):
            step_time.append(trimmedTakeoffs[jj]-trimmedLandings[jj])
        
        # Crossover detection: Create a variable for "Good Steps"
        # Note: the last statement is to ensure that steady state walking is 
        # attained for tests where 3 hops are performed.
        GS = []
        for jj in range(len(trimmedLandings)):
            if step_time[jj] < np.median(step_time) + 20 and np.min(dat.Right_GRF_Z[trimmedLandings[jj]:trimmedTakeoffs[jj]]) < fThresh and trimmedLandings[jj] > 2000:
                GS.append(jj)
            
        GS = np.array(GS)
        # Compute COM work
        [tmpCW_pos,tmpCW_neg] = COMPower_Work_walking(LGRF,RGRF,ang,trimmedLandings,GS,freq)
        COMWork_pos.extend(tmpCW_pos)
        COMWork_neg.extend(tmpCW_neg)      
        
        # dat = dat.fillna(0)
        if fName.count('ownhill'):
            test = intp_strides(dat.RightAnklePower,landings,GS)        
            plt.plot(test,'k')
            plt.close()
        
        # Compute force-based metrics
        # Index through the good steps
        for jj in GS[:-1]:
                try:
                    # Define where next zero is
                    # Loading Rate commented out - moving away from metric
                    # VALRs.append(calcVLR(forceZ, trimmedLandings[jj]+1, 150, timeToLoad, freq))
                    # VLRtwo.append( (np.max( np.diff(forceZ[trimmedLandings[jj]+5:trimmedLandings[jj]+150]) )/(1/freq) ) )
                    
                    if fName.count('ownhill'):
                        brakeImpulse.append( sum(i for i in -LGRF[trimmedLandings[jj]:trimmedTakeoffs[jj],1] if i < 0)/freq ) #sum all negative brake force vals
                        propImpulse.append( sum(i for i in -LGRF[trimmedLandings[jj]:trimmedTakeoffs[jj],1] if i > 0)/freq ) #sum all positive values
                        peakBrakeF.append(np.min(-LGRF[trimmedLandings[jj]:trimmedTakeoffs[jj],1]))
                        
                        if subName != 'JeffGay':
                            # Kinematics
                            idx20 = round(0.2*(trimmedTakeoffs[jj] - trimmedLandings[jj])) + trimmedLandings[jj]
                            pAnkEvVel.append(abs(np.min(dat.RAnkleAngVel_Frontal[trimmedLandings[jj]-20:idx20])))
                            kinName.append(subName)
                            kinConfig.append(config)
                            kinCond.append(tmpCond)
                        
                    else:
                        brakeImpulse.append( sum(i for i in LGRF[trimmedLandings[jj]:trimmedTakeoffs[jj],1] if i < 0)/freq ) #sum all negative brake force vals
                        propImpulse.append( sum(i for i in LGRF[trimmedLandings[jj]:trimmedTakeoffs[jj],1] if i > 0)/freq ) #sum all positive values
                        peakBrakeF.append(np.min(LGRF[trimmedLandings[jj]:trimmedTakeoffs[jj],1]))
                        
                    
                    try:
                        CTs.append(trimmedTakeoffs[jj] - trimmedLandings[jj])
                    except:
                        CTs.append(0)
                    
                    try:
                        PkMed.append(np.max(MForce[trimmedLandings[jj]:trimmedTakeoffs[jj]]))
                    except:
                        PkMed.append(0)
                    try:
                        PkLat.append(np.min(MForce[trimmedLandings[jj]:trimmedTakeoffs[jj]]))
                    except:
                        PkLat.append(0)
                    forSub.append(subName)
                    forConfig.append(config)
                    forCond.append(tmpCond)
                except:
                    print(trimmedLandings[jj])
        

# foroutcomes = pd.DataFrame({'Subject':list(forSub), 'Config': list(forConfig),'Cond': list(forCond),'pBF': list(peakBrakeF),
                          # 'brakeImpulse': list(brakeImpulse), 'VALR': list(VALRs), 'VILR':list(VLRtwo),'pMF':list(PkMed),
                          # 'pLF':list(PkLat), 'CT':list(CTs),'PropImp':list(propImpulse)})
# workoutcomes = pd.DataFrame({'Subject':list(forSub), 'Config': list(forConfig),'Cond': list(forCond), 'COMWork_pos': list(COMWork_pos), 'COMWork_neg': list(COMWork_neg),
#                              'brakeImpulse': list(brakeImpulse),'PropImp':list(propImpulse)})

# if save_on == 1:
#     workoutcomes.to_csv("C:\\Users\\eric.honert\\Boa Technology Inc\\PFL Team - General\\Testing Segments\\Hike\\ZonalFit_Midcut_Aug2022\\COMWork.csv",header=True)

# kinoutcomes = pd.DataFrame({'Subject':list(kinName), 'Config': list(kinConfig), 'Cond': list(kinCond),
#                           'pAnkEvVel': list(pAnkEvVel)})

# kinoutcomes.to_csv("C:\\Users\\eric.honert\\Boa Technology Inc\\PFL Team - General\\Testing Segments\\Hike\\ZonalFit_Midcut_Aug2022\\Kinematics.csv",header=True)



