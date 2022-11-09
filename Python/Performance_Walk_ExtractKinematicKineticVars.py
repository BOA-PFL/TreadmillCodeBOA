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
from tkinter import messagebox


# Define constants and options
fThresh = 50 #below this value will be set to 0.
lookFwd = 50
timeToLoad = 150 #length to look forward for an impact peak
save_on = 1
debug = 1
pd.options.mode.chained_assignment = None  # default='warn' set to warn for a lot of warnings


### set plot font size ###
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
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

def COMPower_Work_walking(LGRF,RGRF,slope,walk_speed,HS,GoodStrides,freq):
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
    walk_speed : float or int
        velocity of the treadmill
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
    BM = np.nanmean(LGRF[:,1]*np.sin(slope)+LGRF[:,2]*np.cos(slope)+RGRF[:,1]*np.sin(slope)+RGRF[:,2]*np.cos(slope))/9.81
    # Compute the COM acceleration
    acc = (LGRF+RGRF)/BM - [0,9.81*np.sin(slope),9.81*np.cos(slope)]
    
    # Pre-allocate variable space
    CW_pos = []; CW_neg = []
    
    COM_power_store = np.zeros((101,len(GoodStrides)-1))
    # Index through the good strides for computing COM Power + Work
    for cc, jj in enumerate(GoodStrides[:-1]):
        acc_stride = acc[HS[jj]:HS[jj+1],:]
        time_stride = np.array(range(len(acc_stride)))/freq
        com_vel = cumtrapz(acc_stride,time_stride,initial=0,axis=0)
        com_vel = com_vel - np.mean(com_vel,axis=0) + [0,walk_speed,0]
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

def dist_seg_power_treadmill(Seg_COM_Pos,Seg_COM_Vel,Seg_Ang_Vel,CenterOfPressure,GRF,FreeMoment,speed,landings,takeoffs,yn_run):
    """
    The purpose of this function is to compute the distal segment power -
    commonly applied to the rearfoot to obtain the distal rearfoot power. The
    power in this formation was provided in Takahashi et al. 2012; but
    originally in Siegel et al. 1996. For full derivations, see Zelik and 
    Honert 2018 appendix. This code assumes that the locomotion direction is +y

    Parameters
    ----------
    Seg_COM_Pos : numpy array (N X 3)
        Segment COM Position (ex: Foot COM Position)
    Seg_COM_Vel : numpy array (N X 3)
        Segment COM Velocity (ex: Foot COM Velocity)
    Seg_Ang_Vel : numpy array (N X 3)
        Segment Angular Velocity (ex: Foot Angular Velocity)
    CenterOfPressure : numpy array (N X 3)
        Location of the center of pressure
    GRF : numpy array (N X 3)
        Ground Reaction Force. Ensure that the input GRF is the REACTION
    FreeMoment : numpy array (N X 3)
        Free moment on force platform
    speed : float
        Treadmill belt speed - can be used as a debugging variable or to set
        the speed of the foot in 3D space. 
    landings : list
        Initial foot contact: used only during walking
    takeoffs : list
        Or toe-offs: used only during walking
    yn_run : int
        1 for running, 0 for walking
    
    Returns
    -------
    power : numpy array
        distal rearfoot power

    """
    
    # If walking: compute the treadmill belt speed
    if yn_run == 0:
        # Debugging variable to examine foot speed
        debug = 0
        # When using a treadmill is used for locomotion and the distal segment
        # power is computed, the treadmill belt speed needs to be taken into
        # account. Based on prior experience, DURING WALKING, foot flat can provide
        # a decent approximation of the treadmill belt speed. 
        foot_flat = [0.2,0.4]
        
        # Allocate variables
        step_speed = np.zeros((len(landings)-1,1))
        
        # Index through the landings
        for ii in range(len(landings)-1):
            stepframe = takeoffs[ii]-landings[ii]
            # Frames to analyze based on the foot flat percentages
            FFframes = range(landings[ii]+round(foot_flat[0]*stepframe),landings[ii]+round(foot_flat[1]*stepframe),1)
            step_speed[ii] = np.mean(Seg_COM_Vel[FFframes,1])
    
        # Find the average treadmill belt speed of the trial (also exclude any 
        # zeros in the estimate)
        avg_speed = -np.mean(step_speed[step_speed != 0])
        
        if debug == 1:
            plt.figure(1010)
            plt.plot(step_speed)
            
    # If running: use the inputted Bertec treadmill speed
    else: 
        # It is difficult to compute the belt speed from running - thus rely on
        # the set treadmill belt speed
        avg_speed = np.array(speed)
    
    # Treadmill belt speed: will need to be updated based on the slope
    belt_vel = np.array(list(zip([0]*len(Seg_COM_Vel),[avg_speed]*len(Seg_COM_Vel),[0]*len(Seg_COM_Vel))))
    
    # Adjust the segment velocity based on belt speed
    adj_Seg_COM_Vel = Seg_COM_Vel+belt_vel
    # Compute the rotational and translational components of the power    
    power_rot = np.sum(np.cross(CenterOfPressure-Seg_COM_Pos,GRF,axis=1)*Seg_Ang_Vel,axis=1)+np.sum(FreeMoment*Seg_Ang_Vel,axis=1)
    power_tran = np.sum(GRF*adj_Seg_COM_Vel,axis=1)
    
    # test = crop_strides_fft(inputDF.FootCOMPos_X,landings)
    
    power = power_rot+power_tran
    return power

def makeVizPlotForce(inputDF, inputLeftLandings, goodSteps):
    plt.subplot(1,3,1)
    plt.plot(intp_strides(inputDF.Left_GRF_X, inputLeftLandings, goodSteps),'k')
    plt.ylabel('GRF X (N)')
    plt.subplot(1,3,2)
    plt.plot(intp_strides(inputDF.Left_GRF_Y, inputLeftLandings, goodSteps),'k')
    plt.ylabel('GRF Y (N)')
    plt.subplot(1,3,3)
    plt.plot(intp_strides(inputDF.Left_GRF_Z, inputLeftLandings, goodSteps),'k')
    plt.ylabel('GRF Z (N)')
    plt.tight_layout()
    
#______________________________________________________________________________
# Read in balance file
fPath = 'C:\\Users\\daniel.feeney\\Boa Technology Inc\\PFL Team - General\\Testing Segments\\Hike\\FocusAnkleDualDial_Midcut_Sept2022\\Treadmill\\'
entries = [fName for fName in os.listdir(fPath) if fName.endswith('PerformanceTestData_V2.txt')]

# Look at the text files from the foot work 
fPath_footwork = 'C:\\Users\\daniel.feeney\\Boa Technology Inc\\PFL Team - General\\Testing Segments\\Hike\\FocusAnkleDualDial_Midcut_Sept2022\\Treadmill\\'
entries_footwork = [fName for fName in os.listdir(fPath_footwork) if fName.endswith('DistalRearfootPower.txt')]

if len(entries) > len(entries_footwork):
    print("Warning: Missing Foot Work Files")


#______________________________________________________________________________
#Preallocation

# Study Details:
oSub = []
oConfig = []
oSlope = []
oSesh = []
oSpeed = []

# Force Plate Variables
CTs = []
VALRs = []
PkMed = []
PkLat = []
peakBrakeF = []
brakeImpulse = []
propImpulse = []
COMWork_pos = []
COMWork_neg = []

# Kinematic/Kinetic Variables
pAnkEvVel = []
AnkWork_pos = []
AnkWork_neg = []
DisWork = []

badFileList = []
## loop through the selected files
for ii in range(0,len(entries)):
    # try:
        fName = entries[ii] #Load one file at a time
        print(fName)
        
        #Parse file name into subject and configuration: temp names 
        tmpSub = fName.split(sep = "_")[0]
        tmpConfig = fName.split(sep = "_")[1]
        
        # Dictate the slope and the direction of walking
        if fName.count('DH'):
            speed = -1.2
            tmpCond = 'Downhill'
            tmpSlope = -10
            # Set the angle of the treadmill
            ang = 10*np.pi/180
            
            
        else:
            speed = 1.2
            tmpCond = 'Uphill'
            tmpSlope = 10
            # Set the angle of the treadmill
            ang = 10*np.pi/180

        # Extract data frequency
        freq = pd.read_csv(fPath+fName,sep='\t',usecols=[0], nrows=1, skiprows=[0,1], header = 0)
        freq = freq.values.tolist()
        freq = freq[0][0]
            
        # Open the treadmill data
        dat = pd.read_csv(fPath+fName,sep='\t', skiprows = 8, header = 0)
        # Open the footwork data
        fwdat = pd.read_csv(fPath_footwork+entries_footwork[ii],sep='\t', skiprows = 8, header = 0)
        
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
        if tmpCond == 'Downhill':
            brakeFilt = -np.array(dat.Left_GRF_Y) 
        else:
            brakeFilt = np.array(dat.Left_GRF_Y) 
            
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
        [tmpCW_pos,tmpCW_neg] = COMPower_Work_walking(LGRF,RGRF,ang,speed,trimmedLandings,GS,freq)
        COMWork_pos.extend(tmpCW_pos)
        COMWork_neg.extend(tmpCW_neg)
        
        # Distal foot power computations
        Foot_Ang_Vel = np.array(list(zip(fwdat.RFootAngVel_X,fwdat.RFootAngVel_Y,fwdat.RFootAngVel_Z)))*(np.pi/180)    
        Foot_COM_Pos = np.array(list(zip(fwdat.RFootCOMPos_X,fwdat.RFootCOMPos_Y,fwdat.RFootCOMPos_Z)))
        Foot_COM_Vel = np.array(list(zip(fwdat.RFootCOMVel_X,fwdat.RFootCOMVel_Y,fwdat.RFootCOMVel_Z)))
        COP = np.array(list(zip(fwdat.COP_X,fwdat.COP_Y,fwdat.COP_Z)))
        COP = np.nan_to_num(COP,nan=0)
        FMOM = np.array(list(zip(fwdat.FMOM_X,fwdat.FMOM_Y,fwdat.FMOM_Z)))
        FMOM = np.nan_to_num(FMOM,nan=0)

        DFootPower = dist_seg_power_treadmill(Foot_COM_Pos,Foot_COM_Vel,Foot_Ang_Vel,COP,LGRF,FMOM,speed,landings,takeoffs,0)
        
        
        # Index through the good steps
        # store good strides for foot work: debugging purposes
        GSfw = []
        GSaw = []
        for jj in GS[:-1]:
                try:
                    # Compute force-based metrics
                    # Loading Rate: used for fit purposes, not injury
                    VALRs.append(calcVLR(forceZ, trimmedLandings[jj]+1, 150, timeToLoad, freq))
                    # Contact Time
                    CTs.append((trimmedTakeoffs[jj] - trimmedLandings[jj])/freq)
                    # Peak Medial/Lateral forces
                    PkMed.append(np.max(MForce[trimmedLandings[jj]:trimmedTakeoffs[jj]]))
                    PkLat.append(np.min(MForce[trimmedLandings[jj]:trimmedTakeoffs[jj]]))
                    # Braking and Propulsive Force Metrics
                    brakeImpulse.append( sum(i for i in brakeFilt[trimmedLandings[jj]:trimmedTakeoffs[jj]] if i < 0)/freq ) #sum all negative brake force vals
                    propImpulse.append( sum(i for i in brakeFilt[trimmedLandings[jj]:trimmedTakeoffs[jj]] if i > 0)/freq ) #sum all positive values
                    peakBrakeF.append(np.min(brakeFilt[trimmedLandings[jj]:trimmedTakeoffs[jj]]))
                    
                    # MoCap+ metrics
                    # Only for downhill metrics
                    if tmpCond == 'Downhill':
                        # Only compute ankle and foot metrics from stable kinematic data
                        # Peak Ankle Eversion Velocity: Fit Metric
                        idx20 = round(0.2*(trimmedTakeoffs[jj] - trimmedLandings[jj])) + trimmedLandings[jj]
                        if sum(np.isnan(dat.RFootPosDetect[trimmedLandings[jj]-20:idx20])) == 0:
                            pAnkEvVel.append(abs(np.min(dat.RAnkleAngVel_Frontal[trimmedLandings[jj]-20:idx20])))
                        else:
                            pAnkEvVel.append(np.nan)

                        if sum(np.isnan(dat.RFootPosDetect[trimmedLandings[jj]:trimmedTakeoffs[jj]])) == 0 and np.max(abs(dat.RightAnklePower[trimmedLandings[jj]:trimmedLandings[jj+1]])) < 2000:
                            # Ankle Work: Endurance/Health Metric
                            [pos_tmp,neg_tmp] = findPosNegWork(dat.RightAnklePower[trimmedLandings[jj]:trimmedTakeoffs[jj]],freq)
                            GSaw.append(jj)
                            AnkWork_pos.append(pos_tmp)
                            AnkWork_neg.append(neg_tmp)
                            # Examine distal rearfoot work
                            dis_idx = round((trimmedLandings[jj+1]-trimmedLandings[jj])*.20)+trimmedLandings[jj]
                            if np.max(abs(DFootPower[trimmedLandings[jj]:trimmedLandings[jj+1]])) < 500:
                                [pos_tmp,neg_tmp] = findPosNegWork(np.array(DFootPower[trimmedLandings[jj]:dis_idx]),freq)
                                DisWork.append(neg_tmp)
                                GSfw.append(jj)
                            else:
                                DisWork.append(np.nan)
                            
                        else:
                            
                            AnkWork_pos.append(np.nan)
                            AnkWork_neg.append(np.nan)
                            DisWork.append(np.nan)
                    # For uphill, forward walking conditions
                    else:
                        pAnkEvVel.append(np.nan)
                        AnkWork_pos.append(np.nan)
                        AnkWork_neg.append(np.nan)
                        DisWork.append(np.nan)
                        
                    # Append study details
                    oSub.append(tmpSub)
                    oConfig.append(tmpConfig)
                    oSlope.append(tmpSlope)
                    oSesh.append(1)
                    oSpeed.append(abs(speed))
                except:
                    print(trimmedLandings[jj])
        
        # Debugging plots:  
        
        if tmpCond == 'Downhill' and debug == 1:
            makeVizPlotForce(dat, trimmedLandings, GS)
            if tmpCond == 'Downhill': 
                plt.figure(ii)
                if len(GSfw) > 0:
                    plt.subplot(1,2,1)
                    plt.plot(intp_strides(DFootPower,trimmedLandings,GSfw),'k')
                    plt.ylabel('Foot Power [W]')
                    plt.tight_layout()
                
                if len(GSaw) > 0:
                    plt.subplot(1,2,2)
                    plt.plot(intp_strides(dat.RightAnklePower,trimmedLandings,GSaw),'k')
                    plt.ylabel('Ankle Power [W]')
                    plt.tight_layout()
                
                    #plt.close()
            answer = messagebox.askyesno("Question","Is data clean?")
        
        if tmpCond == 'Uphill' and debug == 1:
            makeVizPlotForce(dat, trimmedLandings, GS)
            answer = messagebox.askyesno("Question","Is data clean?")
            
        if answer == False:
            plt.close('all')
            print('Adding file to bad file list')
            badFileList.append(fName)
            
        if answer == True:
            plt.close('all')
            print('Estimating point estimates')
            
        ### Append into DF and Save if save turned on ###
            outcomes = pd.DataFrame({'Subject':list(oSub), 'Config': list(oConfig),'Slope': list(oSlope),'Speed': list(oSpeed), 'Sesh': list(oSesh),
                                     'CT':list(CTs), 'VALR': list(VALRs), 'pMF':list(PkMed), 'pLF':list(PkLat),
                                     'pBF': list(peakBrakeF), 'brakeImpulse': list(brakeImpulse), 'PropImp':list(propImpulse),
                                     'pAnkEvVel': list(pAnkEvVel), 'COMWork_pos': list(COMWork_pos), 'COMWork_neg': list(COMWork_neg),
                                     'AnkWork_pos': list(AnkWork_pos), 'AnkWork_neg': list(AnkWork_neg), 'DisWork': list(DisWork)})
                                      
            
            if save_on == 1:
                  
                outfileName = fPath + 'TreadmillOutcomes_test.csv'
                
                if os.path.exists(outfileName) == False:
                    
                    outcomes.to_csv(outfileName, mode='a', header=True, index = False)
                
                else:
                    outcomes.to_csv(outfileName, mode='a', header=False, index = False) 
                
                

