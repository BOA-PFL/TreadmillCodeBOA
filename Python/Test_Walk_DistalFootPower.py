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
import scipy


# Define constants and options
run = 1 # Set this to 1 where participant is running on one belt so only the left are detected. 0 for dual belt
manualTrim = 0  #set this to 1 if you want to manually trim trials with ginput, 0 if you want it auto trimmed (start and end of trial)
fThresh = 50 #below this value will be set to 0.
writeData = 0 #will write to spreadsheet if 1 entered
plottingEnabled = 0 #plots the bottom if 1. No plots if 0
lookFwd = 50
timeToLoad = 75 #length to look forward for an impact peak
pd.options.mode.chained_assignment = None  # default='warn' set to warn for a lot of warnings

# Look at the text files from the foot work 
fPath_footwork = 'C:\\Users\\eric.honert\\Boa Technology Inc\\PFL Team - General\\Segments\\WorkWear_Performance\\Elten_Jan2022\\Treadmill\FootPower\\'
entries_footwork = os.listdir(fPath_footwork)
   
#______________________________________________________________________________    
# list of functions 
# finding landings on the force plate once the filtered force exceeds the force threshold
def findLandings(force):
    lic = []
    for step in range(len(force)-1):
        if force[step] == 0 and force[step + 1] >= fThresh:
            lic.append(step)
            
    if lic[0] == 0:
        lic = lic[1:]
    return lic

#Find takeoff from FP when force goes from above thresh to 0
def findTakeoffs(force):
    lto = []
    for step in range(len(force)-1):
        if force[step] >= fThresh and force[step + 1] == 0:
            lto.append(step + 1)
    return lto


def calcVLR(force, startVal, lengthFwd, endLoading):
    # function to calculate VLR from 80 and 20% of the max value observed in the first n
    # indices (n defined by lengthFwd). 
    # endLoading should be set to where an impact peak should have occured if there is one
    # and can be biased longer so the for loop doesn't error out
    # lengthFwd is how far forward to look to calculate VLR
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
        maxF = np.max(force[startVal:startVal+endLoading])
        eightyPctMax = 0.8 * maxF
        twentyPctMax = 0.2 * maxF
        # find indices of 80 and 20 and calc loading rate as diff in force / diff in time (N/s)
        eightyIndex = next(x for x, val in enumerate(force[startVal:startVal+endLoading]) 
                          if val > eightyPctMax) 
        twentyIndex = next(x for x, val in enumerate(force[startVal:startVal+endLoading]) 
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
        if force[step] <= 0 and force[step + 1] >= 0 and step > 45:
            break
    return step

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

def filterForce(inputForce, sampFrq, cutoffFrq):
        # low-pass filter the input force signals
        #t = np.arange(len(inputForce)) / sampFrq
        w = cutoffFrq / (sampFrq / 2) # Normalize the frequency
        b, a = sig.butter(4, w, 'low')
        filtForce = sig.filtfilt(b, a, inputForce)
        return(filtForce)
    
def trimForce(inputDF, threshForce):
    forceTot = inputDF.GRF_Z
    forceTot[forceTot<threshForce] = 0
    forceTot = np.array(forceTot)
    return(forceTot)

def trimLandings(landingVec, takeoffVec):
    if landingVec[0] > takeoffVec[0]:
        landingVec.pop(0)
        return(landingVec)
    else:
        return(landingVec)
    
def trimTakeoffs(landingVec, takeoffVec):
    if landingVec[0] > takeoffVec[0]:
        takeoffVec.pop(0)
        return(takeoffVec)
    else:
        return(takeoffVec)
    
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

def dist_seg_power_treadmill(inputDF,speed,slope,landings,takeoffs,direction):
    """
    The purpose of this function is to compute the distal segment power -
    commonly applied to the rearfoot to obtain the distal rearfoot power. The
    power in this formation was provided in Takahashi et al. 2012; but
    originally in Siegel et al. 1996. For full derivations, see Zelik and 
    Honert 2018 appendix. 

    Parameters
    ----------
    inputDF : DataFrame
        This data frame needs to contain the following variables:
            Seg_COM_Vel = Segment COM Velocity (ex: Foot COM Velocity)
            Seg_Ang_Vel = Segment Angular Velocity (ex: Foot Angular Velocity)
            Seg_COM_Pos = Segment COM Position (ex: Foot COM Position)
            CenterOfPressure = Location of the center of pressure
            FreeMoment = Free moment on force platform
            GRF = Ground Reaction Force            
    speed : scalar
        Treadmill belt speed - can be used as a debugging variable or to set
        the speed of the foot in 3D space. 
    slope : scalar
        Slope of the treadmill
    landings : list
        Initial foot contact
    takeoffs : list
        Or toe-offs
    direction: list
        1 for x being the forward direction, 2 for y being the forward direction
    
    Returns
    -------
    power : TYPE
        DESCRIPTION.

    """
    
    # Make Variables NumPy Arrays for matrix opperations
    Seg_Ang_Vel = np.array(list(zip(inputDF.FootAngVel_X,inputDF.FootAngVel_Y,inputDF.FootAngVel_Z)))*(np.pi/180)    
    Seg_COM_Pos = np.array(list(zip(inputDF.FootCOMPos_X,inputDF.FootCOMPos_Y,inputDF.FootCOMPos_Z)))
    Seg_COM_Vel = np.array(list(zip(inputDF.FootCOMVel_X,inputDF.FootCOMVel_Y,inputDF.FootCOMVel_Z)))
    CenterOfPressure = np.array(list(zip(inputDF.COP_X,inputDF.COP_Y,inputDF.COP_Z)))
    GRF = np.array(list(zip(inputDF.GRF_X,inputDF.GRF_Y,inputDF.GRF_Z)))
    # Note: The free moment (like the GRF) is needs to be negated in order to
    # be the reaction free moment
    FreeMoment = -1*np.array(list(zip(inputDF.FMOM_X,inputDF.FMOM_Y,inputDF.FMOM_Z)))
    
    # Debugging variable to examine foot speed
    debug = 0
    # When using a treadmill is used for locomotion and the distal segment
    # power is computed, the treadmill belt speed needs to be taken into
    # account. Based on prior experience, DURING WALKING, foot flat can provide
    # a decent approximation of the treadmill belt speed. 
    foot_flat = [0.2,0.4]
    
    # Allocate variables
    step_speed = np.zeros((len(landings)-1,1))
    
    
    for ii in range(len(landings)-1):
        stepframe = takeoffs[ii]-landings[ii]
        # Frames to analyze based on the foot flat percentages
        FFframes = range(landings[ii]+round(foot_flat[0]*stepframe),landings[ii]+round(foot_flat[1]*stepframe),1)
        if direction == 1:
            step_speed[ii] = np.mean(inputDF.FootCOMVel_X[FFframes])
        elif direction == 2:
            step_speed[ii] = np.mean(inputDF.FootCOMVel_Y[FFframes])

    # Find the average treadmill belt speed of the trial (also exclude any 
    # zeros in the estimate)
    avg_speed = -np.mean(step_speed[step_speed != 0])
    
    if debug == 1:
        plt.figure(1010)
        plt.plot(step_speed)
    
    # Treadmill belt speed: will need to be updated based on the slope
    if direction == 1:
        belt_vel = np.array(list(zip([avg_speed]*len(inputDF.FootCOMVel_X),[0]*len(inputDF.FootCOMVel_Y),[0]*len(inputDF.FootCOMVel_Z))))
    elif direction == 2:
        belt_vel = np.array(list(zip([0]*len(inputDF.FootCOMVel_X),[avg_speed]*len(inputDF.FootCOMVel_Y),[0]*len(inputDF.FootCOMVel_Z))))
    
    # Adjust the segment velocity based on belt speed
    adj_Seg_COM_Vel = Seg_COM_Vel+belt_vel
    # Compute the rotational and translational components of the power    
    power_rot = np.sum(np.cross(CenterOfPressure-Seg_COM_Pos,GRF,axis=1)*Seg_Ang_Vel,axis=1)+np.sum(FreeMoment*Seg_Ang_Vel,axis=1)
    power_tran = np.sum(GRF*adj_Seg_COM_Vel,axis=1)
    
    # test = crop_strides_fft(inputDF.FootCOMPos_X,landings)
    
    
    power = power_rot+power_tran
    
    
    return power

def intp_strides(var,landings):

    intp_var = np.zeros((101,len(landings)-1))

    for ii in range(len(landings)-1):
        dum = var[landings[ii]:landings[ii+1]]
        f = scipy.interpolate.interp1d(np.arange(0,len(dum)),dum)
        intp_var[:,ii] = f(np.linspace(0,len(dum)-1,101))
        
    return intp_var

def crop_strides_fft(var,landings):
    from scipy.fft import fft, fftfreq
    
    intp_var = np.zeros((500,len(landings)-1))
    fft_out = np.zeros((500,len(landings)-1))
    
    for ii in range(len(landings)-1):
        intp_var[0:landings[ii+1]-landings[ii],ii] = var[landings[ii]:landings[ii+1]]
        fft_out[:,ii] = fft(intp_var[:,ii])
        xf = fftfreq(500,1/200)
        
    return [fft_out,xf]
    
    
    
    
    

#______________________________________________________________________________

#Preallocate variables for storage
Subject = []
Config = []
DisWork = []


#______________________________________________________________________________
avgAnkPow = np.zeros((101,len(entries_footwork)))
avgFootPow = np.zeros((101,len(entries_footwork)))

# when COPx is more negative, that is left foot strike
## loop through the selected files
for ii in range(len(entries_footwork)):
    try:
        
        #______________________________________________________________________
        # Load the files associated with the foot power/work
        fName = entries_footwork[ii] #Load one file at a time
        
        #Parse file name into subject and configuration 
        subName = fName.split(sep = "_")[0]
        ConfigTmp = fName.split(sep = "_")[2]

        dat = pd.read_csv(fPath_footwork+fName,sep='\t', skiprows = 8, header = 0)  
        dat = dat.fillna(0)
        #______________________________________________________________________
        # Flip the direction of the GRFs
        dat.GRF_X = -1 * dat.GRF_X
        dat.GRF_Y = -1 * dat.GRF_Y
        dat.GRF_Z = -1 * dat.GRF_Z
        
        # Trim the trials to a smaller section and threshold force
        if manualTrim == 1:
            print('Select start and end of analysis trial 1')
            forceDat = delimitTrial(dat)
        else: 
            forceDat = dat
        
        # Zero the forces below the threshold
        forceZ = trimForce(forceDat, fThresh)
        
        # Find the landings and takeoffs of the FP as vectors
        landings = findLandings(forceZ)
        takeoffs = findTakeoffs(forceZ)
        # Crop the landings and takeoffs
        takeoffs = trimTakeoffs(landings, takeoffs)
        if len(takeoffs) < len(landings):
            landings = landings[0:-1]
        
        # dum = np.gradient(dat.Foot4_Z)
        # [sigFFT,XX] = crop_strides_fft(dat.FootAngVel_Y,landings)
        
        # Compute the distal rearfoot power:
        if subName == 'RosaLoveszy':
            DFootPower = dist_seg_power_treadmill(dat,1.2,0,landings,takeoffs,1)
        else:
            DFootPower = dist_seg_power_treadmill(dat,1.2,0,landings,takeoffs,2)
        
        avgFootPow[:,ii] = np.mean(intp_strides(DFootPower,landings),axis = 1)
        
        for counterVar, landing in enumerate(landings):
            try:
                dis_idx = round((takeoffs[counterVar]-landing)*.20)+landing
                dis_work = scipy.integrate.trapezoid(DFootPower[landing:dis_idx],dx = 1/200)
                
                DisWork.append(dis_work)
                Subject.append(subName)
                Config.append(ConfigTmp)
            except:
                print(fName, landing)
        

        1
        
    except:
        print(fName)
        
        
        
        
        
        
        
        
        
        
        

outcomes = pd.DataFrame({'Subject':list(Subject), 'Config': list(Config),'DisWork': list(DisWork)})

outcomes.to_csv("C:\\Users\\eric.honert\\Boa Technology Inc\\PFL Team - General\\Segments\\WorkWear_Performance\\Elten_Jan2022\\Treadmill\\FootWork.csv",mode='a',header=True)


