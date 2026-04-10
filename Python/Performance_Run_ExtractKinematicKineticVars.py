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
import scipy
import scipy.signal as sig
from scipy.integrate import cumulative_trapezoid as cumtrapz
from tkinter import messagebox
from TreadmillFunctions import (findLandings, findTakeoffs, calcVLR,
                                 trimForce, trimLandings, trimTakeoffs,
                                 findPosNegWork)

#______________________________________________________________________________
# Define constants and options
fThresh = 50 #below this value will be set to 0.
lookFwd = 50
timeToLoad = 150 #length to look forward for an impact peak
pd.options.mode.chained_assignment = None  # default='warn' set to warn for a lot of warnings

# Read in general treadmill data file
fPath = 'Z:/Testing Segments/Outdoor/TrailRunning/2025/2025_Pressure_EX3-EX330_Kailas/Treadmill/'
entries = [fName for fName in os.listdir(fPath) if fName.endswith('PerformanceTestData_V2.txt')]

save_on = 0
debug = 1 #this must be set to 1 on the first pass at the data

#______________________________________________________________________________
# list of functions

def intp_steps(var,fHS,fTO,GS):
    """
    Function to interpolate the variable of interest across a step
    (from foot contact to toe-off) in order to plot the
    variable of interest over top each other

    Parameters
    ----------
    var : list or numpy array
        Variable of interest. Can be taken from a dataframe or from a numpy array
    fHS : list
        Foot contact (heel strike) indices
    fTO : list
        Toe-off indices
    GS : list or numpy array
        Indices of good steps to interpolate

    Returns
    -------
    intp_var : numpy array
        Interpolated variable to 101 points with the number of columns dictated
        by the number of steps.

    """
    # Preallocate
    intp_var = np.zeros((101,len(GS)))
    # Index through the strides
    for jj, ii in enumerate(GS):
        dum = var[fHS[ii]:fTO[ii]+1]
        f = scipy.interpolate.interp1d(np.arange(0,len(dum)),dum)
        intp_var[:,jj] = f(np.linspace(0,len(dum)-1,101))
        
    return intp_var

def COMPower_Work_run(GRF,speed,slope,HS,TO,GoodStrides,freq):
    """
    This function computes the center-of-mass power and work for the "leading"
    limb (ie the limb that is used to segment the GRFs)

    Parameters
    ----------
    GRF : numpy array (Nx3)
        Ground reaction force
    speed : float or int
        running speed
    slope : float or int
        slope of the treadmill
    HS : numpy array (Nx1)
        Heel strike (foot contact) array
    TO : numpy array (Nx1)
        Toe-off array
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
    BM = np.nanmean(GRF[:,1]*np.sin(slope*np.pi/180)+GRF[:,2]*np.cos(slope))/9.81
    # Compute the COM acceleration
    acc = GRF/BM - [0,9.81*np.sin(slope),9.81*np.cos(slope)]
    
    # Pre-allocate variable space
    CW_pos = []; CW_neg = []
    
    COM_power_store = np.zeros((101,len(GoodStrides)-1))
    # Index through the good strides for computing COM Power + Work
    for cc,jj in enumerate(GoodStrides[:-1]):
        acc_step = acc[HS[jj]:TO[jj],:]
        time_step = np.array(range(len(acc_step)))/freq
        com_vel = cumtrapz(acc_step,time_step,initial=0,axis=0)
        com_vel = com_vel - np.mean(com_vel,axis=0) + [0,speed,0]
        # COM Power
        com_power = np.sum(com_vel*GRF[HS[jj]:TO[jj],:],axis=1)
        # Compute the positive/negative work
        # Note: this may need to be updated for level ground for collision/push-off work
        [pos_tmp,neg_temp] = findPosNegWork(com_power,freq)

        CW_pos.append(pos_tmp)
        CW_neg.append(neg_temp)
        # Store the time-continous COM curve
        f = scipy.interpolate.interp1d(np.arange(0,len(com_power)),com_power)
        COM_power_store[:,cc] = f(np.linspace(0,len(com_power)-1,101))
    
    # Debugging tool: examine the time-continous curves    
    if show_COMpower == 1:
        plt.plot(COM_power_store)
        plt.close() # create a breakpoint here for visualizing plots
    
    return(CW_pos,CW_neg,COM_power_store)


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

badFileList = []

## loop through the selected files
   
for ii, entry in enumerate(entries):
    # try:
        fName = entry #Load one file at a time
        print(fName)
        
        #Parse file name into subject and configuration: temp names 
        tmpSub = fName.split(sep = "_")[0]
        tmpConfig = fName.split(sep = "_")[1]
        tmpOrd = fName.split(sep = "_")[3].split(sep = ' ')[0]
        
        # Dictate the slope and the direction of walking
        if fName.count('DH'):
            speed = -1.2
            tmpCond = 'Downhill'
            tmpSlope = -10
            # Set the angle of the treadmill
            ang = 10
            
            
        elif fName.count('UH'):
            speed = 1.2
            tmpCond = 'Uphill'
            tmpSlope = 10
            # Set the angle of the treadmill
            ang = 10
            
        elif fName.count('run'): 
            speed = 3.0
            tmpCond = 'Level'
            tmpSlope = 0
            ang = 0
        else:
            print('Defaulting to 0 Slope and Running Speed of 3.0 m/s')
            speed = 3.0
            tmpCond = 'Level'
            tmpSlope = 0
            ang = 0
            

        # Extract data frequency
        freq = pd.read_csv(fPath+fName,sep='\t',usecols=[0], nrows=1, skiprows=[0,1], header = 0)
        freq = freq.values.tolist()
        freq = freq[0][0]
        
        # Open the treadmill data
        dat = pd.read_csv(fPath+fName,sep='\t', skiprows = 8, header = 0)

        # Always check force directions
        if np.mean(dat.Left_GRF_Z) < 0:
            dat.Left_GRF_X = -1 * dat.Left_GRF_X
            dat.Left_GRF_Y = 1 * dat.Left_GRF_Y
            dat.Left_GRF_Z = -1 * dat.Left_GRF_Z
            
        else:
            dat.Left_GRF_Y = -1 * dat.Left_GRF_Y
        
        # Extract subject mass from forces
        mass = np.nanmean(dat.Left_GRF_Z)/9.81
                
        LGRF = np.array(list(zip(dat.Left_GRF_X,dat.Left_GRF_Y,dat.Left_GRF_Z)))
        
        # Trim the trials to a smaller section and threshold force
        forceDat = dat
        
        MForce = dat.Left_GRF_X
        if tmpCond == 'Downhill':
            brakeFilt = -np.array(dat.Left_GRF_Y) 
        else:
            brakeFilt = np.array(dat.Left_GRF_Y) 
            
        forceZ = trimForce(dat.Left_GRF_Z, fThresh)        
                
        #find the landings and takeoffs of the FP as vectors
        landings = findLandings(forceZ, fThresh)
        takeoffs = findTakeoffs(forceZ, fThresh)

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
        for jj, val in enumerate(trimmedLandings):
            step_time.append(trimmedTakeoffs[jj] - val)
        
        # Crossover detection: Create a variable for "Good Steps"
        # Note: the last statement is to ensure that steady state walking is 
        # attained for tests where 3 hops are performed.
        GS = []
        for jj, val in enumerate(trimmedLandings):
            if step_time[jj] < np.median(step_time) + 20 and np.min(dat.Right_GRF_Z[val:trimmedTakeoffs[jj]]) < fThresh and val > 2000:
                GS.append(jj)
        GS = np.array(GS)
        
        # Compute COM work
        [tmpCW_pos,tmpCW_neg,debug_COMPW] = COMPower_Work_run(LGRF,speed,ang,trimmedLandings,trimmedTakeoffs,GS,freq)
        COMWork_pos.extend(tmpCW_pos)
        COMWork_neg.extend(tmpCW_neg)
        
        #______________________________________________________________
        # Debugging: Creation of dialog box for looking where foot contact are accurate
        answer = True # Defaulting to true: In case "debug" is not used
        # Debugging plots:
        if debug == 1:
            plt.subplot(1,3,1)
            plt.plot(intp_steps(LGRF[:,1],trimmedLandings,trimmedTakeoffs,GS))
            plt.ylabel('A/P GRF [N]')
            plt.xlabel('% Step')
            
            plt.subplot(1,3,2)
            plt.plot(intp_steps(LGRF[:,2],trimmedLandings,trimmedTakeoffs,GS))
            plt.ylabel('Vertical GRF [N]')
            plt.xlabel('% Step')
        
            plt.subplot(1,3,3)
            plt.plot(debug_COMPW)
            plt.ylabel('COM Power [N]')
            plt.xlabel('% Step')
            
            answer = messagebox.askyesno("Question","Is data clean?")
            
            if answer == False:
                print('Adding file to bad file list')
                badFileList.append(fName)
                plt.close()
            
        if answer == True:
            saveFolder = fPath + 'TreadmillPlots'
            if os.path.exists(saveFolder) == False:
              os.mkdir(saveFolder) 
            plt.savefig(saveFolder + '/' + fName.split('.csv')[0] +'.png')
            plt.close()
            
            # Index through the good steps
            for jj in GS[:-1]:
                    try:
                        # Compute force-based metrics
                        # Loading Rate: used for fit purposes, not injury. Great Easter Egg, Eric
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
                        pAnkEvVel.append(np.nan)
                        AnkWork_pos.append(np.nan)
                        AnkWork_neg.append(np.nan)

                        # Append study details
                        oSub.append(tmpSub)
                        oConfig.append(tmpConfig)
                        oSlope.append(tmpSlope)
                        oSesh.append(tmpOrd)
                        oSpeed.append(abs(speed))
                    except:
                        print(trimmedLandings[jj])

outcomes = pd.DataFrame({'Subject':list(oSub), 'Config': list(oConfig),'Slope': list(oSlope),'Speed': list(oSpeed), 'Order': list(oSesh),
                         'CT':list(CTs), 'VALR': list(VALRs), 'pMF':list(PkMed), 'pLF':list(PkLat),
                         'pBF': list(peakBrakeF), 'brakeImpulse': list(brakeImpulse), 'PropImp':list(propImpulse),
                         'pAnkEvVel': list(pAnkEvVel), 'COMWork_pos': list(COMWork_pos), 'COMWork_neg': list(COMWork_neg),
                         'AnkWork_pos': list(AnkWork_pos), 'AnkWork_neg': list(AnkWork_neg)})
                          

if save_on == 1:
    outfileName = fPath + '0_TreadmillOutcomes_test.csv'
    outcomes.to_csv(outfileName, index = False)
    
    if os.path.exists(outfileName) == False:
        
        outcomes.to_csv(outfileName, mode='a', header=True, index = False)
        badFileList.to_csv(fPath + 'BadFiles.csv', mode = 'a', header = True, index = False)
    
    else:
        outcomes.to_csv(outfileName, mode='a', header=False, index = False) 
        badFileList.to_csv(fPath + 'BadFiles.csv', mode = 'a', header = False, index = False)




