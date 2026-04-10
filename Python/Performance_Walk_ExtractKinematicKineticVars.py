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
from scipy.integrate import cumulative_trapezoid
from tkinter import messagebox
from TreadmillFunctions import (findLandings, findTakeoffs, calcVLR,
                                 calcPeakBrake, findNextZero, delimitTrial,
                                 filterForce, trimForce, trimLandings,
                                 trimTakeoffs, forceMatrix, findPosNegWork)


fPath = 'Z:\\Testing Segments\\WorkWear\\2025\\2025_WWDrape_Materials_Danner\\Treadmill\\'
entries = [fName for fName in os.listdir(fPath) if fName.endswith('PerformanceTestData_V2.txt')]

# Define constants and options
fThresh = 50 #below this value will be set to 0.
lookFwd = 50
timeToLoad = 150 #length to look forward for an impact peak
save_on = 0
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
    GS : list or numpy array
        Indices of good strides to interpolate

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

def COMPower_Work_walking(LeftGRF,RightGRF,slope,walk_speed,LeftHS,RightHS,LeftGS,RightGS,freq):
    """
    This function computes the center-of-mass power and work for both
    limbs separately

    Parameters
    ----------
    LeftGRF : numpy array (Nx3)
        Left foot GRF
    RightGRF : numpy array (Nx3)
        Right foot GRF
    slope : float or int
        slope of the treadmill
    walk_speed : float or int
        velocity of the treadmill
    LeftHS : numpy array (Nx1)
        Left heel strike (foot contact) array
    RightHS : numpy array (Nx1)
        Right heel strike (foot contact) array
    LeftGS : numpy array (Nx1)
        Array of good strides for the left side
    RightGS : numpy array (Nx1)
        Array of good strides for the right side
    freq : foat or int
        Data collection frequency

    Returns
    -------
    LCW_pos : list
        Left positive COM work [J]
    LCW_neg : list
        Left negative COM work [J]
    RCW_pos : list
        Right positive COM work [J]
    RCW_neg : list
        Right negative COM work [J]
    LCOM_power_store : numpy array
        Left time-continuous COM power curves
    RCOM_power_store : numpy array
        Right time-continuous COM power curves
    mass : float
        Estimated body mass [kg]

    """
    # Compute the COM power using the individual limbs method

    # First compute the approximate body weight: will need to rotate the
    # ground reaction forces into the inertial coordinate system
    mass = np.nanmean(LeftGRF[:,1]*np.sin(slope)+LeftGRF[:,2]*np.cos(slope)+RightGRF[:,1]*np.sin(slope)+RightGRF[:,2]*np.cos(slope))/9.81
    # Compute the COM acceleration
    acc = (LeftGRF+RightGRF)/mass - [0,9.81*np.sin(slope),9.81*np.cos(slope)]

    # Pre-allocate variable space
    LCW_pos = []; LCW_neg = []
    RCW_pos = []; RCW_neg = []

    LCOM_power_store = np.zeros((101,len(LeftGS)-1))
    RCOM_power_store = np.zeros((101,len(RightGS)-1))
    # Index through the good strides for computing Left COM Power + Work
    for cc, jj in enumerate(LeftGS[:-1]):
        acc_stride = acc[LeftHS[jj]:LeftHS[jj+1],:]
        time_stride = np.array(range(len(acc_stride)))/freq
        com_vel = cumulative_trapezoid(acc_stride,time_stride,initial=0,axis=0)
        com_vel = com_vel - np.mean(com_vel,axis=0) + [0,walk_speed,0]
        # COM Power
        com_power_lead = np.sum(com_vel*LeftGRF[LeftHS[jj]:LeftHS[jj+1],:],axis=1)
        # Compute the positive/negative work
        [pos_tmp,neg_temp] = findPosNegWork(com_power_lead,freq)

        LCW_pos.append(pos_tmp)
        LCW_neg.append(neg_temp)
        # Store the time-continous COM curve
        f = scipy.interpolate.interp1d(np.arange(0,len(com_power_lead)),com_power_lead)
        LCOM_power_store[:,cc] = f(np.linspace(0,len(com_power_lead)-1,101))

    # Index through the good strides for computing Right COM Power + Work
    for cc, jj in enumerate(RightGS[:-1]):
        acc_stride = acc[RightHS[jj]:RightHS[jj+1],:]
        time_stride = np.array(range(len(acc_stride)))/freq
        com_vel = cumulative_trapezoid(acc_stride,time_stride,initial=0,axis=0)
        com_vel = com_vel - np.mean(com_vel,axis=0) + [0,walk_speed,0]
        # COM Power
        com_power_lead = np.sum(com_vel*RightGRF[RightHS[jj]:RightHS[jj+1],:],axis=1)
        # Compute the positive/negative work
        [pos_tmp,neg_temp] = findPosNegWork(com_power_lead,freq)

        RCW_pos.append(pos_tmp)
        RCW_neg.append(neg_temp)
        # Store the time-continous COM curve
        f = scipy.interpolate.interp1d(np.arange(0,len(com_power_lead)),com_power_lead)
        RCOM_power_store[:,cc] = f(np.linspace(0,len(com_power_lead)-1,101))

    return(LCW_pos,LCW_neg,RCW_pos,RCW_neg,LCOM_power_store,RCOM_power_store,mass)

def makeVizPlotForce(LGRF, RGRF, LHS, RHS, LGS, RGS, downhill):
    """
    Create a 2x3 panel plot of GRF components for both legs.
    Top row: left force plate, bottom row: right force plate.
    Note: for downhill (backwards walking), the left force plate corresponds
    to the right leg and vice versa, so labels are swapped accordingly.

    Parameters
    ----------
    LGRF : numpy array (Nx3)
        Left force plate ground reaction force [X, Y, Z]
    RGRF : numpy array (Nx3)
        Right force plate ground reaction force [X, Y, Z]
    LHS : numpy array
        Left side heel strike (foot contact) indices
    RHS : numpy array
        Right side heel strike (foot contact) indices
    LGS : numpy array
        Left side good stride indices
    RGS : numpy array
        Right side good stride indices
    downhill : bool
        True if the condition is downhill (backwards walking), which swaps
        the leg labels on the plot

    Returns
    -------
    None
        Displays a matplotlib figure

    """
    if downhill:
        topLabel = 'Right Leg'
        botLabel = 'Left Leg'
    else:
        topLabel = 'Left Leg'
        botLabel = 'Right Leg'

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    # Top row: Left force plate GRFs
    axes[0,0].plot(intp_strides(LGRF[:,0], LHS, LGS),'k')
    axes[0,0].set_ylabel(topLabel + ' GRF X (N)')
    axes[0,1].plot(intp_strides(LGRF[:,1], LHS, LGS),'k')
    axes[0,1].set_ylabel(topLabel + ' GRF Y (N)')
    axes[0,2].plot(intp_strides(LGRF[:,2], LHS, LGS),'k')
    axes[0,2].set_ylabel(topLabel + ' GRF Z (N)')
    # Bottom row: Right force plate GRFs
    axes[1,0].plot(intp_strides(RGRF[:,0], RHS, RGS),'k')
    axes[1,0].set_ylabel(botLabel + ' GRF X (N)')
    axes[1,1].plot(intp_strides(RGRF[:,1], RHS, RGS),'k')
    axes[1,1].set_ylabel(botLabel + ' GRF Y (N)')
    axes[1,2].plot(intp_strides(RGRF[:,2], RHS, RGS),'k')
    axes[1,2].set_ylabel(botLabel + ' GRF Z (N)')
    plt.tight_layout()
    
#______________________________________________________________________________
#Preallocation

# Study Details:
oSub = []
oConfig = []
oSlope = []
oSesh = []
oSpeed = []
Side = []

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
for ii in range(len(entries)):
    # try:
        fName = entries[ii] #Load one file at a time
        print(fName)
        
        #Parse file name into subject and configuration: temp names 
        tmpSub = fName.split(sep = "_")[0]
        tmpConfig = fName.split(sep = "_")[1]
        tmpOrder = fName.split(sep = "_")[3][0]
        
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

        # Threshold both left and right GRFs
        idx = LGRF[:,2] < fThresh
        LGRF[idx,:] = 0
        idx = RGRF[:,2] < fThresh
        RGRF[idx,:] = 0

        # Find the landings and takeoffs for each side
        LHS = np.array(findLandings(LGRF[:,2], fThresh))
        LTO = findTakeoffs(LGRF[:,2], fThresh)
        LTO = np.array(trimTakeoffs(LHS, LTO))
        if LHS[-1] > LTO[-1]:
            LTO = LTO[0:-1]

        RHS = np.array(findLandings(RGRF[:,2], fThresh))
        RTO = findTakeoffs(RGRF[:,2], fThresh)
        RTO = np.array(trimTakeoffs(RHS, RTO))
        if RHS[-1] > RTO[-1]:
            RTO = RTO[0:-1]

        # Medial force for each side
        LMForce = LGRF[:,0]
        RMForce = RGRF[:,0]
        # Brake force for each side (sign depends on walking direction)
        if tmpCond == 'Downhill':
            LBrakeFilt = -LGRF[:,1]
            RBrakeFilt = -RGRF[:,1]
        else:
            LBrakeFilt = LGRF[:,1]
            RBrakeFilt = RGRF[:,1]

        timemin = 0.3
        timemax = 2

        # GS: good strides - make sure that there are no cross over steps & that
        # the athlete has reached steady state
        LGS = []
        for jj in range(6,len(LHS)-1):
            if LHS[jj+1] - LHS[jj] > timemin*freq and LHS[jj+1] - LHS[jj] < timemax*freq and min(RGRF[LHS[jj]:LHS[jj+1],2]) == 0:
                LGS.append(jj)
        LGS = np.array(LGS)

        RGS = []
        for jj in range(6,len(RHS)-1):
            if RHS[jj+1] - RHS[jj] > timemin*freq and RHS[jj+1] - RHS[jj] < timemax*freq and min(LGRF[RHS[jj]:RHS[jj+1],2]) == 0:
                RGS.append(jj)
        RGS = np.array(RGS)

        # Compute COM work
        [tmpLCW_pos,tmpLCW_neg,tmpRCW_pos,tmpRCW_neg,LCOMdebug,RCOMdebug,BM] = COMPower_Work_walking(LGRF,RGRF,ang,speed,LHS,RHS,LGS,RGS,freq)
        COMWork_pos.extend(tmpLCW_pos); COMWork_pos.extend(tmpRCW_pos)
        COMWork_neg.extend(tmpLCW_neg); COMWork_neg.extend(tmpRCW_neg)

        # Index through left good steps
        for jj in LGS[:-1]:
                try:
                    # Compute force-based metrics
                    # Loading Rate: used for fit purposes, not injury
                    VALRs.append(calcVLR(LGRF[:,2], LHS[jj]+1, 150, timeToLoad, freq))
                    # Contact Time
                    CTs.append((LTO[jj] - LHS[jj])/freq)
                    # Peak Medial/Lateral forces
                    PkMed.append(np.max(LMForce[LHS[jj]:LTO[jj]]))
                    PkLat.append(np.min(LMForce[LHS[jj]:LTO[jj]]))
                    # Braking and Propulsive Force Metrics
                    brakeImpulse.append( sum(i for i in LBrakeFilt[LHS[jj]:LTO[jj]] if i < 0)/freq )
                    propImpulse.append( sum(i for i in LBrakeFilt[LHS[jj]:LTO[jj]] if i > 0)/freq )
                    peakBrakeF.append(np.min(LBrakeFilt[LHS[jj]:LTO[jj]]))

                    # MoCap+ metrics
                    # Only for downhill metrics
                    if tmpCond == 'Downhill':
                        # Only compute ankle and foot metrics from stable kinematic data
                        # Peak Ankle Eversion Velocity: Fit Metric
                        idx20 = round(0.2*(LTO[jj] - LHS[jj])) + LHS[jj]
                        if sum(np.isnan(dat.RFootPosDetect[LHS[jj]-20:idx20])) == 0:
                            pAnkEvVel.append(abs(np.min(dat.RAnkleAngVel_Frontal[LHS[jj]-20:idx20])))
                        else:
                            pAnkEvVel.append(np.nan)

                        if sum(np.isnan(dat.RFootPosDetect[LHS[jj]:LTO[jj]])) == 0 and np.max(abs(dat.RightAnklePower[LHS[jj]:LHS[jj+1]])) < 2000:
                            # Ankle Work: Endurance/Health Metric
                            [pos_tmp,neg_tmp] = findPosNegWork(dat.RightAnklePower[LHS[jj]:LTO[jj]],freq)
                            AnkWork_pos.append(pos_tmp)
                            AnkWork_neg.append(neg_tmp)

                        else:
                            AnkWork_pos.append(np.nan)
                            AnkWork_neg.append(np.nan)
                    # For uphill, forward walking conditions
                    else:
                        pAnkEvVel.append(np.nan)
                        AnkWork_pos.append(np.nan)
                        AnkWork_neg.append(np.nan)

                    # Append study details
                    oSub.append(tmpSub)
                    oConfig.append(tmpConfig)
                    oSlope.append(tmpSlope)
                    oSesh.append(tmpOrder)
                    oSpeed.append(abs(speed))
                    Side.append('Left')
                except:
                    print(LHS[jj])

        # Index through right good steps
        for jj in RGS[:-1]:
                try:
                    # Compute force-based metrics
                    VALRs.append(calcVLR(RGRF[:,2], RHS[jj]+1, 150, timeToLoad, freq))
                    CTs.append((RTO[jj] - RHS[jj])/freq)
                    PkMed.append(np.max(RMForce[RHS[jj]:RTO[jj]]))
                    PkLat.append(np.min(RMForce[RHS[jj]:RTO[jj]]))
                    brakeImpulse.append( sum(i for i in RBrakeFilt[RHS[jj]:RTO[jj]] if i < 0)/freq )
                    propImpulse.append( sum(i for i in RBrakeFilt[RHS[jj]:RTO[jj]] if i > 0)/freq )
                    peakBrakeF.append(np.min(RBrakeFilt[RHS[jj]:RTO[jj]]))

                    # MoCap+ metrics not available for right side currently
                    pAnkEvVel.append(np.nan)
                    AnkWork_pos.append(np.nan)
                    AnkWork_neg.append(np.nan)

                    # Append study details
                    oSub.append(tmpSub)
                    oConfig.append(tmpConfig)
                    oSlope.append(tmpSlope)
                    oSesh.append(tmpOrder)
                    oSpeed.append(abs(speed))
                    Side.append('Right')
                except:
                    print(RHS[jj])
        
        # Debugging plots:  
        
        if tmpCond == 'Downhill' and debug == 1:
            makeVizPlotForce(LGRF, RGRF, LHS, RHS, LGS, RGS, downhill=True)
            answer = messagebox.askyesno("Question","Is data clean?")

        if tmpCond == 'Uphill' and debug == 1:
            makeVizPlotForce(LGRF, RGRF, LHS, RHS, LGS, RGS, downhill=False)
            answer = messagebox.askyesno("Question","Is data clean?")
            
        if answer == False:
            plt.close('all')
            print('Adding file to bad file list')
            badFileList.append(fName)
            
        if answer == True:
            saveFolder = fPath + 'TreadmillPlots'
            if os.path.exists(saveFolder) == False:
              os.mkdir(saveFolder) 
            plt.savefig(saveFolder + '/' + fName.split('.csv')[0] +'.png')
            plt.close('all')
            print('Estimating point estimates \n')
            
        ### Append into DF and Save if save turned on ###
outcomes = pd.DataFrame({'Subject':list(oSub), 'Config': list(oConfig),'Slope': list(oSlope),'Speed': list(oSpeed), 'Order': list(oSesh),
                                     'Side': list(Side), 'CT':list(CTs), 'VALR': list(VALRs), 'pMF':list(PkMed), 'pLF':list(PkLat),
                                     'pBF': list(peakBrakeF), 'brakeImpulse': list(brakeImpulse), 'PropImp':list(propImpulse),
                                     'pAnkEvVel': list(pAnkEvVel), 'COMWork_pos': list(COMWork_pos), 'COMWork_neg': list(COMWork_neg),
                                     'AnkWork_pos':list(AnkWork_pos), 'AnkWork_neg':list(AnkWork_neg)})

            
if save_on == 1:
    outfileName = fPath + '0_TreadmillOutcomes_test.csv'
    outcomes.to_csv(outfileName, index = False)
    
    if os.path.exists(outfileName) == False:
        
        outcomes.to_csv(outfileName, mode='a', header=True, index = False)
        badFileList.to_csv(fPath + 'BadFiles.csv', mode = 'a', header = True, index = False)
    
    else:
        outcomes.to_csv(outfileName, mode='a', header=False, index = False) 
        badFileList.to_csv(fPath + 'BadFiles.csv', mode = 'a', header = False, index = False)
    

