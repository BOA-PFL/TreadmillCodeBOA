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
steady_time = 20 #time at the start of the trial to disregard while the athlete gets up to speed [sec]
pd.options.mode.chained_assignment = None  # default='warn' set to warn for a lot of warnings

# Read in general treadmill data file
fPath = 'C:/Users/eric.honert/OneDrive - BOA Technology Inc/PFL Team - General/Testing Segments/Medical/2026_Performance_StokoII_Stoko/Treadmill/reexport/'
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

def COMPower_Work_run(GRF,speed,slope,LHS,LTO,RHS,RTO,LGS,RGS,freq):
    """
    This function computes the center-of-mass power and work for the left and
    right limbs separately using the individual limbs method. The COM
    velocity is integrated over a stride (foot contact to the subsequent foot
    contact of the same foot) from the total GRF, and the mean COM velocity
    over the stride is set to the treadmill speed (consistent with the walking
    code). Both feet land sequentially on the same force plate, so the GRF of
    the lead limb is the plate GRF during its stance phase and zero after
    toe-off (swing phase).

    Parameters
    ----------
    GRF : numpy array (Nx3)
        Ground reaction force
    speed : float or int
        running speed
    slope : float or int
        slope of the treadmill (rad)
    LHS : numpy array (Nx1)
        Left heel strike (foot contact) array
    LTO : numpy array (Nx1)
        Left toe-off array
    RHS : numpy array (Nx1)
        Right heel strike (foot contact) array
    RTO : numpy array (Nx1)
        Right toe-off array
    LGS : numpy array (Nx1)
        Array of good strides for the left side
    RGS : numpy array (Nx1)
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

    """
    # Compute the COM power using the individual limbs method

    # Debugging tool: Showing the time-continuous COM power
    show_COMpower = 0

    # First compute the approximate body weight: will need to rotate the
    # ground reaction forces into the inertial coordinate system
    BM = np.nanmean(GRF[:,1]*np.sin(slope)+GRF[:,2]*np.cos(slope))/9.81
    # Compute the COM acceleration
    acc = GRF/BM - [0,9.81*np.sin(slope),9.81*np.cos(slope)]

    # Pre-allocate variable space
    LCW_pos = []; LCW_neg = []
    RCW_pos = []; RCW_neg = []

    LCOM_power_store = np.zeros((101,len(LGS)))
    RCOM_power_store = np.zeros((101,len(RGS)))
    # Index through the good strides for computing Left COM Power + Work
    for cc,jj in enumerate(LGS):
        acc_stride = acc[LHS[jj]:LHS[jj+1],:]
        time_stride = np.array(range(len(acc_stride)))/freq
        com_vel = cumtrapz(acc_stride,time_stride,initial=0,axis=0)
        com_vel = com_vel - np.mean(com_vel,axis=0) + [0,speed,0]
        # Lead limb GRF: zero after toe-off, when the opposite foot is on the plate
        GRF_lead = np.array(GRF[LHS[jj]:LHS[jj+1],:])
        GRF_lead[LTO[jj]-LHS[jj]:,:] = 0
        # COM Power
        com_power = np.sum(com_vel*GRF_lead,axis=1)
        # Compute the positive/negative work
        # Note: this may need to be updated for level ground for collision/push-off work
        [pos_tmp,neg_temp] = findPosNegWork(com_power,freq)

        LCW_pos.append(pos_tmp)
        LCW_neg.append(neg_temp)
        # Store the time-continous COM curve
        f = scipy.interpolate.interp1d(np.arange(0,len(com_power)),com_power)
        LCOM_power_store[:,cc] = f(np.linspace(0,len(com_power)-1,101))

    # Index through the good strides for computing Right COM Power + Work
    for cc,jj in enumerate(RGS):
        acc_stride = acc[RHS[jj]:RHS[jj+1],:]
        time_stride = np.array(range(len(acc_stride)))/freq
        com_vel = cumtrapz(acc_stride,time_stride,initial=0,axis=0)
        com_vel = com_vel - np.mean(com_vel,axis=0) + [0,speed,0]
        # Lead limb GRF: zero after toe-off, when the opposite foot is on the plate
        GRF_lead = np.array(GRF[RHS[jj]:RHS[jj+1],:])
        GRF_lead[RTO[jj]-RHS[jj]:,:] = 0
        # COM Power
        com_power = np.sum(com_vel*GRF_lead,axis=1)
        # Compute the positive/negative work
        [pos_tmp,neg_temp] = findPosNegWork(com_power,freq)

        RCW_pos.append(pos_tmp)
        RCW_neg.append(neg_temp)
        # Store the time-continous COM curve
        f = scipy.interpolate.interp1d(np.arange(0,len(com_power)),com_power)
        RCOM_power_store[:,cc] = f(np.linspace(0,len(com_power)-1,101))

    # Debugging tool: examine the time-continous curves
    if show_COMpower == 1:
        plt.plot(LCOM_power_store,'k')
        plt.plot(RCOM_power_store,'r')
        plt.close() # create a breakpoint here for visualizing plots

    return(LCW_pos,LCW_neg,RCW_pos,RCW_neg,LCOM_power_store,RCOM_power_store)


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
StrideTime = []
StrideDist = []
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
            ang = 10*np.pi/180
            
            
        elif fName.count('UH'):
            speed = 1.2
            tmpCond = 'Uphill'
            tmpSlope = 10
            # Set the angle of the treadmill
            ang = 10*np.pi/180
            
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
        if landings[-1] > trimmedTakeoffs[-1]:
            trimmedLandings = landings[0:-1]
        else:
            trimmedLandings = landings
        trimmedLandings = np.array(trimmedLandings)
        trimmedTakeoffs = np.array(trimmedTakeoffs)

        ## check to make sure brake force is applied in the correct direction ##
        if np.mean(brakeFilt[landings[1]:landings[1]+100]) > 0:
            brakeFilt = -1 * brakeFilt

        # Need to eliminate bad strides
        # Assume: that most of the strides are good
        step_time = []
        for jj, val in enumerate(trimmedLandings):
            step_time.append(trimmedTakeoffs[jj] - val)

        # Crossover detection: Create a variable for "Good Steps"
        # Note: the last statement is to ensure that steady state running is
        # attained: the athlete is often getting up to speed during the first
        # 20 seconds of the trial, so those steps are disregarded.
        # Note: some trials are exported without the right force plate (all
        # NaN). The crossover check cannot be performed for these trials, so
        # rely on the stride checks below (a crossover onto the right belt
        # shows up as a missing contact on the left force plate).
        RFP_avail = dat.Right_GRF_Z.isna().all() == False
        if RFP_avail == False:
            print('Right force plate data not available: crossover check not performed')

        good_step = np.zeros(len(trimmedLandings),dtype=bool)
        for jj, val in enumerate(trimmedLandings):
            if RFP_avail:
                no_crossover = np.min(dat.Right_GRF_Z[val:trimmedTakeoffs[jj]]) < fThresh
            else:
                no_crossover = True
            if step_time[jj] < np.median(step_time) + 20 and no_crossover and val > steady_time*freq:
                good_step[jj] = True

        #______________________________________________________________
        # Left/Right Step Segmentation
        # Both feet step sequentially on the same force plate. Sum the M/L GRF
        # (X) over each stance phase: with Z vertical (up) and Y forward (A/P),
        # X points laterally to the right, so the right foot, which pushes
        # medially, has a negative M/L impulse. Right = negative.
        ML_imp = np.zeros(len(trimmedLandings))
        for jj, val in enumerate(trimmedLandings):
            ML_imp[jj] = np.sum(LGRF[val:trimmedTakeoffs[jj],0])/freq

        # The sign of a single step's M/L impulse is not reliable for runners
        # with a narrow step width, and an M/L force offset biases every step
        # towards one side. Compare each step to the average of its
        # neighbouring steps to remove the offset/drift.
        ML_nb = np.zeros(len(ML_imp))
        for jj in range(1,len(ML_imp)-1):
            ML_nb[jj] = ML_imp[jj] - (ML_imp[jj-1] + ML_imp[jj+1])/2
        ML_nb[0] = ML_imp[0] - ML_imp[1]; ML_nb[-1] = ML_imp[-1] - ML_imp[-2]

        # Note: for downhill (backwards walking) the participant faces the
        # opposite direction on the treadmill, so the M/L sign flips with
        # respect to the body and the left/right labels are switched
        if tmpCond == 'Downhill':
            ML_nb_body = -ML_nb
        else:
            ML_nb_body = ML_nb

        # Running on a single force plate: foot contacts strictly alternate
        # (checked on the 2025 Kailas data: no missed contacts in 40 trials).
        # Assign sides by alternation and use a majority vote of the good
        # steps' M/L impulses to decide whether the even steps are right or left.
        even_step = np.arange(len(ML_nb_body)) % 2 == 0
        vote_even_right = (ML_nb_body < 0) == even_step
        vote_even_right = vote_even_right[good_step]
        vote_margin = np.mean(vote_even_right)

        step_side = []
        for jj in range(len(ML_nb_body)):
            if even_step[jj] == (vote_margin > 0.5):
                step_side.append('Right')
            else:
                step_side.append('Left')
        step_side = np.array(step_side)
        # A vote margin close to 50% means that the side assignment is uncertain
        print('L/R vote margin: ' + str(round(max(vote_margin,1-vote_margin)*100,1)) + '%')

        L_idx = np.where(step_side == 'Left')[0]
        R_idx = np.where(step_side == 'Right')[0]
        LHS = trimmedLandings[L_idx]; RHS = trimmedLandings[R_idx]
        LTO = trimmedTakeoffs[L_idx]; RTO = trimmedTakeoffs[R_idx]
        L_good_step = good_step[L_idx]; R_good_step = good_step[R_idx]

        # GS: good strides - the step must be a good step, the stride
        # (foot contact to subsequent foot contact of the same foot) must have
        # exactly 1 opposite foot contact within it (checks that the left/right
        # segmentation alternates and no contacts were missed), and the stride
        # time must be physiologically reasonable
        timemin = 0.3
        timemax = 2
        LGS = []
        for jj in range(len(LHS)-1):
            n_opp = np.sum((RHS > LHS[jj]) & (RHS < LHS[jj+1]))
            if L_good_step[jj] and n_opp == 1 and LHS[jj+1] - LHS[jj] > timemin*freq and LHS[jj+1] - LHS[jj] < timemax*freq:
                LGS.append(jj)
        LGS = np.array(LGS)

        RGS = []
        for jj in range(len(RHS)-1):
            n_opp = np.sum((LHS > RHS[jj]) & (LHS < RHS[jj+1]))
            if R_good_step[jj] and n_opp == 1 and RHS[jj+1] - RHS[jj] > timemin*freq and RHS[jj+1] - RHS[jj] < timemax*freq:
                RGS.append(jj)
        RGS = np.array(RGS)

        # Compute COM work
        [LCW_pos,LCW_neg,RCW_pos,RCW_neg,LCOMdebug,RCOMdebug] = COMPower_Work_run(LGRF,speed,ang,LHS,LTO,RHS,RTO,LGS,RGS,freq)

        #______________________________________________________________
        # Debugging: Creation of dialog box for looking where foot contact are accurate
        answer = True # Defaulting to true: In case "debug" is not used
        # Debugging plots: Left = black, Right = red
        if debug == 1:
            plt.figure(figsize=(16,5))
            plt.subplot(1,4,1)
            plt.plot(intp_steps(LGRF[:,1],LHS,LTO,LGS),'k')
            plt.plot(intp_steps(LGRF[:,1],RHS,RTO,RGS),'r')
            plt.ylabel('A/P GRF [N]')
            plt.xlabel('% Step')

            plt.subplot(1,4,2)
            plt.plot(intp_steps(LGRF[:,2],LHS,LTO,LGS),'k')
            plt.plot(intp_steps(LGRF[:,2],RHS,RTO,RGS),'r')
            plt.ylabel('Vertical GRF [N]')
            plt.xlabel('% Step')

            plt.subplot(1,4,3)
            plt.plot(LCOMdebug,'k')
            plt.plot(RCOMdebug,'r')
            plt.ylabel('COM Power [W]')
            plt.xlabel('% Stride')

            # Check the left/right segmentation: most left steps should sit
            # above zero and most right steps below zero
            plt.subplot(1,4,4)
            plt.plot(L_idx,ML_nb[L_idx],'ko')
            plt.plot(R_idx,ML_nb[R_idx],'ro')
            plt.axhline(0,color='gray')
            plt.ylabel('M/L Impulse - Neighbours [N*s]')
            plt.xlabel('Step #')
            plt.tight_layout()

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

            # Index through the left good strides
            # Note: every value for a stride is computed before any list is
            # appended so that an exception cannot misalign the output rows
            for cc, jj in enumerate(LGS):
                    try:
                        # Compute force-based metrics
                        # Loading Rate: used for fit purposes, not injury. Great Easter Egg, Eric
                        tmpVALR = calcVLR(forceZ, LHS[jj]+1, 150, timeToLoad, freq)
                        # Contact Time
                        tmpCT = (LTO[jj] - LHS[jj])/freq
                        # Stride time and distance: foot contact to the
                        # subsequent foot contact of the same foot
                        tmpStrideTime = (LHS[jj+1] - LHS[jj])/freq
                        tmpStrideDist = tmpStrideTime*abs(speed)
                        # Peak Medial/Lateral forces
                        tmpPkMed = np.max(MForce[LHS[jj]:LTO[jj]])
                        tmpPkLat = np.min(MForce[LHS[jj]:LTO[jj]])
                        # Braking and Propulsive Force Metrics
                        tmpBrakeImp = sum(i for i in brakeFilt[LHS[jj]:LTO[jj]] if i < 0)/freq #sum all negative brake force vals
                        tmpPropImp = sum(i for i in brakeFilt[LHS[jj]:LTO[jj]] if i > 0)/freq #sum all positive values
                        tmpPkBrake = np.min(brakeFilt[LHS[jj]:LTO[jj]])

                        VALRs.append(tmpVALR)
                        CTs.append(tmpCT)
                        StrideTime.append(tmpStrideTime)
                        StrideDist.append(tmpStrideDist)
                        PkMed.append(tmpPkMed)
                        PkLat.append(tmpPkLat)
                        brakeImpulse.append(tmpBrakeImp)
                        propImpulse.append(tmpPropImp)
                        peakBrakeF.append(tmpPkBrake)
                        COMWork_pos.append(LCW_pos[cc])
                        COMWork_neg.append(LCW_neg[cc])

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
                        Side.append('Left')
                    except:
                        print(LHS[jj])

            # Index through the right good strides
            for cc, jj in enumerate(RGS):
                    try:
                        # Compute force-based metrics
                        tmpVALR = calcVLR(forceZ, RHS[jj]+1, 150, timeToLoad, freq)
                        tmpCT = (RTO[jj] - RHS[jj])/freq
                        tmpStrideTime = (RHS[jj+1] - RHS[jj])/freq
                        tmpStrideDist = tmpStrideTime*abs(speed)
                        tmpPkMed = np.max(MForce[RHS[jj]:RTO[jj]])
                        tmpPkLat = np.min(MForce[RHS[jj]:RTO[jj]])
                        tmpBrakeImp = sum(i for i in brakeFilt[RHS[jj]:RTO[jj]] if i < 0)/freq
                        tmpPropImp = sum(i for i in brakeFilt[RHS[jj]:RTO[jj]] if i > 0)/freq
                        tmpPkBrake = np.min(brakeFilt[RHS[jj]:RTO[jj]])

                        VALRs.append(tmpVALR)
                        CTs.append(tmpCT)
                        StrideTime.append(tmpStrideTime)
                        StrideDist.append(tmpStrideDist)
                        PkMed.append(tmpPkMed)
                        PkLat.append(tmpPkLat)
                        brakeImpulse.append(tmpBrakeImp)
                        propImpulse.append(tmpPropImp)
                        peakBrakeF.append(tmpPkBrake)
                        COMWork_pos.append(RCW_pos[cc])
                        COMWork_neg.append(RCW_neg[cc])

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
                        Side.append('Right')
                    except:
                        print(RHS[jj])

outcomes = pd.DataFrame({'Subject':list(oSub), 'Config': list(oConfig),'Slope': list(oSlope),'Speed': list(oSpeed), 'Order': list(oSesh),
                         'Side': list(Side), 'CT':list(CTs), 'StrideTime': list(StrideTime), 'StrideDist': list(StrideDist), 'VALR': list(VALRs), 'pMF':list(PkMed), 'pLF':list(PkLat),
                         'pBF': list(peakBrakeF), 'brakeImpulse': list(brakeImpulse), 'PropImp':list(propImpulse),
                         'pAnkEvVel': list(pAnkEvVel), 'COMWork_pos': list(COMWork_pos), 'COMWork_neg': list(COMWork_neg),
                         'AnkWork_pos': list(AnkWork_pos), 'AnkWork_neg': list(AnkWork_neg)})
                          

if save_on == 1:
    outfileName = fPath + '0_TreadmillOutcomes_v3.csv'
    badfileName = fPath + 'BadFiles.csv'
    # Note: badFileList is a list, so convert it to a dataframe for saving
    badFileDF = pd.DataFrame({'BadFiles': badFileList})

    # Write a new file with a header if one does not exist, otherwise append
    if os.path.exists(outfileName) == False:
        outcomes.to_csv(outfileName, mode='a', header=True, index = False)
    else:
        outcomes.to_csv(outfileName, mode='a', header=False, index = False)

    if os.path.exists(badfileName) == False:
        badFileDF.to_csv(badfileName, mode = 'a', header = True, index = False)
    else:
        badFileDF.to_csv(badfileName, mode = 'a', header = False, index = False)




