# -*- coding: utf-8 -*-
"""
Created on Wed March 2022

This code has been augmented from previous treadmill-based codes to test the 
distal rearfoot power computations. 

@author: Eric.Honert
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.signal as sig
from scipy.fft import fft, fftfreq
import scipy
import addcopyfighandler
from scipy.integrate import cumtrapz
from tkinter import messagebox


# Define constants and options
fThresh = 50 #below this value will be set to 0.
lookFwd = 50
timeToLoad = 75 #length to look forward for an impact peak
pd.options.mode.chained_assignment = None  # default='warn' set to warn for a lot of warnings


#______________________________________________________________________________
# File management

# load the running speed file for reference
SubRunSpeed = pd.read_csv('C:\\Users\eric.honert\\Boa Technology Inc\\PFL Team - General\\Testing Segments\\EndurancePerformance\\TrailRun_2022\\InLabSetSubSpeed.csv')

# Look at the text files from the foot work 
fPath_footwork = 'C:\\Users\eric.honert\\Boa Technology Inc\\PFL Team - General\\Testing Segments\\EndurancePerformance\\TrailRun_2022\\InLabData\\TreadmillData\\'
entries_footwork = [fName for fName in os.listdir(fPath_footwork) if fName.endswith('DistalRearfootPower.txt')]
# Look at the text files from the rest of the kinematics/kinetics
fPath_kin = 'C:\\Users\eric.honert\\Boa Technology Inc\\PFL Team - General\\Testing Segments\\EndurancePerformance\\TrailRun_2022\\InLabData\\TreadmillData\\'
entries_kin = [fName for fName in os.listdir(fPath_footwork) if fName.endswith('PerformanceTestData_V2.txt')]

#______________________________________________________________________________
save_on = 1 # Turn to 1 to save outcomes to csv
debug = 0


   
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
    forceTot = inputDF.GRF_Z
    forceTot[forceTot<threshForce] = 0
    forceTot = np.array(forceTot)
    return(forceTot)

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
    

def dist_seg_power_treadmill(Seg_COM_Pos,Seg_COM_Vel,Seg_Ang_Vel,CenterOfPressure,GRF,FreeMoment,speed,landings,takeoffs,yn_run):
    """
    The purpose of this function is to compute the distal segment power -
    commonly applied to the rearfoot to obtain the distal rearfoot power. The
    power in this formation was provided in Takahashi et al. 2012; but
    originally in Siegel et al. 1996. For full derivations, see Zelik and 
    Honert 2018 appendix. This code assumes that the walking direction is +y

    Parameters
    ----------
    inputDF : DataFrame
        This data frame needs to contain the following variables:
            Seg_COM_Vel = Segment COM Velocity (ex: Foot COM Velocity)
            Seg_Ang_Vel = Segment Angular Velocity (ex: Foot Angular Velocity)
            Seg_COM_Pos = Segment COM Position (ex: Foot COM Position)
            CenterOfPressure = Location of the center of pressure
            FreeMoment = Free moment on force platform
            GRF = Ground Reaction Force. Ensure that the input GRF is the REACTION
            force; whereas, motion monitor exports the action force
    speed : float
        Treadmill belt speed - can be used as a debugging variable or to set
        the speed of the foot in 3D space. 
    landings : list
        Initial foot contact
    takeoffs : list
        Or toe-offs
    
    Returns
    -------
    power : numpy array
        distal rearfoot power

    """

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

def intp_steps(var,fHS,fTO,GS):
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
    intp_var = np.zeros((101,len(GS)))
    # Index through the strides
    for jj, ii in enumerate(GS):
        dum = var[fHS[ii]:fTO[ii]+1]
        f = scipy.interpolate.interp1d(np.arange(0,len(dum)),dum)
        intp_var[:,jj] = f(np.linspace(0,len(dum)-1,101))
        
    return intp_var

def crop_strides_fft(var,landings):
    """
    Function to crop the intended variable into strides, pad the strides with 
    zeros and perform the FFT on the variable of interest

    Parameters
    ----------
    var : list or numpy array
        Variable of interest
    landings : list
        foot-contact or landing indices

    Returns
    -------
    fft_out : numpy array
        FFT of the variable of interest during the stride
    xf : numpy array
        Frequency vector from the FFT in [Hz]

    """
    # Preallocate
    intp_var = np.zeros((500,len(landings)-1))
    fft_out = np.zeros((500,len(landings)-1))
    # Index through the strides
    for ii in range(len(landings)-1):
        intp_var[0:landings[ii+1]-landings[ii],ii] = var[landings[ii]:landings[ii+1]]
        fft_out[:,ii] = fft(intp_var[:,ii])
        xf = fftfreq(500,1/200)
        
    return [fft_out,xf]

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

def COMPower_Work_run(GRF,slope,HS,TO,GoodStrides,speed,freq):
    """
    This function computes the center-of-mass power and work for the "leading"
    limb (ie the limb that is used to segment the GRFs)

    Parameters
    ----------
    GRF : numpy array (Nx3)
        Ground reaction force
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
    COM_power_store : numpy array
        The time-continuous COM power from each good step
    BM : float
        subject body mass from the particular trial

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
    
    COM_power_store = np.zeros((101,len(GoodStrides)))
    # Index through the good strides for computing COM Power + Work
    for cc,jj in enumerate(GoodStrides):
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
    
    return(CW_pos,CW_neg,COM_power_store,BM)




#Preallocate variables for storage
Subject = []
Config = []
SetSpeed = []
SetSlope = []
Sesh = []

VALRs = []

NegFootWork = []
PosFootWork = []
NegAnkWork = []
PosAnkWork = []
PosCOMWork = []
NegCOMWork = []
pAnkEvVel = []
FootConAng = []
AnkConAng = []

MaxFootPower = []
MinFootPower = []

footp4 = np.array([])
footp2 = np.array([])
footn6 = np.array([])

ankp4 = np.array([])
ankp2 = np.array([])
ankn6 = np.array([])

pfs_foot = np.array([])
lace_foot = np.array([])
sock_foot = np.array([])

pfs_footang = np.array([]) 
lace_footang = np.array([])
sock_footang = np.array([])

pfs_ankang = np.array([])
lace_ankang = np.array([])
sock_ankang = np.array([])


badFileList = []

#______________________________________________________________________________
for ii in range(0,len(entries_footwork)):
    # try:
        
        #_____________________________________________________
        # Load the files associated with the foot power/work
        fName = entries_footwork[ii]
        print(fName)
        
        #Parse file name into subject and configuration 
        SubjectTmp = fName.split(sep = "_")[0]
        ConfigTmp = fName.split(sep = "_")[1]
        SpeedTmp = fName.split(sep = "_")[2]
        SlopeTmp = fName.split(sep = "_")[3]
        SeshTmp = fName.split(sep = "_")[4][0]

        dat = pd.read_csv(fPath_footwork+fName,sep='\t', skiprows = 8, header = 0)
        datKin = pd.read_csv(fPath_kin+entries_kin[ii],sep='\t', skiprows = 8, header = 0)
        # Obtain the treadmill belt speed from the reference file
        if SpeedTmp == 'ss' and SlopeTmp == 'n6':
            beltspeed = -3
        elif SpeedTmp == 'ss' and SlopeTmp == 'p4':
            beltspeed = 3
        elif SpeedTmp == 'ss' and SlopeTmp == 'p2':
            beltspeed = 3
        else: 
            beltspeed = float(SubRunSpeed[(SubRunSpeed['Subject']==SubjectTmp) & (SubRunSpeed['Slope']==SlopeTmp)].Speed)
            
        print(beltspeed)
        
        #______________________________________________________________________
        # Determine gait events:
        # Zero the forces below the threshold
        forceZ = trimForce(dat, fThresh)
        forceZ = forceZ
        
        # Find the landings and takeoffs of the FP as vectors: this will find all landings
        landings = np.array(findLandings(forceZ))
        takeoffs = np.array(findTakeoffs(forceZ))
        if landings[-1] > takeoffs[-1]:
            landings = landings[:-1]
        
        # Assign signals based on direction of running
        if SlopeTmp[0] == 'n':
            fc_sig = dat.RFootCOMPos_Z
            foot_drop_sig = np.array(dat.RFootPosDetect)
            shank_drop_sig = np.array(datKin.RShankPosDetect)
            ank_power = datKin.RightAnklePower
            ank_front_vel = datKin.RAnkleAngVel_Frontal
            foot_ang = datKin.RFootStrikeAngle
            ank_sag_ang = datKin.RAnkleAngle_Sagittal
        elif SlopeTmp[0] == 'p':
            fc_sig = dat.LFootCOMPos_Z
            foot_drop_sig = np.array(dat.LFootPosDetect)
            shank_drop_sig = np.array(datKin.LShankPosDetect)
            ank_power = datKin.LeftAnklePower
            ank_front_vel = -datKin.LAnkleAngVel_Frontal
            foot_ang = datKin.LFootStrikeangle
            ank_sag_ang = datKin.LAnkleAngle_Sagittal
        # Need to make sure the appropriate foot is close to the ground
        HS = []
        TO = []
        for jj in landings:
            if fc_sig[jj] < 0.5*np.max(fc_sig):
                HS.append(jj)
                idx = takeoffs > jj
                dum = takeoffs[idx]
                TO.append(dum[0])
        
        HS = np.array(HS)
        TO = np.array(TO)
        
        # Find the 3 hops
        approx_CT = np.diff(landings)
        # Counters
        jc = 0  # jump counter
        stc = 0 # start trial counter
        jj = 0
        while stc == 0:
            if approx_CT[jj] < 300:
                jc = jc+1
            if jc >= 1 and approx_CT[jj] > 300:
                last_jumpHS = landings[jj]
                idx = (HS > (last_jumpHS + 10*200))*(HS < (last_jumpHS + 55*200))
                HS = HS[idx]
                idx = (TO > (last_jumpHS + 10*200))*(TO < (last_jumpHS + 55*200))
                TO = TO[idx]
                stc = 1
            jj = jj+1

        # Fill the nan's with 0 => after the foot contacts have been detected
        dat = dat.fillna(0)
        
        # Crop the landings and takeoffs
        if TO[0] < HS[0]:
            TO = TO[1:]        
        
        if TO[-1] < HS[-1]:
            HS = HS[0:-1]
        
        
        # Combine the necessary signals for the distal rearfoot power
        # Make Variables NumPy Arrays for matrix opperations
        if SlopeTmp[0] == 'n':
            Foot_Ang_Vel = np.array(list(zip(dat.RFootAngVel_X,dat.RFootAngVel_Y,dat.RFootAngVel_Z)))*(np.pi/180)    
            Foot_COM_Pos = np.array(list(zip(dat.RFootCOMPos_X,dat.RFootCOMPos_Y,dat.RFootCOMPos_Z)))
            Foot_COM_Vel = np.array(list(zip(dat.RFootCOMVel_X,dat.RFootCOMVel_Y,dat.RFootCOMVel_Z)))
        elif SlopeTmp[0] == 'p':
            Foot_Ang_Vel = np.array(list(zip(dat.LFootAngVel_X,dat.LFootAngVel_Y,dat.LFootAngVel_Z)))*(np.pi/180)    
            Foot_COM_Pos = np.array(list(zip(dat.LFootCOMPos_X,dat.LFootCOMPos_Y,dat.LFootCOMPos_Z)))
            Foot_COM_Vel = np.array(list(zip(dat.LFootCOMVel_X,dat.LFootCOMVel_Y,dat.LFootCOMVel_Z)))
        
        
        COP = np.array(list(zip(dat.COP_X,dat.COP_Y,dat.COP_Z)))
        GRF = np.array(list(zip(dat.GRF_X,-dat.GRF_Y,dat.GRF_Z)))
        FMOM = np.array(list(zip(dat.FMOM_X,dat.FMOM_Y,dat.FMOM_Z)))
        
        
        # Compute the distal rearfoot power:
        DFootPower = dist_seg_power_treadmill(Foot_COM_Pos,Foot_COM_Vel,Foot_Ang_Vel,COP,GRF,FMOM,beltspeed,landings,takeoffs,1)
        
        GS = []
        for jj, landing in enumerate(HS):
            if sum(np.isnan(foot_drop_sig[landing:TO[jj]])) == 0 and np.max(abs(ank_power[landing:TO[jj]+1])) < 3000 and np.max(abs(DFootPower[landing:TO[jj]+1])) < 3000 and sum(np.isnan(shank_drop_sig[landing:TO[jj]])) == 0:
                GS.append(jj)
        
        [PW,NW,COMpower,mass] = COMPower_Work_run(GRF,float(SlopeTmp[1]),HS,TO,GS,beltspeed,200)       
        #______________________________________________________________
        # Debugging: Creation of dialog box for looking where foot contact are accurate
        answer = True # Defaulting to true: In case "debug" is not used
        # Debugging plots:
        if debug == 1:
            plt.figure()
            plt.subplot(1,3,1)
            plt.plot(intp_steps(DFootPower,HS,TO,GS))
            plt.xlabel('% Step')
            plt.title('Distal Rearfoot Power')
            
            plt.subplot(1,3,2)
            plt.plot(intp_steps(ank_power,HS,TO,GS))
            plt.xlabel('% Step')
            plt.title('Ankle Power')
            
            plt.subplot(1,3,3)
            plt.plot(COMpower)
            plt.xlabel('% Step')
            plt.title('COM Power')
            answer = messagebox.askyesno("Question","Is data clean?")
            plt.close()
            if answer == False:
                print('Adding file to bad file list')
                badFileList.append(fName)
            
        if answer == True:
            # Append the COM work
            PosCOMWork.extend(PW)
            NegCOMWork.extend(NW)
            for jj in GS:
                # Compute force-based metrics
                # Loading Rate: used for fit purposes, not injury. Great Easter Egg, Eric
                VALRs.append(calcVLR(GRF[:,2], HS[jj]+1, 150, timeToLoad, 200))
                # Eversion Velocity
                idx20 = round(0.2*(TO[jj] - HS[jj])) + HS[jj]
                pAnkEvVel.append(abs(np.min(ank_front_vel[HS[jj]-20:idx20])))
                # Look at what's going on with the ankle and foot at contact
                FootConAng.append(foot_ang[HS[jj]])
                AnkConAng.append(ank_sag_ang[HS[jj]])
                # Max, min foot
                MaxFootPower.append(max(DFootPower[HS[jj]:TO[jj]]))
                MinFootPower.append(min(DFootPower[HS[jj]:TO[jj]]))
                # Compute joint work
                [PW,NW] = findPosNegWork(DFootPower[HS[jj]:TO[jj]],200)
                NegFootWork.append(NW)
                PosFootWork.append(PW)
                [PW,NW] = findPosNegWork(ank_power[HS[jj]:TO[jj]],200)
                NegAnkWork.append(NW)
                PosAnkWork.append(PW)
                
                Subject.append(SubjectTmp)
                Config.append(ConfigTmp)
                SetSpeed.append(abs(beltspeed))
                SetSlope.append(SlopeTmp)
                Sesh.append(SeshTmp)
    
        # Store the strides for later plotting
        if SpeedTmp == 'ss' and SlopeTmp == 'n6':
            footn6 = np.append(footn6,np.nanmean(intp_steps(DFootPower,HS,TO,GS),axis=1)/mass)
            ankn6 = np.append(ankn6,np.nanmean(intp_steps(ank_power,HS,TO,GS),axis=1)/mass)
        elif SpeedTmp == 'ss' and SlopeTmp == 'p4':
            footp4 = np.append(footp4,np.nanmean(intp_steps(DFootPower,HS,TO,GS),axis=1)/mass)
            ankp4 = np.append(ankp4,np.nanmean(intp_steps(ank_power,HS,TO,GS),axis=1)/mass)
        elif SpeedTmp == 'ss' and SlopeTmp == 'p2':
            footp2 = np.append(footp2,np.nanmean(intp_steps(DFootPower,HS,TO,GS),axis=1)/mass)
            ankp2 = np.append(ankp2,np.nanmean(intp_steps(ank_power,HS,TO,GS),axis=1)/mass)
        
        if SpeedTmp == 'ss' and SlopeTmp == 'n6' and ConfigTmp == 'pfs':
            pfs_foot = np.append(pfs_foot,np.nanmean(intp_steps(DFootPower,HS,TO,GS),axis=1)/mass)
            pfs_footang = np.append(pfs_footang,np.nanmean(intp_steps(foot_ang,HS,TO,GS),axis=1))
            pfs_ankang = np.append(pfs_ankang,np.nanmean(intp_steps(ank_sag_ang,HS,TO,GS),axis=1))
        elif SpeedTmp == 'ss' and SlopeTmp == 'n6' and ConfigTmp == 'lace':
            lace_foot = np.append(lace_foot,np.nanmean(intp_steps(DFootPower,HS,TO,GS),axis=1)/mass)
            lace_footang = np.append(lace_footang,np.nanmean(intp_steps(foot_ang,HS,TO,GS),axis=1))
            lace_ankang = np.append(lace_ankang,np.nanmean(intp_steps(ank_sag_ang,HS,TO,GS),axis=1))
        elif SpeedTmp == 'ss' and SlopeTmp == 'n6' and ConfigTmp == 'sock':
            sock_foot = np.append(sock_foot,np.nanmean(intp_steps(DFootPower,HS,TO,GS),axis=1)/mass)
            sock_footang = np.append(sock_footang,np.nanmean(intp_steps(foot_ang,HS,TO,GS),axis=1))
            sock_ankang = np.append(sock_ankang,np.nanmean(intp_steps(ank_sag_ang,HS,TO,GS),axis=1))
        


footn6 = np.reshape(footn6,[int(len(footn6)/101),101])       
ankn6 = np.reshape(ankn6,[int(len(ankn6)/101),101])

footp2 = np.reshape(footp2,[int(len(footp2)/101),101])     
ankp2 = np.reshape(ankp2,[int(len(ankp2)/101),101]) 

footp4 = np.reshape(footp4,[int(len(footp4)/101),101])     
ankp4 = np.reshape(ankp4,[int(len(ankp4)/101),101])

pfs_foot = np.reshape(pfs_foot,[int(len(pfs_foot)/101),101]) 
lace_foot = np.reshape(lace_foot,[int(len(lace_foot)/101),101])
sock_foot = np.reshape(sock_foot,[int(len(sock_foot)/101),101]) 

pfs_footang = np.reshape(pfs_footang,[int(len(pfs_footang)/101),101]) 
lace_footang = np.reshape(lace_footang,[int(len(lace_footang)/101),101])
sock_footang = np.reshape(sock_footang,[int(len(sock_footang)/101),101]) 

pfs_ankang = np.reshape(pfs_ankang,[int(len(pfs_ankang)/101),101]) 
lace_ankang = np.reshape(lace_ankang,[int(len(lace_ankang)/101),101])
sock_ankang = np.reshape(sock_ankang,[int(len(sock_ankang)/101),101])       
              
outcomes = pd.DataFrame({'Subject':list(Subject), 'Config': list(Config), 'SetSpeed': list(SetSpeed), 'SetSlope': list(SetSlope), 'Sesh': list(Sesh),'NegFootWork': list(NegFootWork),'PosFootWork': list(PosFootWork),
                         'NegAnkWork': list(NegAnkWork),'PosAnkWork': list(PosAnkWork),'NegCOMWork': list(NegCOMWork),'PosCOMWork': list(PosCOMWork), 'pAnkEvVel': list(pAnkEvVel), 'VALR': list(VALRs),
                         'FootConAng': list(FootConAng), 'AnkConAng': list(AnkConAng),'MaxFootPower': list(MaxFootPower), 'MinFootPower': list(MinFootPower)})

if save_on == 1:
    outcomes.to_csv('C:\\Users\eric.honert\\Boa Technology Inc\\PFL Team - General\\Testing Segments\\EndurancePerformance\\TrailRun_2022\\InLabData\\AnkleFootWork.csv',header=True)
elif save_on == 2: 
    outcomes.to_csv('C:\\Users\eric.honert\\Boa Technology Inc\\PFL Team - General\\Testing Segments\\EndurancePerformance\\TrailRun_2022\\InLabData\\AnkleFootWork.csv', mode = 'a', header=False)



# Plotting
plt.figure
plt.subplot(1,2,1)
plt.plot(np.mean(footn6,axis=0))
plt.plot(np.mean(footp2,axis=0))
plt.plot(np.mean(footp4,axis=0))
plt.xlim([0,100])
plt.ylim([-15,15])
plt.legend(['n6','p2','p4'])
plt.title('Distal Rearfoot')

plt.subplot(1,2,2)
plt.plot(np.mean(ankn6,axis=0))
plt.plot(np.mean(ankp2,axis=0))
plt.plot(np.mean(ankp4,axis=0))
plt.xlim([0,100])
plt.ylim([-15,15])
plt.title('Ankle')

plt.figure
plt.plot(np.mean(footn6,axis=0))
plt.plot(np.mean(footp2,axis=0))
plt.plot(np.mean(footp4,axis=0))
plt.xlim([0,100])
# plt.ylim([-15,10])
plt.legend(['n6','p2','p4'])
plt.title('Distal Rearfoot')

plt.figure(10)
plt.plot(np.mean(pfs_foot,axis=0))
plt.plot(np.mean(lace_foot,axis=0))
plt.plot(np.mean(sock_foot,axis=0))
plt.legend(['pfs','lace','sock'])

plt.figure(11)
plt.plot(np.mean(pfs_footang,axis=0))
plt.plot(np.mean(lace_footang,axis=0))
plt.plot(np.mean(sock_footang,axis=0))
plt.legend(['pfs','lace','sock'])

plt.figure(12)
plt.plot(np.mean(pfs_ankang,axis=0))
plt.plot(np.mean(lace_ankang,axis=0))
plt.plot(np.mean(sock_ankang,axis=0))
plt.legend(['pfs','lace','sock'])


