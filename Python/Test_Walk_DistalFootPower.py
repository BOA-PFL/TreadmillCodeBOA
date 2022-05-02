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


# Define constants and options
run = 1 # Set this to 1 where participant is running on one belt so only the left are detected. 0 for dual belt
manualTrim = 0  #set this to 1 if you want to manually trim trials with ginput, 0 if you want it auto trimmed (start and end of trial)
fThresh = 50 #below this value will be set to 0.
writeData = 0 #will write to spreadsheet if 1 entered
plottingEnabled = 0 #plots the bottom if 1. No plots if 0
lookFwd = 50
timeToLoad = 75 #length to look forward for an impact peak
pd.options.mode.chained_assignment = None  # default='warn' set to warn for a lot of warnings
save_on = 0 # Turn to 1 to save outcomes to csv

# Look at the text files from the foot work 
fPath_footwork = 'C:\\Users\\eric.honert\\Boa Technology Inc\\PFL Team - General\\Testing Segments\\WorkWear_Performance\\Elten_Jan2022\\Treadmill\FootPower\\'
entries_footwork = os.listdir(fPath_footwork)
   
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
    

def dist_seg_power_treadmill(inputDF,speed,slope,landings,takeoffs):
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
    slope : int or float
        Slope of the treadmill
    landings : list
        Initial foot contact
    takeoffs : list
        Or toe-offs
    
    Returns
    -------
    power : numpy array
        distal rearfoot power

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
    
    # Index through the landings
    for ii in range(len(landings)-1):
        stepframe = takeoffs[ii]-landings[ii]
        # Frames to analyze based on the foot flat percentages
        FFframes = range(landings[ii]+round(foot_flat[0]*stepframe),landings[ii]+round(foot_flat[1]*stepframe),1)
        step_speed[ii] = np.mean(inputDF.FootCOMVel_Y[FFframes])

    # Find the average treadmill belt speed of the trial (also exclude any 
    # zeros in the estimate)
    avg_speed = -np.mean(step_speed[step_speed != 0])
    
    if debug == 1:
        plt.figure(1010)
        plt.plot(step_speed)
    
    # Treadmill belt speed: will need to be updated based on the slope
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
    intp_var = np.zeros((101,len(landings)-1))
    # Index through the strides
    for ii in range(len(landings)-1):
        dum = var[landings[ii]:landings[ii+1]]
        f = scipy.interpolate.interp1d(np.arange(0,len(dum)),dum)
        intp_var[:,ii] = f(np.linspace(0,len(dum)-1,101))
        
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

        # Zero the forces below the threshold
        forceZ = trimForce(dat, fThresh)
        
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
        DFootPower = dist_seg_power_treadmill(dat,1.2,0,landings,takeoffs)
        
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
    except:
        print(fName)
        
outcomes = pd.DataFrame({'Subject':list(Subject), 'Config': list(Config),'DisWork': list(DisWork)})

if save_on == 1:
    outcomes.to_csv("C:\\Users\\eric.honert\\Boa Technology Inc\\PFL Team - General\\Segments\\WorkWear_Performance\\Elten_Jan2022\\Treadmill\\FootWork.csv",mode='a',header=True)


