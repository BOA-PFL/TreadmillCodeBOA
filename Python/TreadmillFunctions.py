# -*- coding: utf-8 -*-
"""
Shared functions for treadmill-based biomechanical analyses. These functions
are used by both Performance_Walk_ExtractKinematicKineticVars.py and
Performance_Run_ExtractKinematicKineticVars.py.

@author: Daniel.Feeney, Eric.Honert
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import scipy
import scipy.signal as sig
from datetime import datetime


def findLandings(force, fThresh):
    """
    The purpose of this function is to determine the landings (foot contacts)
    events on the force plate when the filtered vertical ground reaction force
    exceeds the force threshold

    Parameters
    ----------
    force : list
        vertical ground reaction force.
    fThresh : float
        force threshold for detecting foot contact

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

def findTakeoffs(force, fThresh):
    """
    Find takeoff from FP when force goes from above thresh to 0

    Parameters
    ----------
    force : list
        vertical ground reaction force
    fThresh : float
        force threshold for detecting take-off

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
    preForce = np.zeros((noSteps,stepLength))

    for iterVar, landing in enumerate(landings):
        try:
            preForce[iterVar,] = inputForce[landing:landing+stepLength]
        except:
            print(landing)

    return preForce

def findPosNegWork(var, freq):
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


#______________________________________________________________________________
# Processing-decision helpers.
# Note: these two functions are kept byte-identical to the copies in
# XSENSOR/XSENSORFunctions.py and IMU_BOA/IMUFunctions.py. There is no
# shared package across the BOA-PFL repos, so if you change one, change the
# others in the same commit.


def loadProcessingDecisions(fPath, scriptName):
    """
    Load the data-quality decisions recorded by previous runs of this script.

    Decisions live in 0_ProcessingDecisions_<scriptName>.csv beside the data.
    The name carries the script because one data folder is often read by
    several scripts, and they do not judge the same thing - a trial that is
    unusable for one analysis can be perfectly good for another. A shared file
    would let one script's rejection silently govern the rest.

    The file is append-only: correcting a decision means adding a later row for
    the same file rather than editing the original, so the record of what was
    judged and when is preserved. The most recent row for a file is the one
    that counts.

    Reading these back is what stops a trial being judged a second time. A file
    already marked good is processed without the dialog box appearing again.

    Rejected files are left where they are. Nothing is moved out of the folder,
    so a rejected trial can still be opened and looked at; it is this record,
    not the folder, that decides what gets processed.

    Parameters
    ----------
    fPath : str
        Path to the folder holding the data and the decision file
    scriptName : str
        Short name of the calling script, set as a constant at the top of it

    Returns
    -------
    decisions : dict
        File name -> 'good' or 'bad'. Empty if nothing has been decided yet.

    """
    decisionFile = fPath + '0_ProcessingDecisions_' + scriptName + '.csv'
    if os.path.exists(decisionFile) == False:
        return {}

    dat = pd.read_csv(decisionFile)

    # The latest row for a file wins. Sort on the parsed timestamp, not on the
    # raw text: opening this file in Excel to correct a decision rewrites every
    # timestamp into the local format, and sorting those as text puts 9:52
    # after 10:15 because the hour has no leading zero. A row whose timestamp
    # will not parse at all is taken as the newest, since that is almost always
    # someone having typed a correction in by hand.
    stamps = pd.to_datetime(dat['Timestamp'], errors='coerce', format='mixed')
    dat['SortStamp'] = stamps
    dat_sorted = dat.sort_values('SortStamp', kind='stable', na_position='last')
    latest = dat_sorted.drop_duplicates(subset='FileName', keep='last')

    # Preallocate
    decisions = {}
    for ii in range(len(latest)):
        fName = latest['FileName'].iloc[ii]
        decisions[fName] = latest['Decision'].iloc[ii]

    return decisions

def saveProcessingDecision(fPath, scriptName, fName, decision, Subject='', Config='', reason=''):
    """
    Append one data-quality decision to this script's decision file.

    Call this inside the processing loop, immediately after the decision is
    made, so that a run which fails part way through still keeps every decision
    taken up to that point.

    Note: this appends on purpose, and is the one place in these scripts where
    appending is correct. The file is a decision history that must never be
    overwritten. Data outputs are a different matter - those are written fresh
    each run, because appending to them produces duplicate rows.

    Parameters
    ----------
    fPath : str
        Path to the folder holding the data
    scriptName : str
        Short name of the calling script, matching loadProcessingDecisions
    fName : str
        Name of the file the decision applies to. For IMU trials this is the
        low-g file, since that is what the processing loops index through.
    decision : str
        'good' or 'bad'
    Subject : str
        Subject name, for reading the file by eye
    Config : str
        Configuration, for reading the file by eye
    reason : str
        Free text. Worth filling in when a trial is rejected.

    Returns
    -------
    None

    """
    decisionFile = fPath + '0_ProcessingDecisions_' + scriptName + '.csv'
    stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    row = pd.DataFrame({'Timestamp':[stamp], 'FileName':[fName],
                        'Subject':[Subject], 'Config':[Config],
                        'Decision':[decision], 'Reason':[reason]})

    # Only write the header when the file is being created
    writeHeader = os.path.exists(decisionFile) == False
    row.to_csv(decisionFile, mode='a', header=writeHeader, index=False)
