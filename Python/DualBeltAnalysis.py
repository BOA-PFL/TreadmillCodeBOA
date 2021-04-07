# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 11:38:57 2020

@author: Daniel.Feeney
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.signal as sig
import seaborn as sns

# Define constants and options
fThresh = 50; #below this value will be set to 0.
writeData = 0; #will write to spreadsheet if 1 entered

# Read in balance file
fPath = 'C:/Users/Daniel.Feeney/Dropbox (Boa)/EnduranceProtocolWork/WalkData/Forces/'
fPath = 'C:/Users/Daniel.Feeney/Dropbox (Boa)/EnduranceProtocolWork/EnduranceProtocolHike/TMForces/'
entries = os.listdir(fPath)

# list of functions 
# finding landings on the force plate once the filtered force exceeds the force threshold
def findLandings(force):
    lic = []
    for step in range(len(force)-1):
        if force[step] == 0 and force[step + 1] >= fThresh:
            lic.append(step)
    return lic

#Find takeoff from FP when force goes from above thresh to 0
def findTakeoffs(force):
    lto = []
    for step in range(len(force)-1):
        if force[step] >= fThresh and force[step + 1] == 0:
            lto.append(step + 1)
    return lto


def calcVLR(force, startVal, lengthFwd):
    # function to calculate VLR from 80 and 20% of the max value observed in the first n
    # indices (n defined by lengthFwd). 
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
    
#Find max braking force moving forward
def calcPeakBrake(force, landing, length):
    newForce = np.array(force)
    return min(newForce[landing:landing+length])

def findNextZero(force, length):
    # Starting at a landing, look forward (after first 15 indices)
    # to the find the next time the signal goes from - to +
    for step in range(length):
        if force[step] <= 0 and force[step + 1] >= 0 and step > 15:
            break
    return step


#Preallocation
loadingRate = []
peakBrakeF = []
brakeImpulse = []
VLR = []
VLRtwo = []
sName = []
tmpConfig = []
timeP = []
NL = []
PkMed = []
PkLat = []

# Save Time series data in separate DF #
vertForce = []
longConfig = []
longSub = []
timeIndex = []

## loop through the selected files
for file in entries:
    try:
        
        fName = file #Load one file at a time
        
        dat = pd.read_csv(fPath+fName,sep='\t', skiprows = 8, header = 0)
        #Parse file name into subject and configuration 
        subName = fName.split(sep = "_")[0]
        config = fName.split(sep = "_")[2]
        config = config.split(sep = ' - ')[0]
        #timePoint = fName.split(sep = "_")[3]
        dat['LForceY'] = dat['LForceY'].fillna(0) #removing the often NA first 3-10 entries
        
        # Filter force
        forceZ = dat.LForceZ * -1
        forceZ[forceZ<fThresh] = 0
        brakeForce = dat.LForceY[0:59999] * -1
        MForce = dat.LForceX[0:59999] 
        
        
        fs = 1000 #Sampling rate
        t = np.arange(59999) / fs
        fc = 20  # Cut-off frequency of the filter
        w = fc / (fs / 2) # Normalize the frequency
        b, a = sig.butter(4, w, 'low')
        brakeForce[0] = 0
        brakeFilt = sig.filtfilt(b, a, brakeForce)
        
        #find the landings and offs of the FP as vectors
        landings = findLandings(forceZ)
        takeoffs = findTakeoffs(forceZ)
                
        
        #For each landing, calculate rolling averages and time to stabilize
    
        for landing in landings:
            try:
               # Define where next zero is
                nextLanding = findNextZero(brakeFilt[landing:landing+600],600)
                NL.append(nextLanding)
                #stepLen.append(findStepLen(forceZ[landing:landing+800],800))
                brakeImpulse.append(sum(brakeFilt[landing+10:landing+nextLanding]))
                sName.append(subName)
                tmpConfig.append(config)
                #timeP.append(timePoint)
                peakBrakeF.append(calcPeakBrake(brakeFilt,landing, 600))
                VLR.append(calcVLR(forceZ, landing, 200))
                PkMed.append(np.max(MForce[landing:landing+600]))
                PkLat.append(np.min(MForce[landing:landing+600]))
                
                ## Time series Saving ##
                timeIndex.append(np.arange(0,800))
                vertForce.append(forceZ[landing:landing+800])
                longConfig.append(np.repeat(config, 800))
                longSub.append(np.repeat(subName,800))
            except:
                print(landing)
        
    except:
            print(file)

outcomes = pd.DataFrame({'Subject':list(sName), 'Config': list(tmpConfig),'NL':list(NL),'peakBrake': list(peakBrakeF),
                         'brakeImpulse': list(brakeImpulse), 'VLR': list(VLR), 'PkMed':list(PkMed), 'PkLat':list(PkLat)})

#df2 = pd.DataFrame(pd.concat(vertForce), np.concatenate(longConfig))    

#longDat = pd.concat(vertForce, ignore_index = True)
#longDat2['Config'] = pd.DataFrame(np.concatenate(longConfig))
#longDat['Sub'] = pd.DataFrame(np.concatenate(longSub))
#longDat['TimePoint'] = pd.DataFrame(np.concatenate(timeIndex))

outcomes[['peakBrake']] = -1 * outcomes[['peakBrake']]
outcomes[['PkLat']] = -1 * outcomes[['PkLat']]
cleanedOutcomes = outcomes[outcomes['brakeImpulse'] <= -1000]
cleanedOutcomes[['brakeImpulse']] = -1 * cleanedOutcomes[['brakeImpulse']]
    

f, axes = plt.subplots(1,2)
sns.boxplot(y='peakBrake', x='Subject', hue="Config",
                 data=cleanedOutcomes, 
                 palette="colorblind", ax=axes[0])

sns.boxplot(y='VLR', x='Subject', hue = "Config", 
                 data=cleanedOutcomes, 
                 palette="colorblind", ax=axes[1])

f, axes = plt.subplots(1,2)
sns.boxplot(y='brakeImpulse', x='Subject', hue = "Config", 
                 data=cleanedOutcomes, 
                 palette="colorblind", ax=axes[0])

sns.boxplot(y='NL', x='Subject', hue = "Config", 
                 data=cleanedOutcomes, 
                 palette="colorblind", ax=axes[1])
plt.tight_layout()

f, axes = plt.subplots(1,2)
sns.boxplot(y='PkMed', x='Subject', hue="Config",
                 data=cleanedOutcomes, 
                 palette="colorblind", ax=axes[0])
sns.boxplot(y='PkLat', x='Subject', hue="Config",
                 data=cleanedOutcomes, 
                 palette="colorblind", ax=axes[1])

#
#newForce = pd.concat(vertForce)
#newSub = pd.concat(longSub)
#newConfig = pd.concat(longConfig)
#newTry = {'vertForce': newForce,
#        'Sub': newSub,
#        'Config': newConfig
#        }
#
#df = pd.DataFrame(newTry, columns = ['Force', 'Sub','Config'])
#### 
#fmri = sns.load_dataset("fmri")
#sns.relplot(
#    data=fmri, kind="line",
#    x="timepoint", y="signal", col="region",
#    hue="event", style="event",
#)

###