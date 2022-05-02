# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 11:38:57 2020
Calculates point estimates from treadmill kinetic data
@author: Daniel.Feeney
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.signal as sig
import seaborn as sns

# Define constants and options
fThresh = 80; #below this value will be set to 0.
writeData = 0; #will write to spreadsheet if 1 entered

# Read in balance file
fPath = 'C:/Users/Daniel.Feeney/Dropbox (Boa)/EnduranceProtocolWork/TibiaForceData/'
entries = os.listdir(fPath)

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
    


#Preallocation
tibForcePk = []
tibImpulse = []
sName = []
tmpConfig = []
timeP = []
# need to add force vector to this because the ankle force 
## loop through the selected files
for file in entries:
    try:
        
        fName = file #Load one file at a time
        
        dat = pd.read_csv(fPath+fName,sep='\t', skiprows = 8, header = 0)
        #Parse file name into subject and configuration 
        subName = fName.split(sep = "_")[0]
        config = fName.split(sep = "_")[2]
        timePoint = fName.split(sep = "_")[3]
        
        # Filter force
        ankleForce = dat.LeftAnkleForce * -1
        ankleForce[ankleForce<fThresh] = 0

        # Compute Tibia Force and Plantar Flexion Force assuming the Achilles
        # Tendon moment arm is 5 cm
        dat['TibialForce'] = ankleForce + (dat.LAnkleMomenty / 0.05)
        dat['PFForce'] = (dat.LAnkleMomenty / 0.05)
        #find the landings and offs of the FP as vectors
        landings = findLandings(ankleForce)
        takeoffs = findTakeoffs(ankleForce)

        #For each landing, calculate rolling averages
        for landing in landings:
            try:
               # Define where next zero is
                def condition(x): return x <= 0 
                zeros = [idx for idx, element in enumerate(np.array(ankleForce[landing:landing+100])) if condition(element)]
                nextZero = zeros[1]
                
                tibForcePk.append(dat.TibialForce[landing:landing+nextZero].max())
                tibImpulse.append(dat.TibialForce[landing:landing+nextZero].sum())
                sName.append(subName)
                tmpConfig.append(config)
                timeP.append(timePoint)
            except:
                print(landing)
        
    except:
            print(file)
            
# Place outcomes in a single variable
outcomes = pd.DataFrame({'Sub':list(sName), 'Config': list(tmpConfig), 'TimePoint':list(timeP),
                         'PkTibForce':list(tibForcePk), 'TibImpulse':list(tibImpulse)})
cleanedOutcomes = outcomes[outcomes['PkTibForce'] >= 2900]


#plt.plot(dat.LAnkleMomenty, label = 'Ankle Moment')
#plt.plot(ankleForce, label = 'Ankle Force')
#plt.legend()
#
#plt.plot(dat.TibialForce, label = 'Tibial Force')
#plt.plot(ankleForce, label = 'Ankle Force')
#plt.plot(dat.PFForce, label = 'Plantarflexor Force')
#plt.legend()
#    
#f, axes = plt.subplots(1,4)
#sns.boxplot(y='peakBrake', x='Sub', hue="Config",
#                 data=outcomes, 
#                 palette="colorblind", ax=axes[0])
#
#sns.boxplot(y='VLR', x='Sub', hue = "Config", 
#                 data=outcomes, 
#                 palette="colorblind", ax=axes[1])
#
#sns.boxplot(y='brakeImpulse', x='Sub', hue = "Config", 
#                 data=cleanedOutcomes, 
#                 palette="colorblind", ax=axes[2])
#
#sns.boxplot(y='NL', x='Sub', hue = "Config", 
#                 data=cleanedOutcomes, 
#                 palette="colorblind", ax=axes[3])
#plt.tight_layout()
