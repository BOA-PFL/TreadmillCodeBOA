# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 10:48:20 2021

@author: Bethany.Kilpatrick
"""""" 

This code was from the Cycling Athlete feedback code, 
but amended for Down Hill Cycling Tests. Does not have HR, EE, or Vo2Max

Created on Fri Aug 13 16:39:46 2021

@author: Kate.Harrison
"""""""""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilenames
import os

meter = 0 ### Specify which power meter you used. WattBike == 0, Garmin == 1


#### Wattbike data

print('Open all wattbike/power files recorded for the subject')
filenames = askopenfilenames()

# Initiate storing variables
steadyPower = []
steadyCadence = []
steadySym = []

sprintPower = []
sprintCadence = []
sprintSym = []

# Index through the files selected
for entry in filenames:
    
    dat = pd.read_csv(entry)
    
    # Just in case the headers for the dataframe need to be replaced
    # dat = dat.rename(columns={'Cadence ':'Cadence', 'Speed ':'Speed',
    #    'Distance ':'Distance','Force ':'Force','Power ': 'Power'})

    # Create time-continuous power figure to select regions of interest
    plt.figure()
    plt.plot(dat.power)
    
    print('click the start of as many steady state periods are recorded in the file. Press enter when done')
    steadyStart = plt.ginput(-1)
    
    print('click the start of as many sprints are recorded in the file. Press enter when done')
    sprintStart = plt.ginput(-1) 
    plt.close
    
    # Index through the steady-state regions and extract metrics of interest
    for k in range(len(steadyStart)):
        
        (ss, y) = steadyStart[k]
        ss = round(ss)
        steadyPower.append(np.mean(dat.power[ss:ss + 300]))
        steadyCadence.append(np.mean(dat.cadence[ss:ss + 300]))
        
        if meter == 0:
            steadySym.append(np.mean(dat.balance[ss:ss + 300]))
    
    # Index through the steady-state regions and extract metrics of interest
    for j in range(len(sprintStart)):
        
        (sps, y) = sprintStart[0]
        sps = round(sps)
        for p in range(0,11):
            # Moving average for sprinting values to account for noise
            sprintPower.append(np.mean(dat.power[sps + p: sps + p + 5]))
            sprintCadence.append(np.mean(dat.cadence[sps + p:sps + p + 5]))
            
            if meter == 0:
                sprintSym.append(np.mean(dat.balance[sps + p:sps + p + 5]))
            
# Store sprinting metrics of interest
sprintMax = max(sprintPower)
Max_idx = sprintPower.index(sprintMax)
cadenceMax = sprintCadence[Max_idx]

if meter == 0:
    symmetryMax = sprintSym[Max_idx]

Power_steady = np.mean(steadyPower)

# Combine all data into a dataframe to be copied into an athlete report
if meter == 0:
    data = pd.DataFrame({'Power_steady':np.mean(steadyPower), 'Cadence_steady': np.mean(steadyCadence), 
                           'Symmetry_steady': np.mean(steadySym), 'Power_sprint':sprintMax, 
                           'Cadence_sprint':cadenceMax, 'Symmetry_sprint':symmetryMax}, index = [0])


if meter == 1:
    data = pd.DataFrame({'Power_steady':np.mean(steadyPower), 'Cadence_steady': np.mean(steadyCadence), 
                           'Power_sprint':sprintMax, 
                           'Cadence_sprint':cadenceMax}, index = [0])


print('Copy info from *data* DataFrame into athlete feedback form')

