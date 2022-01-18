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

print('Open all wattbike/power files recorded for teh subject')
filename = askopenfilenames()

steadyPower = []
steadyCadence = []
steadySym = []

sprintPower = []
sprintCadence = []
sprintSym = []

for i in range(len(filename)):
    
    file = filename[i]
    dat = pd.read_csv(file)
    
    '''
    This is for renaming columns that come from the Watt Bike Monitor. You may also need
    to adjust this for units in the names as well
    '''
    # dat = dat.rename(columns={'Cadence ':'Cadence', 'Speed ':'Speed',
    #    'Distance ':'Distance','Force ':'Force','Power ': 'Power'})

   
    plt.figure()
    plt.plot(dat.power)
    
    print('click the start of as many steady state periods are recorded in the file. Press enter when done')
    steadyStart = plt.ginput(-1)
    
    print('click the start of as many sprints are recorded in the file. Press enter when done')
    sprintStart = plt.ginput(-1) 
    plt.close
    
    for k in range(len(steadyStart)):
        
        (ss, y) = steadyStart[k]
        ss = round(ss)
        steadyPower.append(np.mean(dat.power[ss:ss + 300]))
        steadyCadence.append(np.mean(dat.cadence[ss:ss + 300]))
        
        if meter == 0:
            steadySym.append(np.mean(dat.balance[ss:ss + 300]))
    
    
    for j in range(len(sprintStart)):
        
        (sps, y) = sprintStart[0]
        sps = round(sps)
        for p in range(0,11):
            
            sprintPower.append(np.mean(dat.power[sps + p: sps + p + 5]))
            sprintCadence.append(np.mean(dat.cadence[sps + p:sps + p + 5]))
            
            if meter == 0:
                sprintSym.append(np.mean(dat.balance[sps + p:sps + p + 5]))
            
sprintMax = max(sprintPower)
Max_idx = sprintPower.index(sprintMax)
cadenceMax = sprintCadence[Max_idx]

if meter == 0:
    symmetryMax = sprintSym[Max_idx]

Power_steady = np.mean(steadyPower)


if meter == 0:
    data = pd.DataFrame({'Power_steady':np.mean(steadyPower), 'Cadence_steady': np.mean(steadyCadence), 
                           'Symmetry_steady': np.mean(steadySym), 'Power_sprint':sprintMax, 
                           'Cadence_sprint':cadenceMax, 'Symmetry_sprint':symmetryMax}, index = [0])


if meter == 1:
    data = pd.DataFrame({'Power_steady':np.mean(steadyPower), 'Cadence_steady': np.mean(steadyCadence), 
                           'Power_sprint':sprintMax, 
                           'Cadence_sprint':cadenceMax}, index = [0])
    
    
    
# outFileName = filenameNoExt + '_SubjectData.csv'

# data.to_csv(outFileName, index = False)

print('Copy info from *data* DataFrame into athlete feedback form')

