# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 16:18:53 2021

@author: Bethany.Kilpatrick
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Note on plotting
# If plotting results in an error use the command:
# %matplotlib qt
# You can also change the preferences in Spyder (Tools>Preferences>IPython console
# Select "Graphics" tab, and for "Graphics backen" select "Automatic")

meter = 0 ### Specify which power meter you used. WattBike == 0, Garmin == 1

# Ensure that the file path contains YOUR NAME (e.g. bethany.kilpatrick)
fPath = 'C:/Users/eric.honert/Boa Technology Inc/PFL Team - General/Cycling Performance Tests/Cycling_4guideSD_Feb2022/Wattbike Data/'
fileExt = r".csv"
entries = [fName for fName in os.listdir(fPath) if fName.endswith(fileExt)]

#### Wattbike data

print('Open all wattbike/power files recorded for the subject')
# Initiate storing variables
config = []
subName = []
trial = []
steadyPower = []
steadyCadence = []
steadySym = []

sprintMax = []
cadenceMax = []
sprintSym = []


# Index through the files selected
for fName in entries:
    # If the loop is not able accomplished, "try" will skip the file
    try:   
        
      dat = pd.read_csv(fPath+fName, header = 0)
    # This is for renaming columns that come from the Watt Bike Monitor. You may also need
    # to adjust this for units in the names as well
    
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
        sprintPower = [] 
        sprintCadence = []
        
        (sps, y) = sprintStart[0]
        sps = round(sps)
        for p in range(0,11):
            # Create a moving average
            sprintPower.append(np.mean(dat.power[sps + p: sps + p + 5]))
            sprintCadence.append(np.mean(dat.cadence[sps + p:sps + p + 5]))
            
            if meter == 0:
                sprintSym.append(np.mean(dat.balance[sps + p:sps + p + 5]))
        
        # Find the maximum from the moving averaged sprint data
        sprintMax.append(max(sprintPower))
        Max_idx = sprintPower.index(sprintMax[-1])
        cadenceMax.append(sprintCadence[Max_idx])

        # if meter == 0:
        #  symmetryMax = sprintSym[Max_idx]

      subName.append(fName.split('_')[0])
      config.append(fName.split('_')[1])
      trial.append(fName.split('_')[2].split('.')[0])

    
    except:
        print(fName)



# Combine outcomes and export to csv
outcomes = pd.DataFrame({'Subject':list(subName), 'Config': list(config), 'Trial':list(trial), 'Power_steady':steadyPower, 'Cadence_steady': steadyCadence, 
                            'Power_sprint':sprintMax, 'Cadence_sprint':cadenceMax})  

outfileName = fPath + 'CompiledPowerData.csv'
outcomes.to_csv(outfileName, index = False)

