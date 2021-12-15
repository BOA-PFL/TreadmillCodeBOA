# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 16:18:53 2021

@author: Bethany.Kilpatrick
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

meter = 0 ### Specify which power meter you used. WattBike == 0, Garmin == 1

fPath = 'C:/Users/bethany.kilpatrick/Boa Technology Inc/PFL - General/Cycling2021/DH_Cycling_Nov2021/WattBike/'
fileExt = r".csv"
entries = [fName for fName in os.listdir(fPath) if fName.endswith(fileExt)]

#### Wattbike data

print('Open all wattbike/power files recorded for teh subject')
config = []
subName = []
steadyPower = []
steadyCadence = []
steadySym = []

sprintMax = []
cadenceMax = []
sprintSym = []



for fName in entries:
    try:

     
     #for i in range(len(entries)):
    
      #fName = entries[i]
    
      config1 = fName.split('_')[1]
        
        
      dat = pd.read_csv(fPath+fName, header = 0)
    # This is for renaming columns that come from the Watt Bike Monitor. You may also need
    # to adjust this for units in the names as well
    
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
        sprintPower = [] 
        sprintCadence = []
        
        (sps, y) = sprintStart[0]
        sps = round(sps)
        for p in range(0,11):
            
            sprintPower.append(np.mean(dat.power[sps + p: sps + p + 5]))
            sprintCadence.append(np.mean(dat.cadence[sps + p:sps + p + 5]))
            
            if meter == 0:
                sprintSym.append(np.mean(dat.balance[sps + p:sps + p + 5]))
            
        sprintMax.append(max(sprintPower))
        Max_idx = sprintPower.index(sprintMax[-1])
        cadenceMax.append(sprintCadence[Max_idx])

        # if meter == 0:
        #  symmetryMax = sprintSym[Max_idx]

        
        # powerSprint = np.max(sprintPower) 
        # cadenceMax = np.max(sprintCadence) 
        
        #Power_steady = np.mean(steadyPower) 
      subName.append(fName.split('_')[0])
      config.append( config1 )

    
    except:
        print(fName)



    
outcomes = pd.DataFrame({'Subject':list(subName), 'Config': list(config),'Power_steady':steadyPower, 'Cadence_steady': steadyCadence, 
                            'Power_sprint':sprintMax, 'Cadence_sprint':cadenceMax})  

outfileName = fPath + 'CompiledPowerData.csv'
outcomes.to_csv(outfileName, index = False)

