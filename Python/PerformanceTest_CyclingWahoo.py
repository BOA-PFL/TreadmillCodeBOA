# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 16:51:22 2025

@author: Eric.Honert
"""

from fitparse import FitFile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tkinter import messagebox 
import addcopyfighandler

# Note on plotting
# If plotting results in an error use the command:
# %matplotlib qt
# You can also change the preferences in Spyder (Tools>Preferences>IPython console
# Select "Graphics" tab, and for "Graphics backen" select "Automatic")

save_on = 1
ss_dur = 45 # This is the steady state duration in seconds

# Ensure that the file path contains YOUR NAME (e.g. bethany.kilpatrick)
fPath = 'C:\\Users\\eric.honert\\Boa Technology Inc\\PFL Team - General\\Testing Segments\\Cycling Performance Tests\\2025_Performance_CyclingLacevBOA_Specialized\\Wahoo\\SteadySprintTrials\\'
fileExt = r".fit"
entries = [fName for fName in os.listdir(fPath) if fName.endswith(fileExt)]

# Initiate storing variables
config = []
subName = []
Order = []
Movement = []
avgPower = []
maxPower = []
avgCadence = []
maxCadence = []

# Index through the files selected
for fName in entries:
    # If the loop is not able accomplished, "try" will skip the file   
      #__________________________________________________________________________
      # Extract the fitfile information into dataframe (df)
      print(fName)
      fitfile = FitFile(fPath+fName)
      while True:
          try:
              fitfile.messages
              break
          except KeyError:
              continue
      workout = []
      for record in fitfile.get_messages('record'):
          r = {}
          for record_data in record:
              r[record_data.name] = record_data.value
          workout.append(r)
      dat = pd.DataFrame(workout)
      #________________________________________________________________________
      # Create time-continuous power figure to select regions of interest 
      saveFolder = fPath + 'PowerPlots'
       
      if os.path.exists(saveFolder) == False:
          os.mkdir(saveFolder) 
                 
      if os.path.exists(fPath+fName+'TrialSeg.npy'):
          trial_segment_old = np.load(fPath+fName+'TrialSeg.npy', allow_pickle =True)
          steadyStart = trial_segment_old[0]
          sprintStart = trial_segment_old[1]
          
      else: 
          plt.figure()
          plt.plot(dat.power)
          plt.ylabel('Power [Watt]')
          plt.xlabel('Time [sec]')
          plt.title(fName.split('.csv')[0])
          plt.savefig(saveFolder + '/' + fName.split('.csv')[0] +'.png')
          
          print('click the start of as many steady state periods are recorded in the file. Press enter when done')
          steadyStart = plt.ginput(-1, timeout = 60)
        
          print('click the start of as many sprints are recorded in the file. Press enter when done')
          sprintStart = plt.ginput(-1, timeout = 60)
          plt.close()
          
          pts = [steadyStart,sprintStart]
          trial_segment = np.array(pts, dtype = object)
          np.save(fPath+fName+'TrialSeg.npy',trial_segment)
      

      # Index through the steady-state regions and extract metrics of interest
      for k in range(len(steadyStart)):
        
        (ss, y) = steadyStart[k]
        ss = round(ss)
        avgPower.append(np.mean(dat.power[ss:ss + ss_dur]))
        avgCadence.append(np.mean(dat.cadence[ss:ss + ss_dur]))
        maxPower.append(np.max(dat.power[ss:ss + ss_dur]))
        maxCadence.append(np.max(dat.cadence[ss:ss + ss_dur]))
        Movement.append('SteadyState')
        subName.append(fName.split('_')[0])
        config.append(fName.split('_')[1])
        Order.append(fName.split('_')[2].split('.')[0])
    
      # Index through the steady-state regions and extract metrics of interest
      for j in range(len(sprintStart)):
        sprintPower = [] 
        sprintCadence = []
                
        (sps, y) = sprintStart[0]
        sps = round(sps)
        for p in range(0,11):
            # Create a moving average over the 10 sec sprint
            sprintPower.append(np.mean(dat.power[sps + p: sps + p + 5]))
            sprintCadence.append(np.mean(dat.cadence[sps + p:sps + p + 5]))
        
        # Find the maximum from the moving averaged sprint data
        maxPower.append(max(sprintPower))
        Max_idx = sprintPower.index(maxPower[-1])
        maxCadence.append(sprintCadence[Max_idx])
        avgPower.append(np.mean(sprintPower))
        avgCadence.append(np.mean(sprintCadence))
        
        Movement.append('Sprint')
        subName.append(fName.split('_')[0])
        config.append(fName.split('_')[1])
        Order.append(fName.split('_')[2].split('.')[0])



# Combine outcomes and export to csv
outcomes = pd.DataFrame({'Subject':list(subName), 'Config': list(config), 'Order':list(Order), 'Movement':list(Movement),
                         'avgPower':list(avgPower),'avgCadence':list(avgCadence),
                         'peakPower':list(maxPower), 'peakCadence':list(maxCadence)})  

outfileName = fPath + '0_CompiledPowerData.csv'
if save_on == 1:
    if os.path.exists(outfileName) == False:
        outcomes.to_csv(outfileName, header=True, index = False)
    else:
        outcomes.to_csv(outfileName, mode='a', header=False, index = False)