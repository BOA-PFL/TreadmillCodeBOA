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
import tkinter as tk
from tkinter import messagebox 
import addcopyfighandler
from TreadmillFunctions import loadProcessingDecisions, saveProcessingDecision

# Note on plotting
# If plotting results in an error use the command:
# %matplotlib qt
# You can also change the preferences in Spyder (Tools>Preferences>IPython console
# Select "Graphics" tab, and for "Graphics backen" select "Automatic")

save_on = 0
# 0 = no data check, nothing is asked or recorded
# 1 = ask only about trials with no recorded decision (the normal setting)
# 2 = re-review EVERY trial, including ones previously rejected. Use after
#     changing processing, when earlier decisions may no longer hold. This
#     mode is good for reruning files with a different threshold or examining
#     files after a different filter has been applied.
data_check = 1
ss_dur = 45 # This is the steady state duration in seconds
sprint_dur = 10 # This is the sprint duration in seconds

# Ensure that the file path contains YOUR NAME (e.g. bethany.kilpatrick)
fPath = 'Z:/Testing Segments/Cycling/2025/2025_Performance_CyclingLacevBOA_Specialized/Wahoo/SteadySprintTrials/'
fileExt = r".fit"
entries = [fName for fName in os.listdir(fPath) if fName.endswith(fileExt) and not fName.startswith('0_')]

# Names this script's decision file, 0_ProcessingDecisions_<scriptName>.csv.
# One data folder is often read by several scripts, and a trial that is no
# good for one analysis can be fine for another, so each keeps its own
# record rather than one script's rejection governing the rest.
scriptName = 'CyclingWahoo'

# Data-quality decisions recorded by previous runs of this script. Loading
# these is what stops a trial being judged twice. Rejected files are left
# where they are so they can still be opened and looked at - it is this
# record, not the folder, that decides what gets processed.
decisions = loadProcessingDecisions(fPath, scriptName)

# Results from an out-of-date copy of this script will not match anyone
# else's. Asked only when nothing has been decided here yet: a decision file
# means the question was answered on an earlier run of this dataset.
if len(decisions) == 0:
    # Give this dialog its own window and put it in front. It is the first
    # thing the script does, so unlike the data check further down there is no
    # figure on screen to pull focus, and a borrowed default root can open the
    # box behind Spyder - which just looks like the script has hung.
    dialogRoot = tk.Tk()
    dialogRoot.withdraw()
    dialogRoot.attributes('-topmost', True)
    upToDate = messagebox.askyesno("Question", "Is this script up to date with main?",
                                   parent = dialogRoot)
    dialogRoot.destroy()
    # A plain error rather than SystemExit: these scripts get run a cell at a
    # time, and SystemExit can take the console with it. This stops just as
    # firmly, prints the reason, and leaves the console alone.
    if upToDate == False:
        raise RuntimeError('Pull this script from main before processing')

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
      # A trial rejected in a previous run is skipped. The file is still there
      # to open and look at: the recorded decision governs, not the folder
      priorDecision = decisions.get(fName, '')
      if priorDecision == 'bad' and data_check != 2:
          print('Previously excluded, skipping: ' + fName)
          continue

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
      

      #________________________________________________________________________
      # Data check. This script had none, so the plot shows what there is to
      # judge: the power trace with the clicked region starts on it, and the
      # window each steady-state metric is taken over.
      tmpSub = fName.split('_')[0]
      tmpConfig = fName.split('_')[1]

      answer = True # if data check is off.
      if priorDecision == 'good' and data_check == 1:
          print('Previously accepted, not asking again: ' + fName)
      elif data_check >= 1:
          plt.figure()
          plt.plot(dat.power, color = 'gray')
          for (ss, y) in steadyStart:
              ss = round(ss)
              plt.axvspan(ss, ss + ss_dur, color = 'tab:blue', alpha = 0.3)
          for (sps, y) in sprintStart:
              plt.axvspan(round(sps), round(sps) + sprint_dur, color = 'tab:red', alpha = 0.3)
          plt.ylabel('Power [Watt]')
          plt.xlabel('Time [sec]')
          plt.title(fName + '  -  blue = steady state, red = sprint')

          answer = messagebox.askyesno("Question","Is data clean?")
          plt.close('all')

      # Record the decision straight away, inside the loop, so a run that
      # fails later keeps every decision made so far
      if priorDecision == '' or data_check == 2:
          if answer == False:
              saveProcessingDecision(fPath, scriptName, fName, 'bad',
                                     Subject=tmpSub, Config=tmpConfig,
                                     reason='rejected at the data check')
          else:
              saveProcessingDecision(fPath, scriptName, fName, 'good',
                                     Subject=tmpSub, Config=tmpConfig)

      if answer == False:
          print('Adding file to bad file list')
          continue

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
                
        # sprintStart[j], not [0]: every pass was re-measuring the first
        # sprint while labelling the rows as separate sprints
        (sps, y) = sprintStart[j]
        sps = round(sps)
        for p in range(0,sprint_dur+1):
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
if save_on == 1 and data_check < 2:
    # Note: written fresh each run, never appended. Appending duplicates
    # every row when a study is re-processed after more data is collected.
    outcomes.to_csv(outfileName, header=True, index = False)
