# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 16:39:46 2021

@author: Kate.Harrison
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askopenfilenames
import os

meter = 1 ### Specify which power meter you used. WattBike == 0, Garmin == 1


print('open file containing metabolic data for the subject')
filename = askopenfilename()

sub = filename.split(sep = '/')[2].split(sep = "_")[0]
filenameNoExt = filename.split(sep = ".")[0]
dat = pd.read_excel(filename, sheet_name = 'Data', usecols = 'O:BQ')

dat = dat[2:-1]



info = pd.read_excel(filename, sheet_name = 'Data', usecols = 'B:B')
wt = info['Unnamed: 1'].values[5]

# Compute the energy expendature from the oxygen and carbon dioxide measures
ee = (dat.VO2*16.58/60 + dat.VCO2*4.15/60)*0.85984522785899

# Create time-continuous energy expendature figure to select regions of interest
plt.figure()
plt.plot(ee)
steadyStart = plt.ginput(6)
plt.close

trialNo = list(range(1,7))
# Initiate storing variables
vo2mean = []
hrmean = []
eemean = []

# Index through the steady state selections
for i in range(len(steadyStart)):
    

    (ss, y)  = steadyStart[i]
    ss = round(ss)
    se = ss + 40
    vo2mean.append(np.mean(dat.VO2[ss:se])/wt)
    
    hrmean.append(np.mean(dat.HR[ss:se]))
    eemean.append(np.mean(ee[ss:se]))
    

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
vo2max_Low = (((Power_steady/0.85)+5)*10.8/wt)+7
vo2max_High = (((Power_steady/0.8)+15)*10.8/wt)+7

if meter == 0:
    data = pd.DataFrame({'vo2':np.mean(vo2mean), 'vo2max_Low':vo2max_Low, 'vo2max_High':vo2max_High,'HR':np.mean(hrmean), 'EnergyExpenditure':np.mean(eemean), 
                           'Power_steady':np.mean(steadyPower), 'Cadence_steady': np.mean(steadyCadence), 
                           'Symmetry_steady': np.mean(steadySym), 'Power_sprint':sprintMax, 
                           'Cadence_sprint':cadenceMax, 'Symmetry_sprint':symmetryMax}, index = [0])

if meter == 1:
    data = pd.DataFrame({'vo2':np.mean(vo2mean), 'vo2max_Low':vo2max_Low, 'vo2max_High':vo2max_High,'HR':np.mean(hrmean), 'EnergyExpenditure':np.mean(eemean), 
                           'Power_steady':np.mean(steadyPower), 'Cadence_steady': np.mean(steadyCadence), 
                           'Power_sprint':sprintMax, 
                           'Cadence_sprint':cadenceMax}, index = [0])
    
# outFileName = filenameNoExt + '_SubjectData.csv'

# data.to_csv(outFileName, index = False)

print('Copy info from *data* DataFrame into athlete feedback form')