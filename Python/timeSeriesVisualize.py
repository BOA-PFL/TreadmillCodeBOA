# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 09:45:18 2020
Visualize shadded errorbars
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
    
## Load file in
fName = entries[2] #Load one file at a time
        
dat = pd.read_csv(fPath+fName,sep='\t', skiprows = 8, header = 0)

#Calcualte ankle and tibial forces
ankleForce = dat.LeftAnkleForce * -1
ankleForce[ankleForce<fThresh] = 0

dat['TibialForce'] = ankleForce + (dat.LAnkleMomenty / 0.05)
dat['PFForce'] = (dat.LAnkleMomenty / 0.05)

#find the landings and offs of the FP as vectors from function above
landings = findLandings(ankleForce)
takeoffs = findTakeoffs(ankleForce)

plt.plot(ankleForce)

# Find landings, create np matrix for plotting
stanceTime = []
for landing in landings:
    try:
       # Define where next zero is
        def condition(x): return x <= 0 
        zeros = [idx for idx, element in enumerate(np.array(ankleForce[landing:landing+100])) if condition(element)]
        stanceTime.append(zeros[1])
    except:
        print(landing)
        
# index into which landings are false
stanceTime[stanceTime < 40]
[x for x in stanceTime if x < 40]

# preallocate matrix for force and fill in with force data
preForce = np.zeros((3,80))
preTib = np.zeros((3,80))
iterVar = 0
for landing in landings:
    preForce[iterVar,] = ankleForce[landing:landing+80]
    preTib[iterVar,] = dat.TibialForce[landing:landing+80]
    iterVar = iterVar +1

# create matrices with average and SD of force trajectories
x = np.linspace(0,80,80)
avgF = np.mean(preForce, axis = 0)
sdF = np.std(preForce, axis = 0)
#Plot force
plt.plot(x, avgF, 'k', color='#CC4F1B')
plt.fill_between(x, avgF-sdF, avgF+sdF,
    alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
    
#Average Tibial force
avgTib = np.mean(preTib,axis=0)
sdTib = np.std(preTib, axis = 0)
#Plot average Tibial force 
plt.plot(x, avgTib, 'k', color='#CC4F1B')
plt.fill_between(x, avgTib-sdTib, avgTib+sdTib,
    alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
    