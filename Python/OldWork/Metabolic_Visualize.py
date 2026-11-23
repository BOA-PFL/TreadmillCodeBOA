# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 14:17:01 2021
Analyze metabolic data with given inputs
@author: Daniel.Feeney
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def calcMean2MinEpoch(fullDat, inputMarker):
    startTime = fullDat[fullDat['Marker'] == inputMarker].index.tolist()
    td1 = fullDat['t'][startTime] + pd.to_timedelta('0 days 00:03:00')
    td2 = fullDat['t'][startTime] + pd.to_timedelta('0 days 00:04:30')
    tp1 = fullDat[(dat['t'] > str(td1)[5:20]) & (fullDat['t'] < str(td2)[5:20])].reset_index()

    return(np.mean(tp1['EEm']))
    
def calcMean2MinEpoch_fail(fullDat, inputMarker):
    startTime = fullDat[fullDat['Marker'] == inputMarker].index.tolist()
    td1 = fullDat['t'][startTime] + pd.to_timedelta('0 days 00:03:00')
    td2 = fullDat['t'][startTime] + pd.to_timedelta('0 days 00:04:30')
    tp1 = fullDat[(dat['t'] > str(td1)[5:21]) & (fullDat['t'] < str(td2)[5:21])].reset_index()

    return(np.mean(tp1['EEm']))

def saveTS(fullDat, inputMarker):
    startTime = fullDat[fullDat['Marker'] == inputMarker].index.tolist()
    td1 = fullDat['t'][startTime] + pd.to_timedelta('0 days 00:03:00')
    td2 = fullDat['t'][startTime] + pd.to_timedelta('0 days 00:04:30')
    tp1 = fullDat[(dat['t'] > str(td1)[5:20]) & (fullDat['t'] < str(td2)[5:20])].reset_index()

    return(tp1['EEm'])

def saveTS_late(fullDat, inputMarker):
    startTime = fullDat[fullDat['Marker'] == inputMarker].index.tolist()
    td1 = fullDat['t'][startTime] + pd.to_timedelta('0 days 00:03:00')
    td2 = fullDat['t'][startTime] + pd.to_timedelta('0 days 00:04:30')
    tp1 = fullDat[(dat['t'] > str(td1)[5:21]) & (fullDat['t'] < str(td2)[5:21])].reset_index()

    return(tp1['EEm'])

#Parse the column names
headerList = ['zero','one','two','three','four','five','six','seven','eight','t', 'Rf', 'VT', 'VE', 'IV', 'VO2', 'VCO2', 'RQ', 'O2exp', 'CO2exp','VE/VO2', 'VE/VCO2',
'VO2/Kg', 'METS', 'HR', 'VO2/HR', 'FeO2', 'FeCO2', 'FetO2', 'FetCO2', 'FiO2', 'FiCO2', 'PeO2',
 'PeCO2', 'PetO2', 'PetCO2', 'SpO2', 'Phase', 'Marker', 'AmbTemp', 'RHAmb', 'AnalyPress', 'PB', 'EEkc',	
 'EEh',	'EEm', 'EEtot',	'EEkg',	'PRO', 'FAT', 'CHO', 'PRO%', 'FAT%', 'CHO%', 'npRQ', 'GPSDist', 'Ti',
 'Te', 'Ttot', 'Ti/Ttot', 'VD/VTe',	'LogVE', 'tRel', 'markSpeed', 'markDist', 'Phase time', 'VO2/Kg%Pred','BR',	
 'VT/Ti', 'HRR', 'PaCO2_e']

fPath = 'C:\\Users\\daniel.feeney\\OneDrive - Boa Technology Inc\\Desktop\\MetabolicExample\\'
#entries = os.listdir(fPath)
fileExt = r".xlsx"
entries = [fName for fName in os.listdir(fPath) if fName.endswith(fileExt)]

configLabels = ['DD_1','DDP_1','Lace_1','LaceP_1','LaceP_2', 'Lace_2', 'DDP_2', 'DD_2']

met = []
trial = []
config = []

for file in entries:
    try:
        fName = file
        
        dat = pd.read_excel(fPath+fName, skiprows=2, names = headerList)
        
        dat = dat.drop(dat.columns[[0, 1, 2,3,4,5,6,7,8]], axis=1)  
        dat['t'] = pd.to_timedelta(dat['t'].astype(str)) #Convert time to time delta from start
                
        #2 min epochs. different dataframe
        try:
            met.append(calcMean2MinEpoch(dat, configLabels[0]))
            shoe1 = saveTS(dat, configLabels[0])
        except:
            met.append(calcMean2MinEpoch_fail(dat, configLabels[0]))  
            shoe1 = saveTS_late(dat, configLabels[0])
        config.append(configLabels[0].split('_')[0])
        trial.append(configLabels[0].split('_')[1])
        
        try:
            met.append(calcMean2MinEpoch(dat, configLabels[1]))
            shoe2 = saveTS(dat, configLabels[1])
        except:
            met.append(calcMean2MinEpoch_fail(dat, configLabels[1]))
            shoe2 = saveTS_late(dat, configLabels[1])
        config.append(configLabels[1].split('_')[0])
        trial.append(configLabels[1].split('_')[1])
        
        try:
            met.append(calcMean2MinEpoch(dat, configLabels[2]))
            shoe3 = saveTS(dat, configLabels[2])
        except:
            met.append(calcMean2MinEpoch_fail(dat, configLabels[2])) 
            shoe3 = saveTS_late(dat, configLabels[2])
        config.append(configLabels[2].split('_')[0])
        trial.append(configLabels[2].split('_')[1])
        
        # Second set of three trials
        try:
            met.append(calcMean2MinEpoch(dat, configLabels[3]))
            shoe4 = saveTS(dat, configLabels[3])
        except:
            met.append(calcMean2MinEpoch_fail(dat, configLabels[3])) 
            shoe4 = saveTS_late(dat, configLabels[3])
        config.append(configLabels[3].split('_')[0])
        trial.append(configLabels[3].split('_')[1])
        
        try:
            met.append(calcMean2MinEpoch(dat, configLabels[4]))
            shoe5 = saveTS(dat, configLabels[4])
        except:
            met.append(calcMean2MinEpoch_fail(dat, configLabels[4]))
            shoe5 = saveTS_late(dat, configLabels[4])
        config.append(configLabels[4].split('_')[0])
        trial.append(configLabels[4].split('_')[1])
        
        try:
            met.append(calcMean2MinEpoch(dat, configLabels[5]))
            shoe6 = saveTS(dat, configLabels[5])
        except:
            met.append(calcMean2MinEpoch_fail(dat, configLabels[5]))   
            shoe6 = saveTS_late(dat, configLabels[5])
        config.append(configLabels[5].split('_')[0])
        trial.append(configLabels[5].split('_')[1])
  
        try:
            met.append(calcMean2MinEpoch(dat, configLabels[6]))
            shoe7 = saveTS(dat, configLabels[6])
        except:
            met.append(calcMean2MinEpoch_fail(dat, configLabels[6]))   
            shoe7 = saveTS_late(dat, configLabels[6])
        config.append(configLabels[6].split('_')[0])
        trial.append(configLabels[6].split('_')[1])
        
        try:
            met.append(calcMean2MinEpoch(dat, configLabels[7]))
            shoe8 = saveTS(dat, configLabels[7])
        except:
            met.append(calcMean2MinEpoch_fail(dat, configLabels[7]))   
            shoe8 = saveTS_late(dat, configLabels[7])
        config.append(configLabels[7].split('_')[0])
        trial.append(configLabels[7].split('_')[1])
        
    except:
        print(file)

        

outcomes = pd.DataFrame({'Trial':list(trial), 'Config':list(config), 'EE':list(met)})

avgs = outcomes.groupby('Config')['EE'].mean()
sds = outcomes.groupby('Config')['EE'].std()
configs = ['DD_1','DDP_1','Lace_1','LaceP_1','LaceP_2', 'Lace_2', 'DDP_2', 'DD_2']

plt.rcParams['font.size'] = 18
fig = plt.figure(figsize = (10,8))
plt.bar(configs, avgs)
plt.ylim([10,16])
plt.xlabel('Brand')
plt.ylabel('Energetic Cost (W/kg)')

def calcPctDiff(val1, val2):
    absDiff = abs(val1 - val2)
    avgVal = (val1 + val2)/2
    return round((absDiff/avgVal) * 100,2)

shoeGrp1 = outcomes.groupby('Config')['EE'].mean()[0]
shoeGrp2 = outcomes.groupby('Config')['EE'].mean()[2]

calcPctDiff(shoeGrp1, shoeGrp2)
if shoeGrp1 < shoeGrp2:
    print(str(shoeGrp1),calcPctDiff(shoeGrp1, shoeGrp2),'better than ', str(shoeGrp2))
else:
    print(str(shoeGrp2),calcPctDiff(shoeGrp1, shoeGrp2),'better than ', str(shoeGrp1))


fig = plt.figure(figsize = (10,8))
plt.plot(shoe1, label = configLabels[0])
plt.plot(shoe2, label = configLabels[1])
plt.plot(shoe3, label = configLabels[2])
plt.plot(shoe4, label = configLabels[3])
plt.plot(shoe5, label = configLabels[4])
plt.plot(shoe6, label = configLabels[5])
plt.plot(shoe7, label = configLabels[6])
plt.plot(shoe8, label = configLabels[7])
plt.legend()


#outcomes2.to_csv('C:\\Users\\daniel.feeney\\Boa Technology Inc\\PFL - General\\AgilityPerformanceData\\BOA_InternalStrap_July2021\\Metabolics\\MetResults2.csv')
