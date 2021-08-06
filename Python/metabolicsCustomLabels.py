# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 14:17:01 2021
Analyze metabolic data with given inputs
@author: Daniel.Feeney
"""

import pandas as pd
import numpy as np
import os

def calcMean2MinEpoch(fullDat, inputMarker):
    startTime = fullDat[fullDat['Marker'] == inputMarker].index.tolist()
    td1 = fullDat['t'][startTime] + pd.to_timedelta('0 days 00:03:00')
    td2 = fullDat['t'][startTime] + pd.to_timedelta('0 days 00:05:00')
    tp1 = fullDat[(dat['t'] > str(td1)[5:20]) & (fullDat['t'] < str(td2)[5:20])].reset_index()

    return(np.mean(tp1['EEm']))
    

def calcMean3min(fullDat, inputMarker):
    startTime = fullDat[fullDat['Marker'] == inputMarker].index.tolist()
    td1 = fullDat['t'][startTime] + pd.to_timedelta('0 days 00:03:00')
    td2 = fullDat['t'][startTime] + pd.to_timedelta('0 days 00:04:00')
    tp1 = fullDat[(fullDat['t'] > str(td1)[5:20]) & (fullDat['t'] < str(td2)[5:20])].reset_index()

    return(np.mean(tp1['EEm']))

def calcMean4min(fullDat, inputMarker):
    startTime = fullDat[fullDat['Marker'] == inputMarker].index.tolist()
    td1 = fullDat['t'][startTime] + pd.to_timedelta('0 days 00:04:00')
    td2 = fullDat['t'][startTime] + pd.to_timedelta('0 days 00:05:00')
    tp1 = fullDat[(dat['t'] > str(td1)[5:20]) & (fullDat['t'] < str(td2)[5:20])].reset_index()

    return(np.mean(tp1['EEm']))

#Parse the column names
headerList = ['zero','one','two','three','four','five','six','seven','eight','t', 'Rf', 'VT', 'VE', 'IV', 'VO2', 'VCO2', 'RQ', 'O2exp', 'CO2exp','VE/VO2', 'VE/VCO2',
'VO2/Kg', 'METS', 'HR', 'VO2/HR', 'FeO2', 'FeCO2', 'FetO2', 'FetCO2', 'FiO2', 'FiCO2', 'PeO2',
 'PeCO2', 'PetO2', 'PetCO2', 'SpO2', 'Phase', 'Marker', 'AmbTemp', 'RHAmb', 'AnalyPress', 'PB', 'EEkc',	
 'EEh',	'EEm', 'EEtot',	'EEkg',	'PRO', 'FAT', 'CHO', 'PRO%', 'FAT%', 'CHO%', 'npRQ', 'GPSDist', 'Ti',
 'Te', 'Ttot', 'Ti/Ttot', 'VD/VTe',	'LogVE', 'tRel', 'markSpeed', 'markDist', 'Phase time', 'VO2/Kg%Pred','BR',	
 'VT/Ti', 'HRR', 'PaCO2_e']

fPath = 'C:\\Users\\daniel.feeney\\Boa Technology Inc\\PFL - General\\AgilityPerformanceData\\BOA_InternalStrap_July2021\\Metabolics\\'
entries = os.listdir(fPath)

configLabels = ['4guide_Start','Single_Start','Nothing_Start']

met = []
subject = []
config = []

met2 = []
subject2 = []
config2 = []

for file in entries:
    try:
        fName = file
        
        dat = pd.read_excel(fPath+fName, skiprows=2, names = headerList)
        
        dat = dat.drop(dat.columns[[0, 1, 2,3,4,5,6,7,8]], axis=1)  
        dat['t'] = pd.to_timedelta(dat['t'].astype(str)) #Convert time to time delta from start
                
        # 1st Config
        met.append(calcMean3min(dat, configLabels[0]))
        config.append(configLabels[0].split('_')[0])
        subject.append(fName.split('_')[0])
        met.append(calcMean4min(dat, configLabels[0]))
        config.append(configLabels[0].split('_')[0])
        subject.append(fName.split('_')[0])
        
        # 2nd Config
        met.append(calcMean3min(dat, configLabels[1]))
        config.append(configLabels[1].split('_')[0])
        subject.append(fName.split('_')[0])
        met.append(calcMean4min(dat, configLabels[1]))
        config.append(configLabels[1].split('_')[0])
        subject.append(fName.split('_')[0])

        # 3rd Config
        met.append(calcMean3min(dat, configLabels[2]))
        config.append(configLabels[2].split('_')[0])
        subject.append(fName.split('_')[0])
        met.append(calcMean4min(dat, configLabels[2]))
        config.append(configLabels[2].split('_')[0])
        subject.append(fName.split('_')[0])

        #2 min epochs. different dataframe
        met2.append(calcMean2MinEpoch(dat, configLabels[0]))
        config2.append(configLabels[0].split('_')[0])
        subject2.append(fName.split('_')[0])
        
        met2.append(calcMean2MinEpoch(dat, configLabels[1]))
        config2.append(configLabels[1].split('_')[0])
        subject2.append(fName.split('_')[0])
        
        met2.append(calcMean2MinEpoch(dat, configLabels[2]))
        config2.append(configLabels[2].split('_')[0])
        subject2.append(fName.split('_')[0])
        
        
    except:
        print(file)

        
outcomes = pd.DataFrame({'Subject':list(subject),'Config':list(config), 'EE':list(met)})

outcomes.to_csv('C:\\Users\\daniel.feeney\\Boa Technology Inc\\PFL - General\\AgilityPerformanceData\\BOA_InternalStrap_July2021\\Metabolics\\MetResults.csv')#, mode = 'a', header = False)

outcomes2 = pd.DataFrame({'Subject':list(subject2), 'Config':list(config2), 'EE':list(met2)})
outcomes2.to_csv('C:\\Users\\daniel.feeney\\Boa Technology Inc\\PFL - General\\AgilityPerformanceData\\BOA_InternalStrap_July2021\\Metabolics\\MetResults2.csv')