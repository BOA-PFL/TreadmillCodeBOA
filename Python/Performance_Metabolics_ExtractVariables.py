# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 14:17:01 2021
Analyze metabolic data with given inputs
@author: Daniel.Feeney
"""

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tkinter import messagebox 

fPath = 'C:\\Users\\eric.honert\\Boa Technology Inc\\PFL Team - General\\Testing Segments\\Cycling Performance Tests\\2025_Performance_CyclingLacevBOA_Specialized\\Metabolics\\'
fileExt = r".xlsx"
entries = [fName for fName in os.listdir(fPath) if fName.endswith(fileExt)]

save_on = 1
data_check = 1

### set plot font size ###
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Define functions
def degbug2MinEpoch(fullDat, inputMarker):
    """
    Function to plot metabolic metrics from minute 3 to 5 of the
    select portion of the trial as indicated by the inputMarker

    Parameters
    ----------
    fullDat : dataframe
        Dataframe that contains all of the metabolic data
    inputMarker : string
        String indicating the start of a particular portion of the metabolic
        trial

    Returns
    -------
    None

    """
    startidx = np.array(fullDat[fullDat['Marker'] == inputMarker].index)[0]
    # Plot from 3 min (180 seconds) to 5 min (300 sec)
    tp1 = fullDat[(fullDat['t'] > fullDat['t'][startidx]+180) & (fullDat['t'] < fullDat['t'][startidx]+300)].reset_index()
    tp1['t'] = tp1['t'] - tp1['t'][0]
    
    plt.figure()
    plt.subplot(1,3,1)
    plt.plot(tp1['t'],tp1['EEm'])
    plt.title('Energy Expendature')
    plt.xlabel('Time (s)')
    plt.ylim([min(fullDat['EEm']),1.2*max(fullDat['EEm'])])
    plt.subplot(1,3,2)
    plt.plot(tp1['t'],tp1['RQ'])
    plt.title('RQ')
    plt.xlabel('Time (s)')
    plt.ylim([min(fullDat['RQ']),1.2*max(fullDat['RQ'])])
    plt.subplot(1,3,3)
    plt.plot(tp1['t'],tp1['VO2'])
    plt.title('V02')
    plt.xlabel('Time (s)')
    plt.ylim([min(fullDat['VO2']),1.2*max(fullDat['VO2'])])
    plt.tight_layout()
    

def calcMean3min(fullDat, inputMarker, inputMetric):
    """
    Function to compute the mean energy consumtion from minute 3 to 4 of the
    select portion of the trial as indicated by the inputMarker
    
    Parameters
    ----------
    fullDat : dataframe
        Dataframe that contains all of the metabolic data
    inputMarker : string
        String indicating the start of a particular portion of the metabolic
        trial
    inputMetric : string
        String indicating the metric to average for the intended period
    
    Returns
    -------
    (in function form) : float
        Mean metric

    """
    startidx = np.array(fullDat[fullDat['Marker'] == inputMarker].index)[0]
    # Plot from 3 min (180 seconds) to 4 min (240 sec)
    tp1 = fullDat[(fullDat['t'] > fullDat['t'][startidx]+180) & (fullDat['t'] < fullDat['t'][startidx]+240)].reset_index()

    return(np.mean(tp1[inputMetric]))

def calcMean4min(fullDat, inputMarker, inputMetric):
    """
    Function to compute the mean energy consumtion from minute 4 to 5 of the
    select portion of the trial as indicated by the inputMarker
    
    Parameters
    ----------
    fullDat : dataframe
        Dataframe that contains all of the metabolic data
    inputMarker : string
        String indicating the start of a particular portion of the metabolic
        trial
    inputMetric : string
        String indicating the metric to average for the intended period
    
    Returns
    -------
    (in function form) : float
        Mean metric

    """
    startidx = np.array(fullDat[fullDat['Marker'] == inputMarker].index)[0]
    # Plot from 4 min (240 seconds) to 5 min (300 sec)
    tp1 = fullDat[(fullDat['t'] > fullDat['t'][startidx]+240) & (fullDat['t'] < fullDat['t'][startidx]+300)].reset_index()

    return(np.mean(tp1[inputMetric]))

#Parse the column names
headerList = ['zero','one','two','three','four','five','six','seven','eight','t', 'Rf', 'VT', 'VE', 'IV', 'VO2', 'VCO2', 'RQ', 'O2exp', 'CO2exp','VE/VO2', 'VE/VCO2',
'VO2/Kg', 'METS', 'HR', 'VO2/HR', 'FeO2', 'FeCO2', 'FetO2', 'FetCO2', 'FiO2', 'FiCO2', 'PeO2',
 'PeCO2', 'PetO2', 'PetCO2', 'SpO2', 'Phase', 'Marker', 'AmbTemp', 'RHAmb', 'AnalyPress', 'PB', 'EEkc',	
 'EEh',	'EEm', 'EEtot',	'EEkg',	'PRO', 'FAT', 'CHO', 'PRO%', 'FAT%', 'CHO%', 'npRQ', 'GPSDist', 'Ti',
 'Te', 'Ttot', 'Ti/Ttot', 'VD/VTe',	'LogVE', 'tRel', 'markSpeed', 'markDist', 'Phase time', 'VO2/Kg%Pred','BR',	
 'VT/Ti', 'HRR', 'PaCO2_e']

# Preallocate variables
met = []
subject = []
config = []
order = []
VO2 = []
RQ = []
time = []
badFileList = []

# Index through all files
for fName in entries:
    # try:
        print(fName)
        
        dat = pd.read_excel(fPath+fName, skiprows=2, names = headerList)
        
        dat = dat.drop(dat.columns[[0, 1, 2,3,4,5,6,7,8]], axis=1)
        # Convert the datetime in the time column to integer seconds
        dat['t'] = np.array([((timestr.hour*60+timestr.minute)*60+timestr.second) for timestr in dat['t']])
        
        # Obtain the config names from the file
        configLabels = np.array(dat['Marker'][dat['Marker'].notna()])
        
        
        # Create saving folder for the metabolic plots
        saveFolder = fPath + 'MetabolicPlots'
         
        if os.path.exists(saveFolder) == False:
            os.mkdir(saveFolder)
        
        for configLab in configLabels:
            # Create debugging plot
            answer = True
            if data_check == 1:
                degbug2MinEpoch(dat, configLab)
                plt.savefig(saveFolder + '/' + fName.split('.')[0]+'_'+configLab +'.png')
                answer = messagebox.askyesno("Question","Is data clean?")
                plt.close('all')
                if answer == False:
                    print('Adding file to bad file list')
                    badFileList.append(fName)
            
            if answer == True:
                print('Estimating point estimates')
                # 1 min epochs
                met.append(calcMean3min(dat, configLab,'EEm'))
                VO2.append(calcMean3min(dat, configLab,'VO2'))
                RQ.append(calcMean3min(dat, configLab,'RQ'))
                config.append(configLab.split('_')[0])
                order.append(configLab.split('_')[1])
                subject.append(fName.split('.')[0])
                time.append(3)
                
                met.append(calcMean4min(dat, configLab,'EEm'))
                VO2.append(calcMean4min(dat, configLab,'VO2'))
                RQ.append(calcMean4min(dat, configLab,'RQ'))
                config.append(configLab.split('_')[0])
                order.append(configLab.split('_')[1])
                subject.append(fName.split('.')[0])
                time.append(4)
           
    # except:
    #     print(fName)


# Compile desired outcomes to a single variable         
outcomes = pd.DataFrame({'Subject':list(subject),'Config':list(config), 'Order':list(order), 'Time':list(time),
                         'EE':list(met), 'VO2':list(VO2),'RQ':list(RQ)})

outfileName = fPath+'0_MetabolicOutcomes.csv'
if save_on == 1:
    outcomes.to_csv(outfileName, header=True, index = False)

