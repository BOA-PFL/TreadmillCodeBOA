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
import tkinter as tk
from tkinter import messagebox 
from TreadmillFunctions import loadProcessingDecisions, saveProcessingDecision

fPath = 'C:\\Users\\eric.honert\\Boa Technology Inc\\PFL Team - General\\Testing Segments\\Cycling Performance Tests\\2025_Performance_CyclingLacevBOA_Specialized\\Metabolics\\'
fileExt = r".xlsx"
entries = [fName for fName in os.listdir(fPath) if fName.endswith(fileExt) and not fName.startswith('0_')]

# Names this script's decision file, 0_ProcessingDecisions_<scriptName>.csv.
# One data folder is often read by several scripts, and a trial that is no
# good for one analysis can be fine for another, so each keeps its own
# record rather than one script's rejection governing the rest.
scriptName = 'Metabolics'

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

save_on = 1
# 0 = no data check, nothing is asked or recorded
# 1 = ask only about trials with no recorded decision (the normal setting)
# 2 = re-review EVERY trial, including ones previously rejected. Use after
#     changing processing, when earlier decisions may no longer hold. This
#     mode is good for reruning files with a different threshold or examining
#     files after a different filter has been applied.
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
            # The decision is keyed on file AND config, not file alone: one
            # metabolic file holds several configs and is judged once per
            # config, so accepting one says nothing about the others.
            decisionKey = fName + '_' + configLab
            tmpSub = fName.split('.')[0]

            # A config rejected in a previous run is skipped. The file is
            # still there to open and look at: the recorded decision governs,
            # not the folder
            priorDecision = decisions.get(decisionKey, '')
            if priorDecision == 'bad' and data_check != 2:
                print('Previously excluded, skipping: ' + decisionKey)
                continue

            # Create debugging plot
            answer = True # if data check is off.
            if priorDecision == 'good' and data_check == 1:
                print('Previously accepted, not asking again: ' + decisionKey)
            elif data_check >= 1:
                degbug2MinEpoch(dat, configLab)
                plt.savefig(saveFolder + '/' + fName.split('.')[0]+'_'+configLab +'.png')
                answer = messagebox.askyesno("Question","Is data clean?")
                plt.close('all')
                if answer == False:
                    print('Adding file to bad file list')
                    badFileList.append(fName)

            # Record the decision straight away, inside the loop, so a run
            # that fails later keeps every decision made so far
            if priorDecision == '' or data_check == 2:
                if answer == False:
                    saveProcessingDecision(fPath, scriptName, decisionKey, 'bad',
                                           Subject=tmpSub, Config=configLab,
                                           reason='rejected at the data check')
                else:
                    saveProcessingDecision(fPath, scriptName, decisionKey, 'good',
                                           Subject=tmpSub, Config=configLab)

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
if save_on == 1 and data_check < 2:
    # Note: written fresh each run, never appended. Appending duplicates
    # every row when a study is re-processed after more data is collected.
    outcomes.to_csv(outfileName, header=True, index = False)

