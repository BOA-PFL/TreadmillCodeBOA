# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 15:46:24 2021

@author: Daniel.Feeney
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 09:42:15 2021

@author: Daniel.Feeney
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

#Parse the column names
headerList = ['zero','one','two','three','four','five','six','seven','eight','t', 'Rf', 'VT', 'VE', 'IV', 'VO2', 'VCO2', 'RQ', 'O2exp', 'CO2exp','VE/VO2', 'VE/VCO2',
'VO2/Kg', 'METS', 'HR', 'VO2/HR', 'FeO2', 'FeCO2', 'FetO2', 'FetCO2', 'FiO2', 'FiCO2', 'PeO2',
 'PeCO2', 'PetO2', 'PetCO2', 'SpO2', 'Phase', 'Marker', 'AmbTemp', 'RHAmb', 'AnalyPress', 'PB', 'EEkc',	
 'EEh',	'EEm', 'EEtot',	'EEkg',	'PRO', 'FAT', 'CHO', 'PRO%', 'FAT%', 'CHO%', 'npRQ', 'GPSDist', 'Ti',
 'Te', 'Ttot', 'Ti/Ttot', 'VD/VTe',	'LogVE', 'tRel', 'markSpeed', 'markDist', 'Phase time', 'VO2/Kg%Pred','BR',	
 'VT/Ti', 'HRR', 'PaCO2_e']

fPath = 'C:/Users/Daniel.Feeney/Dropbox (Boa)/FBS Abstract/Metabolic (27 of 27)/'
entries = os.listdir(fPath)

for file in entries:
    try:
        fName = file
        
        dat = pd.read_excel(fPath+fName, skiprows=2, names = headerList)
        
        dat = dat.drop(dat.columns[[0, 1, 2,3,4,5,6,7,8]], axis=1)  
        dat['t'] = pd.to_timedelta(dat['t'].astype(str)) #Convert time to time delta from start
        
        tp1 = dat[(dat['t'] > '0 days 00:05:00') & (dat['t'] < '0 days 00:07:00')].reset_index()
        tp2 = dat[(dat['t'] > '0 days 00:07:00') & (dat['t'] < '0 days 00:09:00')].reset_index()
        tp3 = dat[(dat['t'] > '0 days 00:09:00') & (dat['t'] < '0 days 00:11:00')].reset_index()
        tp4 = dat[(dat['t'] > '0 days 00:38:00') & (dat['t'] < '0 days 00:40:00')].reset_index()
        tp5 = dat[(dat['t'] > '0 days 00:40:00') & (dat['t'] < '0 days 00:42:00')].reset_index()
        tp6 = dat[(dat['t'] > '0 days 00:36:00') & (dat['t'] < '0 days 00:38:00')].reset_index()
        

        diff1 = np.mean(tp4['EEm']) - np.mean(tp1['EEm'])
        diff2 = np.mean(tp5['EEm']) - np.mean(tp2['EEm'])
        diff3 = np.mean(tp6['EEm']) - np.mean(tp3['EEm'])
        diff4 = np.mean(tp4['EEm']) - np.mean(tp2['EEm'])
        diff5 = np.mean(tp5['EEm']) - np.mean(tp1['EEm'])
        
        conditions = np.array(['TP1','TP2','TP3','TP4','TP5'])
        Subject = list(np.repeat(fName.split('_')[0],len(conditions)))
        Order = list(np.repeat(fName.split('_')[1],len(conditions)))
        Config = list(np.repeat(fName.split('_')[2],len(conditions)))
        
        outcomes = pd.DataFrame({'Subject':list(Subject),'TimePoints':list(conditions), 'Order':list(Order), 'Config':list(Config),
                      'differences':list([diff1, diff2, diff3, diff4, diff5])})
        
        outcomes.to_csv('C:/Users/Daniel.Feeney/Dropbox (Boa)/FBS Abstract/MetDiffLarge.csv', mode='a', header=False)
    except:
        print(file)


