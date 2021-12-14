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

fPath = 'C:/Users/Daniel.Feeney/Dropbox (Boa)/FBS Abstract/Metabolic (31 of 31)/'
entries = os.listdir(fPath)

for file in entries[93:96]:
    try:
        fName = file
        
        dat = pd.read_excel(fPath+fName, skiprows=2, names = headerList)
        
        dat = dat.drop(dat.columns[[0, 1, 2,3,4,5,6,7,8]], axis=1)  
        dat['t'] = pd.to_timedelta(dat['t'].astype(str)) #Convert time to time delta from start
        
        tp1 = dat[(dat['t'] > '0 days 00:05:00') & (dat['t'] < '0 days 00:08:00')].reset_index()
        tp2 = dat[(dat['t'] > '0 days 00:08:00') & (dat['t'] < '0 days 00:11:00')].reset_index()
        tp3 = dat[(dat['t'] > '0 days 00:11:00') & (dat['t'] < '0 days 00:14:00')].reset_index()
        tp4 = dat[(dat['t'] > '0 days 00:14:00') & (dat['t'] < '0 days 00:17:00')].reset_index()
        tp5 = dat[(dat['t'] > '0 days 00:17:00') & (dat['t'] < '0 days 00:20:00')].reset_index()
        tp6 = dat[(dat['t'] > '0 days 00:20:00') & (dat['t'] < '0 days 00:23:00')].reset_index()
        tp7 = dat[(dat['t'] > '0 days 00:23:00') & (dat['t'] < '0 days 00:26:00')].reset_index()
        tp8 = dat[(dat['t'] > '0 days 00:26:00') & (dat['t'] < '0 days 00:29:00')].reset_index()
        tp9 = dat[(dat['t'] > '0 days 00:29:00') & (dat['t'] < '0 days 00:32:00')].reset_index()
        tp10 = dat[(dat['t'] > '0 days 00:32:00') & (dat['t'] < '0 days 00:35:00')].reset_index()
        tp11 = dat[(dat['t'] > '0 days 00:32:00') & (dat['t'] < '0 days 00:35:00')].reset_index()
        tp12 = dat[(dat['t'] > '0 days 00:35:00') & (dat['t'] < '0 days 00:38:00')].reset_index()
        tp13 = dat[(dat['t'] > '0 days 00:38:00') & (dat['t'] < '0 days 00:41:00')].reset_index()
        
        
        # plt.plot(tp1['EEm'], label='TP1')
        # plt.plot(tp2['EEm'], label = 'TP2')
        # plt.plot(tp3['EEm'], label = 'TP3')
        # plt.plot(tp4['EEm'], label = 'TP4')
        # plt.plot(tp5['EEm'], label = 'TP5')
        # plt.plot(tp6['EEm'], label = 'TP6')
        # plt.plot(tp7['EEm'], label = 'TP7')
        # plt.plot(tp8['EEm'], label = 'TP8')
        # plt.plot(tp9['EEm'], label = 'TP9')
        # plt.plot(tp10['EEm'], label = 'TP10')
        # plt.xlabel('Time')
        # plt.ylabel('EE (W/kg)')
        # plt.legend()
        
        conditions = np.array(['TP1','TP2','TP3','TP4','TP5','TP6','TP7','TP8', 'TP9','TP10','TP11','TP12','TP13'])
        Subject = list(np.repeat(fName.split('_')[0],len(conditions)))
        Order = list(np.repeat(fName.split('_')[1],len(conditions)))
        Config = list(np.repeat(fName.split('_')[2],len(conditions)))
        
        outcomes = pd.DataFrame({'Subject':list(Subject),'TimePoints':list(conditions), 'Order':list(Order), 'Config':list(Config),
                      'EEm':list([np.mean(tp1['EEm']), np.mean(tp2['EEm']), np.mean(tp3['EEm']), np.mean(tp4['EEm']), 
                                  np.mean(tp5['EEm']), np.mean(tp6['EEm']), np.mean(tp7['EEm']), np.mean(tp8['EEm']),
                                  np.mean(tp9['EEm']), np.mean(tp10['EEm']), np.mean(tp11['EEm']), np.mean(tp12['EEm']), np.mean(tp13['EEm'])]),
                      'Temp':list([np.mean(tp1['AmbTemp']), np.mean(tp2['AmbTemp']), np.mean(tp3['AmbTemp']), np.mean(tp4['AmbTemp']),
                                    np.mean(tp5['AmbTemp']), np.mean(tp6['AmbTemp']), np.mean(tp7['AmbTemp']), np.mean(tp8['AmbTemp']),
                                    np.mean(tp9['AmbTemp']), np.mean(tp10['AmbTemp']), np.mean(tp11['AmbTemp']), np.mean(tp12['AmbTemp']),np.mean(tp13['AmbTemp'])])})
                      # 'HR':list([np.mean(tp1['HR']), np.mean(tp2['HR']), np.mean(tp3['HR']), np.mean(tp4['HR']),
                      #            np.mean(tp5['HR']), np.mean(tp6['HR']), np.mean(tp7['HR']), np.mean(tp8['HR'])]),
                      # 'VO2':list([np.mean(tp1['VO2/Kg']), np.mean(tp2['VO2/Kg']), np.mean(tp3['VO2/Kg']), np.mean(tp4['VO2/Kg']),
                      #            np.mean(tp5['VO2/Kg']), np.mean(tp6['VO2/Kg']), np.mean(tp7['VO2/Kg']), np.mean(tp8['VO2/Kg'])]),

        
        outcomes.to_csv('C:/Users/Daniel.Feeney/Dropbox (Boa)/FBS Abstract/MetResults13.csv', mode='a', header=False)
    except:
        print(file)


