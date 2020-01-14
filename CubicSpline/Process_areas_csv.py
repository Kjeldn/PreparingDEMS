# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 15:29:03 2019

@author: ericv
"""

import os
import pandas as pd

input_files = [r"Z:\800 Operational\c07_hollandbean\Season evaluation\Areas\Joke Visser-906031020-GR_plant_count.csv"]
#,
#r"Z:\800 Operational\c01_verdonk\Wever west\Season evaluation\areas.csv",
#r"Z:\800 Operational\c01_verdonk\Rijweg stalling 1\Season evaluation\areas.csv",
#r"Z:\800 Operational\c08_biobrass\AZ91\Season evaluation\areas.csv"
#]

output_file = r"Z:\800 Operational\c07_hollandbean/2019_summary.xlsx"

df3 = pd.DataFrame()

for input_file in input_files:
    #input_file = r"Z:\800 Operational\c01_verdonk\Wever west\Season evaluation\areas.csv"
    #output_file = os.path.dirname(input_file) + '\summary.xlsx'
    
    if input_file.split('\\')[2] == 'c08_biobrass':
        df = pd.read_csv(input_file, sep = ',')
    else:
        df = pd.read_csv(input_file, sep = ';')
    
    
    df = df[(df != 0).all(1)]
    
    for i in range(2,len(df.columns)):
        mean = df.iloc[:,i].mean()
        std = df.iloc[:,i].std()    
        df.iloc[:,i][(df.iloc[:,i] > mean+2*std)] = 0
        df.iloc[:,i][(df.iloc[:,i] < mean-2*std)] = 0
    df = df[(df != 0).all(1)]
    
    
    mean = df.mean()
    std = df.std()
    median = df.median()
    var = df.var()
    plot = [input_file.split('\\')[3]] * len(var)
    
    
    df2 = pd.DataFrame({'mean': mean, 'median':median, 'std': std, 'var':var, 'plot':plot})
    
df3 = df3.append(df2)

df3.to_excel(output_file, header = True)
