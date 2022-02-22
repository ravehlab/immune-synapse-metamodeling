#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 17:15:29 2022

@author: yair
"""

#################################################
"""
# 1. Get training data:
# 1.1 Read raw training data for model1 from Input folder:
df_dep_raw_data = pd.read_csv(
    Model_path+'/Input/dep_raw_data.csv', header=None)

# 1.2 Crop and scale raw data to nanometers,assign values and units for
# x and y axes:
t_array, k_array, dep_training_data_nm =\
    preprocessing.cropAndScaleRawData(df_dep_raw_data)

# 1.3 Arange training data in pandas dataFrame (df):
df_dep_nm = preprocessing.trainingDataToDataFrame(
    t_array, k_array, dep_training_data_nm)

# 1.4 Plot training data:
nRows = 4

DataToPlot = nRows*[None]
DataToPlot[0] = [[df_dep_nm.columns,
                  df_dep_nm.index],
                  [df_dep_nm.values]]
plotWhat = [True, False, False, False]

plotting.plotData(DataToPlot, plotWhat)
"""
#################################################
# equation_dep = parametersNames_dep[0] + \
#                 "+" + \
#                 parametersNames_dep[1] + \
#                 "*" + \
#                 "t" + \
#                 "+" + \
#                 parametersNames_dep[2] +\
#                 "*" + \
#                 "k"

# Get fit parameters:

