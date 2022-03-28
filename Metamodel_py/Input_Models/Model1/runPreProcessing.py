#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 12:09:42 2022

@author: yair
"""
import pandas as pd

# Import Model packages:
from Model1.Code import definitions
from Model1.Code import preProcessing

paths = definitions.paths
submodels = definitions.submodels

Input_path = paths['Input']
# 1. Get training data:
# 1.0 Read raw data as dataFrame:
raw_data_name = 'raw_data_depletion.csv'
df_raw_data_depletion =\
    pd.read_csv(Input_path+raw_data_name, header=None)

# 1.0.1 Crop and scale raw data:
# x_array, y_array, z_array =\
#     preProcessing.cropAndScaleRawData(df_raw_data_depletion)

df_trainingData_depletion_pivot =\
    preProcessing.rawDataToDataFramePivot(df_raw_data_depletion)

# Save dataFrame pivot as .csv:
df_trainingData_depletion_pivot.to_csv(
    Input_path+"/df_trainingData_depletion_pivot.csv")

# Get trainingData aranged as dataFrame in columns (flatten):
df_trainingData_depletion_flatten =\
    preProcessing.pivotToFlatten(df_trainingData_depletion_pivot)

# Save dataFrame flatten as .csv:
df_trainingData_depletion_flatten.to_csv(
    Input_path+"/df_trainingData_depletion_flatten.csv")
