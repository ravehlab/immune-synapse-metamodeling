#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 17:15:29 2022

@author: yair
"""
import numpy as np
import pandas as pd

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

#################################################


def cropAndScaleRawData(df_z_raw_data):
    """
    Gets: DataFrame of raw training z data.
    Returns: x_array, y_array, z_array.
    Calling: None.
    Description:
    """

    nmToPixels = 10

    # df to array:
    z_raw_data = df_z_raw_data.values

    # Scale data to preferred units:
    z_array0 = nmToPixels*z_raw_data

    # Get size of original array:
    size_y, size_x = np.shape(z_raw_data)

    # x_axis:
    min_x = 0
    max_x = 100
    x0 = np.linspace(min_x, max_x, size_x)

    # y_axis:
    min_y = 5
    max_y = 100
    y0 = np.linspace(min_y, max_y, size_y)

    # select start indices for x and y:
    x_start_index = 1
    y_start_index = 1

    # Indices steps:
    x_step = 1
    y_step = 2

    # Get selected x and y indices:
    selected_x_indices = np.arange(x_start_index, size_x, x_step)
    selected_y_indices = np.arange(y_start_index, size_y, y_step)

    x = x0[selected_x_indices]
    y = y0[selected_y_indices]

    # Set x_array and y_array:
    [x_array, y_array] = np.meshgrid(x, y)

    # Select z_array according to x and y indices:
    z_array1 = z_array0[selected_y_indices, :]
    z_array = z_array1[:, selected_x_indices]

    return x_array, y_array, z_array
#################################################
# Training data to dataFrame:


def trainingDataToDataFrame(x_array, y_array, z_array):
    """
    Gets: x_array, y_array, z_array, definitions.
    Returns: df_trainingData_pivot, df_trainingData_flatten.
    Calling: None.
    Description:
    """

    x_name_units = 'time_sec'  # Read from 'definitions'
    y_name_units = 'k0_kTnm2'
    z_name_units = 'dep_nm'

    # f is for flatten:
    df_trainingData_flatten = pd.DataFrame(
        np.array([x_array.flatten(),
                  y_array.flatten(),
                  z_array.flatten()]).T,
        columns=[x_name_units, y_name_units, z_name_units])

    # flatten to pivot:
    df_trainingData_pivot = df_trainingData_flatten.pivot(
        y_name_units, x_name_units, z_name_units)

    return df_trainingData_pivot, df_trainingData_flatten
#################################################
