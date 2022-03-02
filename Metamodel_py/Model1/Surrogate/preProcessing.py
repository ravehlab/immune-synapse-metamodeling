# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 17:08:06 2022

@author: yairn
"""

import numpy as np
import pandas as pd

from Model1.Surrogate import definitions
# from Model1.Surrogate import preProcessing
from Model1.Surrogate import plotting

plots = definitions.plots
#################################################
# Training data to dataFrame:


def cropAndScaleRawData(df_z_raw_data):
    """
    Gets: DataFrame of raw training z data.
    Returns: x_array, y_array, z_array.
    Calling: None.
    Description:
    """

    nmToPixels = 1

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

    # Set x_array and y_array:
    [x_array, y_array] = np.meshgrid(x0[selected_x_indices],
                                     y0[selected_y_indices])

    # Select z_array according to x and y indices:
    z_array1 = z_array0[selected_y_indices, :]
    z_array = z_array1[:, selected_x_indices]

    return x_array, y_array, z_array

#################################################


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


def pivotToFlatten(df_pivot):
    """
    Gets: x_array, y_array, z_array, definitions.
    Returns: df_trainingData_pivot, df_trainingData_flatten.
    Calling: None.
    Called by:
    Description:
    """
    x = df_pivot.columns
    y = df_pivot.index
    z_array = df_pivot.values

    # Set x_array and y_array:
    [x_array, y_array] = np.meshgrid(x, y)

    x_name_units = definitions.axes_names_units[0]  # Read from 'definitions'
    y_name_units = definitions.axes_names_units[1]  # Read from 'definitions'
    z_name_units = definitions.axes_names_units[2]  # Read from 'definitions'

    # f is for flatten:
    df_flatten = pd.DataFrame(
        np.array([x_array.flatten(),
                  y_array.flatten(),
                  z_array.flatten()]).T,
        columns=[x_name_units, y_name_units, z_name_units])

    return df_flatten
#################################################
# 1.3 Plot training data:


def plotTrainingData(df_pivot):
    """
    Gets: df_pivot.
    Returns: None.
    Calling: None.
    Called by:
    Description: Plotting a heatmap of the training data.
    """

    nRows = plots['nRoWs']

    DataToPlot = nRows*[None]
    DataToPlot[0] = [[df_pivot.columns,
                      df_pivot.index],
                     [df_pivot.values]]

    plotWhat = [True, False, False, False]

    plotting.plotData(DataToPlot, plotWhat)

#################################################
# Get training data:


# def get_training_data(data_path):
#     """
#     Gets: Path to raw data.
#     Returns: df_trainingData_pivot, df_trainingData_flatten.
#     Calling: cropAndScaleRawData, trainingDataToDataFrame.
#     Called by:
#     Description:
#     """

#     # 1.1 Read raw training data for the model:
#     df_raw_data = pd.read_csv(data_path, header=None)

#     # 1.2 Crop and scale raw data, assign values and units for x and y axes:
#     x_array, y_array, z_array =\
#         preProcessing.cropAndScaleRawData(df_raw_data)

#     # 1.3 Arange training data in pandas dataFrame (df):
#     df_trainingData_pivot, df_trainingData_flatten =\
#         preProcessing.trainingDataToDataFrame(x_array, y_array, z_array)

#     return df_trainingData_pivot, df_trainingData_flatten
