# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 17:08:06 2022

@author: yairn
"""

import numpy as np
import pandas as pd

import definitions
import plotting

plots = definitions.plots
data = definitions.data
#################################################
# Training data to dataFrame (no assigned index and columns):


def rawDataToDataFramePivot(df_z_raw_data):
    """
    Gets: DataFrame of raw training z data.
    Returns: x_array, y_array, z_array.
    Calling: None.
    Description:
    """

    #############################################
    # df to array:
    z_raw_data = df_z_raw_data.values

    # Scale data to preferred units:
    z_array0 = -np.log10(-z_raw_data)

    # Get size of original array:
    size_y, size_x = np.shape(z_raw_data)

    # x_axis:
    x0 = np.arange(-5, 0.0, 0.5)
    y0 = np.arange(-3, 0.25, 0.25)

    # select start indices for x and y:
    x_start_index = 0
    y_start_index = 0

    # Indices steps:
    x_step = 1
    y_step = 1

    # Get selected x and y indices:
    selected_x_indices = np.arange(x_start_index, size_x-2, x_step)
    selected_y_indices = np.arange(y_start_index, size_y-1, y_step)

    # Set x and y:
    x = x0[selected_x_indices]
    y = y0[selected_y_indices]

    # Select z_array according to x and y indices:
    z_array1 = z_array0[selected_y_indices, :]
    z_array = z_array1[:, selected_x_indices]

    df_trainingData_pivot = pd.DataFrame(data=z_array, index=y, columns=x)

    return df_trainingData_pivot

#################################################
# Training data to dataFrame flatten:


def trainingDataToDataFrameFlatten(x_array, y_array, z_array):
    """
    Gets: x_array, y_array, z_array, definitions.
    Returns: df_trainingData_flatten.
    Calling: None.
    Description:
    """

    flatten_column_name_x = data['flatten_columns_names']['x']
    flatten_column_name_y = data['flatten_columns_names']['y']
    flatten_column_name_z = data['flatten_columns_names']['z']

    # f is for flatten:
    df_trainingData_flatten = pd.DataFrame(
        np.array([x_array.flatten(),
                  y_array.flatten(),
                  z_array.flatten()]).T,
        columns=[flatten_column_name_x,
                 flatten_column_name_y,
                 flatten_column_name_z])

    return df_trainingData_flatten
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

    flatten_column_name_x = data['flatten_columns_names']['x']
    flatten_column_name_y = data['flatten_columns_names']['y']
    flatten_column_name_z = data['flatten_columns_names']['z']

    # f is for flatten:
    df_flatten = pd.DataFrame(
        np.array([x_array.flatten(),
                  y_array.flatten(),
                  z_array.flatten()]).T,
        columns=[flatten_column_name_x,
                 flatten_column_name_y,
                 flatten_column_name_z])

    return df_flatten

#################################################
# 1.3 Plot training data:


def plotTrainingData(df_pivot, submodelName):
    """
    Gets: df_pivot.
    Returns: None.
    Calling: None.
    Called by: main
    Description: Plotting a heatmap of the training data.
    """

    nRows = plots['nRoWs']

    DataToPlot = nRows*[None]
    DataToPlot[0] = [[df_pivot.columns,
                      df_pivot.index],
                     [df_pivot.values]]

    plotWhat = [True, False, False, False]

    plotting.plotData(DataToPlot, plotWhat, submodelName)

#################################################
