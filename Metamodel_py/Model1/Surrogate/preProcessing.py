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

#################################################
# Training data to dataFrame:


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

    nRows = definitions.nRows

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
