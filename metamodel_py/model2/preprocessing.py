# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 17:08:06 2022

@author: yairn
"""

import numpy as np
import pandas as pd

# import model2.styling

"""
Cropping and scaling raw data to nanometers, assign values and units
for x and y axes:
"""


def cropAndScaleRawData(raw_data):

    # Cropping raw data:
    lambdaaLCK_array = raw_data[:-3, :-1]

    # Scaling cropped data:
    DLaLCK_array = -1./lambdaaLCK_array

    logDLaLCK_array = np.log10(DLaLCK_array)

    logDiff0 = np.arange(-3, 0.25, 0.25)
    logPoff0 = np.arange(-5, 0.0, 0.5)

    logDiff = logDiff0[:-3]
    logPoff = logPoff0[:]

    logPoff_array, logDiff_array = np.meshgrid(logPoff, logDiff)

    # return logPoff_array, logDiff_array, DLaLCK_array
    return logPoff_array, logDiff_array, logDLaLCK_array

    """
    size_y, size_x = np.shape(raw_data)

    scale = 1
    training_data0 = raw_data*scale

    # x-axis for arrays:
    min_x = 0
    max_x = 100
    x0 = np.linspace(min_x, max_x, size_x)

    # y-axis for arrays:
    min_y = 5
    max_y = 100
    y0 = np.linspace(min_y, max_y, size_y)

    # select start indices for t and k:
    x_start_index = 1
    y_start_index = 1

    selected_x_indices = np.arange(x_start_index, size_x, 1)
    # every second index:
    selected_y_indices = np.arange(y_start_index, size_y, 2)

    training_data1 = training_data0.iloc[selected_y_indices, :]
    training_data = training_data1.iloc[:, selected_x_indices]

    x = x0[selected_x_indices]
    y = y0[selected_y_indices]

    [x_array, y_array] = np.meshgrid(x, y)

    if True:
        np.save("training_data_model2.npy", training_data)

    return x_array, y_array, training_data
    """
#################################################
# Training data to dataFrame:


def trainingDataToDataFrame(x_array, y_array, z_array):

    df_trainingData = pd.DataFrame(
        np.array([x_array.flatten(),
                  y_array.flatten(),
                  z_array.flatten()]).T,
        columns=['Poff', 'Diff', 'DecayLength_nm'])

    df_trainingData.to_csv('trainingData_model2.csv')

    df_trainingData_p = df_trainingData.pivot(
         'Diff', 'Poff', 'DecayLength_nm')

    return df_trainingData_p
#################################################
