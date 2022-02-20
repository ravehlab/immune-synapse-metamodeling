# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 17:08:06 2022

@author: yairn
"""

import numpy as np
import pandas as pd

# from Model1.Surrogate import styling
from Model1.Surrogate import plotting
from Model1.Surrogate import preprocessing

"""
Cropping and scaling raw data to nanometers, assign values and units
for x and y axes:
"""


def cropAndScaleRawData(dep_raw_data):
    nmToPixels = 10

    dep_training_data0_nm = nmToPixels*dep_raw_data

    size_k, size_t = np.shape(dep_raw_data)
    # x-axis for arrays (time in seconds):
    min_t = 0
    max_t = 100
    t0 = np.linspace(min_t, max_t, size_t)

    # y-axis for arrays (rigidity in kT*nm^2):
    min_k = 5
    max_k = 100
    k0 = np.linspace(min_k, max_k, size_k)

    # select start indices for t and k:
    t_start_index = 1
    k_start_index = 1

    selected_t_indices = np.arange(t_start_index, size_t, 1)
    # every second index:
    selected_k_indices = np.arange(k_start_index, size_k, 2)

    dep_training_data1_nm = dep_training_data0_nm.iloc[selected_k_indices, :]
    dep_training_data_nm = dep_training_data1_nm.iloc[:, selected_t_indices]

    t = t0[selected_t_indices]
    k = k0[selected_k_indices]

    [t_array, k_array] = np.meshgrid(t, k)

    if True:
        np.save("dep_training_data_nm.npy", dep_training_data_nm)

    return t_array, k_array, dep_training_data_nm
#################################################
# Training data to dataFrame:


def trainingDataToDataFrame(t_array, k_array, dep_training_data_nm):

    df_trainingData_model1 = pd.DataFrame(
        np.array([t_array.flatten(),
                  k_array.flatten(),
                  dep_training_data_nm.values.flatten()]).T,
        columns=['time_sec', 'k0_kTnm2', 'dep_nm'])

    df_trainingData_model1.to_csv('trainingData_model1.csv')

    df_dep_nm = df_trainingData_model1.pivot('k0_kTnm2', 'time_sec', 'dep_nm')

    return df_dep_nm
#################################################
# Calling the defs above:


def get_input_data():

    # 1. Read and arange data:
    # 1.1 Read raw training data for model1:
    dep_raw_data = pd.read_csv('model1/dep_raw_data.csv', header=None)

    # 1.2 Crop and scale raw data to nanometers,
    # assign values and units for x and y axes:
    t_array, k_array, dep_training_data_nm =\
        preprocessing.cropAndScaleRawData(dep_raw_data)

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
