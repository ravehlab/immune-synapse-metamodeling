# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 21:50:23 2022

@author: yairn
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# Set fit equation for dep:


def linxliny(tk, intercept, tSlope, kSlope):

    t, k = tk
    f = intercept + tSlope*t + kSlope*k

    return f
#################################################
# Set fit function for dep:


def setFitFunction(df_trainingData_model1):

    # Read x, y, z data from dataFrame:
    flatten_t = df_trainingData_model1['time_sec']
    flatten_k = df_trainingData_model1['k0_kTnm2']
    flatten_dep_nm = df_trainingData_model1['dep_nm']

    p0_dep = 0., 0., 0.
    parametersNames_dep = ['intercept', 'tSlope', 'kSlope']

    df_fitParameters_dep = get_fit_parameters(
        X=(flatten_t, flatten_k),
        fitFunc=linxliny,
        fXdata=flatten_dep_nm,
        parametersNames=parametersNames_dep,
        p0=p0_dep)

    return df_fitParameters_dep

#################################################
# Get fit parameters:


def get_fit_parameters(X, fitFunc, fXdata, parametersNames, p0):
    """
    Returns fit parameters and aranges them in DataFrames where the index
    (rows) are the fit parameters' names and the columns are 'mu' and 'sd'.
    """
    popt, pcov = curve_fit(fitFunc, X, fXdata, p0)
    mu = popt
    sd = np.sqrt(np.diag(pcov))

    data = {'mu': mu, 'sd': sd}
    index = parametersNames

    df_fit_parameters = pd.DataFrame(data, index=index)

    return df_fit_parameters
#################################################
# Fitted data:


def fittedData(df_fitParameters_dep, df_trainingData_model1):

    intercept_fit = df_fitParameters_dep.loc['intercept', 'mu']
    tSlope_fit = df_fitParameters_dep.loc['tSlope', 'mu']
    kSlope_fit = df_fitParameters_dep.loc['kSlope', 'mu']

    flatten_fitted_data =\
        intercept_fit +\
        tSlope_fit*df_trainingData_model1['time_sec'] +\
        kSlope_fit*df_trainingData_model1['k0_kTnm2']

    df_flatten_fitted_data = df_trainingData_model1
    df_flatten_fitted_data['dep_nm'] = flatten_fitted_data

    df_fitted_data_array = df_flatten_fitted_data.pivot(index='k0_kTnm2',
                                                        columns='time_sec',
                                                        values='dep_nm')

    return df_fitted_data_array
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


def get_parameters_fitting():
    pass
