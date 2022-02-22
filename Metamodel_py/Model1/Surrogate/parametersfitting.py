# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 21:50:23 2022

@author: yairn
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# Set fit equation for dep:


def linxliny(xy, intercept, xSlope, ySlope):
    """
    Gets: xy, intercept, xSlope, ySlope.
    Returns: f.
    Calling: None.
    Description:
    """

    x, y = xy
    parametersNames = ['intercept', 'xSlope', 'ySlope']
    f = intercept + xSlope*x + ySlope*y

    return f, parametersNames
#################################################
# Set fit function for dep:


def setFitFunction(df_trainingData_flatten, parametersNames):
    """
    Gets: df_trainingData_model1.
    Returns: df_fitParameters_dep.
    Calling: None.
    Description:
    """

    # Read x, y, z data from dataFrame:
    flatten_x = df_trainingData_flatten['time_sec']
    flatten_y = df_trainingData_flatten['k0_kTnm2']
    flatten_z = df_trainingData_flatten['dep_nm']

    p0_dep = 0., 0., 0.

    df_fitParameters_dep = get_fit_parameters(
        X=(flatten_x, flatten_y),
        fitFunc=linxliny,
        fXdata=flatten_z,
        parametersNames=parametersNames,
        p0=p0_dep)

    return df_fitParameters_dep

#################################################
# Get fit parameters:


def get_fit_parameters(X, fitFunc, fXdata, parametersNames, p0):
    """
    Gets: X, fitFunc, fXdata, parametersNames, p0.
    Returns: df_fit_parameters.
    Calling: None.
    Description: Returns fit parameters and aranges them in DataFrame
    where the index (rows) are the fit parameters' names and the columns
    are 'mu' and 'sd'.
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


def fittedData(df_fitParameters, df_trainingData_flatten):

    
    intercept_fit = df_fitParameters.loc['intercept', 'mu']
    xSlope_fit = df_fitParameters.loc['xSlope', 'mu']
    ySlope_fit = df_fitParameters.loc['ySlope', 'mu']

    fitted_data_flatten =\
        intercept_fit +\
        xSlope_fit*df_trainingData_flatten['time_sec'] +\
        ySlope_fit*df_trainingData_flatten['k0_kTnm2']

    df_fitted_data_flatten = df_trainingData_flatten
    df_fitted_data_flatten['dep_nm'] = fitted_data_flatten

    df_fitted_data_pivot = df_fitted_data_flatten.pivot(index='k0_kTnm2',
                                                        columns='time_sec',
                                                        values='dep_nm')

    return df_fitted_data_pivot
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
