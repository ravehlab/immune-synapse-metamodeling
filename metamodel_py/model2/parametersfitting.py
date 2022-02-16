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

    x, y = xy
    f = intercept + xSlope*x + ySlope*y

    return f
#################################################
# Set fit function for dep:


def setFitFunction(df_trainingData):

    # Read x, y, z data from dataFrame:
    flatten_x = df_trainingData['Poff']
    flatten_y = df_trainingData['Diff']
    flatten_z = df_trainingData['DecayLength_nm']

    p0_dep = 0., 0., 0.
    parametersNames = ['intercept', 'xSlope', 'ySlope']

    df_fitParameters = get_fit_parameters(
        X=(flatten_x, flatten_y),
        fitFunc=linxliny,
        fXdata=flatten_z,
        parametersNames=parametersNames,
        p0=p0_dep)

    return df_fitParameters

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


def fittedData(df_fitParameters, df_trainingData_model2):

    intercept_fit = df_fitParameters.loc['intercept', 'mu']
    xSlope_fit = df_fitParameters.loc['xSlope', 'mu']
    ySlope_fit = df_fitParameters.loc['ySlope', 'mu']

    flatten_fitted_data =\
        intercept_fit +\
        xSlope_fit*df_trainingData_model2['Poff'] +\
        ySlope_fit*df_trainingData_model2['Diff']

    df_flatten_fitted_data = df_trainingData_model2
    df_flatten_fitted_data['DecayLength_nm'] = flatten_fitted_data

    df_fitted_data_array =\
        df_flatten_fitted_data.pivot(index='Poff',
                                     columns='Diff',
                                     values='DecayLength_nm')

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
