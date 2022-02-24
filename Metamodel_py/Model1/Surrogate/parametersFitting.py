# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 21:50:23 2022

@author: yairn
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from Model1.Surrogate import definitions
from Model1.Surrogate import plotting

# Set fit equation for dep:


def linXlinY(xy, intercept, xSlope, ySlope):
    """
    Gets: xy, intercept, xSlope, ySlope.
    Returns: f.
    Calling: None.
    Called by:
    Description:
    """

    x, y = xy
    parametersNames = ['intercept', 'xSlope', 'ySlope']
    f = intercept + xSlope*x + ySlope*y

    return f, parametersNames
#################################################
# Set fit function for dep:


def setFitFunction(df_trainingData_flatten):
    """
    Gets: df_trainingData_model1.
    Returns: df_fitParameters_dep.
    Calling: None.
    Called by: main.
    Description:
    """

    parametersNames_dep = definitions.parametersNames_dep

    # Read x, y, z data from dataFrame:
    flatten_x = df_trainingData_flatten['time_sec']
    flatten_y = df_trainingData_flatten['k0_kTnm2']
    flatten_z = df_trainingData_flatten['dep_nm']

    X = (flatten_x, flatten_y)
    print(X)

    p0_dep = 100., 0., 0.

    df_fitParameters_dep = getFitParameters(
        X=(flatten_x, flatten_y),
        fitFunc=linXlinY,
        fXdata=flatten_z,
        parametersNames=parametersNames_dep,
        p0=p0_dep)

    return df_fitParameters_dep

#################################################
# Get fit parameters:


def getFitParameters(X, fitFunc, fXdata, parametersNames, p0):
    """
    Gets: X, fitFunc, fXdata, parametersNames, p0.
    Returns: df_fit_parameters.
    Calling: None.
    Called by:
    Description: Returns fit parameters and aranges them in DataFrame
    where the index (rows) are the fit parameters' names and the columns
    are 'mu' and 'sd'.
    """

    popt, pcov = curve_fit(fitFunc, X, fXdata, parametersNames, p0)
    mu = popt
    sd = np.sqrt(np.diag(pcov))

    data = {'mu': mu, 'sd': sd}
    index = parametersNames

    df_fit_parameters = pd.DataFrame(data, index=index)

    return df_fit_parameters
#################################################
# Fitted data:


def fittedData(df_fitParameters, df_trainingData_flatten):
    """
    Gets: df_fitParameters, df_trainingData_flatten.
    Returns: df_fitted_data_pivot.
    Calling: None.
    Called by:
    Description:
    """

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


# def getParametersFitting():
#     """
#     Gets:
#     Returns: df_fitted_data_pivot.
#     Calling: None.
#     Description: Pre modeling (finding initial fit parameters).
#     """

#     # 2.1 Define fit equations and parameters:
#     df_trainingData_model1 = pd.read_csv('trainingData_model1.csv')

#     # 2.2 Get fit parameters:
#     df_fitParameters_dep = setFitFunction(df_trainingData_model1)

#     # 2.3 Create fitted data from fit parameters:
#     df_fitted_data_pivot = fittedData(df_fitParameters_dep,
#                                       df_trainingData_model1)

def plotFittedData(df_pivot):
    """
    Gets: df_pivot.
    Returns: None.
    Calling: None.
    Called by: main.
    Description: Plotting a heatmap of the training data.
    """

    nRows = definitions.nRows

    DataToPlot = nRows*[None]
    DataToPlot[0] = [[df_pivot.columns,
                      df_pivot.index],
                     [df_pivot.values]]

    plotWhat = [False, True, False, False]

    plotting.plotData(DataToPlot, plotWhat)
