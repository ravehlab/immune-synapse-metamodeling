# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 21:50:23 2022

@author: yairn
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import definitions
import plotting

submodels = definitions.submodels
plots = definitions.plots
data = definitions.data

# 2.1 Set fit equation for dep:


# def linXlinY(xy, intercept, xSlope, ySlope):
#     """
#     Gets: xy, intercept, xSlope, ySlope.
#     Returns: f.
#     Calling: None.
#     Called by:
#     Description:
#     """

#     x, y = xy
#     f = intercept + xSlope*x + ySlope*y

#     return f


# def gaussXgaussY(xy, xScale, xMu, xSigma, yScale, yMu, ySigma):
#     """
#     Gets: xy, xScale, xMu, xSigma, yScale, yMu, ySigma.
#     Returns: f.
#     Calling: None.
#     Called by:
#     Description:
#     """

#     x, y = xy
#     fx = xScale*np.exp(-0.5*((x - xMu)/xSigma)**2)
#     fy = yScale*np.exp(-0.5*((y - yMu)/ySigma)**2)
#     f = fx + fy

#     return f


# def sigXlinearY(xy, xMin, xMax, xCen, xDev, ySlope):
#     """
#     Gets: xy, xScale, xMu, xSigma, yScale, yMu, ySigma.
#     Returns: f.
#     Calling: None.
#     Called by:
#     Description:
#     """

#     x, y = xy
#     fx = xMin + (xMax - xMin)*np.exp((x - xCen)/xDev)
#     fy = ySlope*y
#     f = fx + fy

#     return f

# %% poly22 #####################################
# def poly22(xy, p00, p10, p01, p20, p11, p02):
#     """
#     Gets: xy, fit_parameters.
#     Returns: f.
#     Calling: None.
#     Called by:
#     Description:
#     """
#     submodelName = 'PhosRatio'
#     # submodels[submodelName]['fitFunction']
#     x, y = xy
#     # f = p00 + p10*x + p01*y + p20*x**2 + p11*x*y + p02*y**2
#     f = eval(submodels[submodelName]['fitFunction'])
    
#     return f

# %% sigXsigY ###################################
def sigXsigY(xy, a, xScale, xCen, xDev, yScale, yCen, yDev):
    
    submodelName = 'PhosRatio'
    x, y = xy
    f = eval(submodels[submodelName]['fitFunction'])

    return f

fit_function = sigXsigY
#################################################
# 2.2 Set fit function for dep:


def setFitFunction(df_trainingData_flatten):
    """
    Gets: df_trainingData_model1.csv
    Returns: df_fitParameters_depletion.
    Calling: None.
    Called by: main.
    Description: Returns a dataFrame with index=parametersNames,
    columns=['mu', 'sd'], values=fitParameters.
    """
    submodelName = 'PhosRatio'
    # parametersNames_depletion = definitions.parametersNames_depletion

    # Read x, y, z data from dataFrame:
    flatten_x = df_trainingData_flatten[data['flatten_columns_names'][0]]
    flatten_y = df_trainingData_flatten[data['flatten_columns_names'][1]]
    flatten_z = df_trainingData_flatten[data['flatten_columns_names'][2]]

    parametersNames = submodels['PhosRatio']['fitParametersNames']

    df_fitParameters_dep = getFitParameters(
        X=(flatten_x, flatten_y),
        fitFunc=fit_function,
        fXdata=flatten_z,
        parametersNames=parametersNames,
        p0=submodels[submodelName]['p0'])

    return df_fitParameters_dep

#################################################
# 2.3 Get fit parameters:


def getFitParameters(X, fitFunc, fXdata, parametersNames, p0):
    """
    Gets: X, fitFunc, fXdata, parametersNames, p0.
    Returns: df_fit_parameters.
    Calling: None.
    Called by: parametersFitting.setFitData
    Description: Returns fit parameters and aranges them in DataFrame
    where the index (rows) are the fit parameters' names and the columns
    are 'mu' and 'sd'.
    """

    popt, pcov = curve_fit(fitFunc, X, fXdata, p0)
    mu = popt
    sd = np.sqrt(np.diag(pcov))

    data = {'mu': mu, 'sd': sd}
    index = parametersNames

    df_fitParameters = pd.DataFrame(data, index=index)

    return df_fitParameters

#################################################
# 2.4 Get fitted data:


def getFittedData(df_trainingData_flatten, df_fitParameters):
    """
    Gets: df_fitParameters, df_trainingData_flatten.
    Returns: df_fitted_data_pivot.
    Calling: None.
    Called by: Surrogate.main
    Description: Returns fitted data created by the fit parameters and the
    x, y data.
    """

    # Read fit parameters from df_fitParameters:
    a_fit = df_fitParameters.loc['a', 'mu']
    xScale_fit = df_fitParameters.loc['xScale', 'mu']
    xCen_fit = df_fitParameters.loc['xCen', 'mu']
    xDev_fit = df_fitParameters.loc['xDev', 'mu']
    yScale_fit = df_fitParameters.loc['yScale', 'mu']
    yCen_fit = df_fitParameters.loc['yCen', 'mu']
    yDev_fit = df_fitParameters.loc['yDev', 'mu']

    flatten_x = df_trainingData_flatten['Decaylength_nm']
    flatten_y = df_trainingData_flatten['Depletion_nm']

    fitted_data_flatten = fit_function(
        (flatten_x, flatten_y),
        a_fit, xScale_fit, xCen_fit, xDev_fit, yScale_fit, yCen_fit, yDev_fit)

    df_fitted_data_flatten = df_trainingData_flatten
    df_fitted_data_flatten['PhosRatio'] = fitted_data_flatten

    df_fitted_data_pivot =\
        df_fitted_data_flatten.pivot(index='Depletion_nm',
                                     columns='Decaylength_nm',
                                     values='PhosRatio')

    return df_fitted_data_pivot

#################################################
# Plot fitted data:


def plotFittedData(df_pivot, submodelName):
    """
    Gets: df_pivot.
    Returns: None.
    Calling: None.
    Called by: main.
    Description: Plotting a heatmap of the training data.
    """

    nRows = plots['nRows']

    DataToPlot = nRows*[None]
    DataToPlot[0] = [[df_pivot.columns,
                      df_pivot.index],
                     [df_pivot.values]]

    plotWhat = [True, False, False, False]

    plotting.plotData(DataToPlot, plotWhat, submodelName)
