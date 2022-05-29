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


# def sigXlinearY(xy, xMin, xMax, xCen, xDev, ySlope):
#     """
#     Gets: xy, xScale, xMu, xSigma, yScale, yMu, ySigma.
#     Returns: f.
#     Calling: None.
#     Called by:
#     Description:
#     """

#     x, y = xy
#     fx = xMin + (xMax - xMin)/np.exp((x - xCen)/xDev)
#     fy = ySlope*y
#     f = fx + fy

#     return f


# def poly21(xy, p00, p10, p01, p20, p11):
    
#     submodelName = 'RgRatio'
#     x, y = xy
#     f = eval(submodels[submodelName]['fitFunction'])

#     return f

# fit_function = poly21

def sigXsigY(xy, a, xScale, xCen, xDev, yScale, yCen, yDev):
    
    submodelName = 'PhosRatio'
    x, y = xy
    f = eval(submodels[submodelName]['fitFunction'])

    return f

fit_function = sigXsigY

#################################################
# 2.2 Set fit function:
    
    
def setFitFunction(df_trainingData_flatten):
    """
    Gets: df_trainingData_model1.csv
    Returns: df_fitParameters_depletion.
    Calling: None.
    Called by: main.
    Description: Returns a dataFrame with index=parametersNames,
    columns=['mu', 'sd'], values=fitParameters.
    """

    # Read x, y, z data from dataFrame:
    flatten_x = df_trainingData_flatten[data['flatten_columns_names'][0]]
    flatten_y = df_trainingData_flatten[data['flatten_columns_names'][1]]
    flatten_z = df_trainingData_flatten[data['flatten_columns_names'][3]]

    parametersNames = submodels['RgRatio']['fitParametersNames']

    df_fitParameters = getFitParameters(
        X=(flatten_x, flatten_y),
        fitFunc=fit_function,
        fXdata=flatten_z,
        parametersNames=parametersNames,
        p0=submodels['RgRatio']['p0'])

    return df_fitParameters

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

    flatten_column_name_x = data['flatten_columns_names'][0]
    flatten_column_name_y = data['flatten_columns_names'][1]
    # flatten_column_name_z = data['flatten_columns_names'][2]

    flatten_x = df_trainingData_flatten[flatten_column_name_x]
    flatten_y = df_trainingData_flatten[flatten_column_name_y]

    # Read fit parameters from df_fitParameters:
    p00_fit = df_fitParameters.loc['p00', 'mu']
    p10_fit = df_fitParameters.loc['p10', 'mu']
    p01_fit = df_fitParameters.loc['p01', 'mu']
    p20_fit = df_fitParameters.loc['p20', 'mu']
    p11_fit = df_fitParameters.loc['p11', 'mu']

    fitted_data_flatten = fit_function(
        (flatten_x, flatten_y),
        p00_fit, p10_fit, p01_fit, p20_fit, p11_fit)

    df_fitted_data_flatten = df_trainingData_flatten
    df_fitted_data_flatten['RgRatio'] = fitted_data_flatten

    df_fitted_data_pivot = df_fitted_data_flatten.pivot(
        index='Depletion_nm',
        columns='Decaylength_nm',
        values='RgRatio')

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
