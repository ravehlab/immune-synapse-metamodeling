# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 21:50:23 2022

@author: yairn
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# from Surrogate_Models.Model1.Code import definitions
# from Surrogate_Models.Model1.Code import plotting
import definitions
import plotting

submodels = definitions.submodels
plots = definitions.plots
data = definitions.data

# 2.1 Set fit equation for dep:


def sigXsigY(xy, xScale, xCen, xDev, yScale, yCen, yDev):
    x, y = xy
    fx = xScale/(1 + np.exp(-(x-xCen)/xDev))
    fy = yScale/(1 + np.exp(-(y-yCen)/yDev))
    f = fx + fy

    return f

def poly21(xy, p00, p10, p01, p20, p11):
    
    submodelName = 'Depletion'
    x, y = xy
    f = eval(submodels[submodelName]['fitFunction'])

    return f

fit_function = poly21
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

    # parametersNames_depletion = definitions.parametersNames_depletion

    # Read x, y, z data from dataFrame:
    flatten_x = df_trainingData_flatten[data['flatten_columns_names'][0]]
    flatten_y = df_trainingData_flatten[data['flatten_columns_names'][1]]
    flatten_z = df_trainingData_flatten[data['flatten_columns_names'][2]]

    parametersNames = submodels['Depletion']['fitParametersNames']

    df_fitParameters_dep = getFitParameters(
        X=(flatten_x, flatten_y),
        fitFunc=fit_function,
        fXdata=flatten_z,
        parametersNames=parametersNames,
        p0=submodels['Depletion']['p0'])

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

    try:
        popt, pcov = curve_fit(fitFunc, X, fXdata, p0)
        mu = popt
        sd = np.sqrt(np.diag(pcov))

    except RuntimeError:
        print("Error - curve_fit failed")

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
    submodelName = 'Depletion'

    # Read fit parameters from df_fitParameters:
    # xScale_fit = df_fitParameters.loc[
    #     submodels[submodelName]['fitParametersNames'][0], 'mu']
    # xCen_fit = df_fitParameters.loc[
    #     submodels[submodelName]['fitParametersNames'][1], 'mu']
    # xDev_fit = df_fitParameters.loc[
    #     submodels[submodelName]['fitParametersNames'][2], 'mu']
    # yScale_fit = df_fitParameters.loc[
    #     submodels[submodelName]['fitParametersNames'][3], 'mu']
    # yCen_fit = df_fitParameters.loc[
    #     submodels[submodelName]['fitParametersNames'][4], 'mu']
    # yDev_fit = df_fitParameters.loc[
    #     submodels[submodelName]['fitParametersNames'][5], 'mu']

    flatten_column_name_x = data['flatten_columns_names'][0]
    flatten_column_name_y = data['flatten_columns_names'][1]
    # flatten_column_name_z = data['flatten_columns_names'][2]

    flatten_x = df_trainingData_flatten[flatten_column_name_x]
    flatten_y = df_trainingData_flatten[flatten_column_name_y]

    ###
    submodelName = 'Depletion'
    # submodels[submodelName]['equation']
    ###


    # Read fit parameters from df_fitParameters:
    p00_fit = df_fitParameters.loc['p00', 'mu']
    p10_fit = df_fitParameters.loc['p10', 'mu']
    p01_fit = df_fitParameters.loc['p01', 'mu']
    p20_fit = df_fitParameters.loc['p20', 'mu']
    p11_fit = df_fitParameters.loc['p11', 'mu']

    # flatten_x = df_trainingData_flatten['Poff']
    # flatten_y = df_trainingData_flatten['Diff']

    fitted_data_flatten = fit_function(
        (flatten_x, flatten_y),
        p00_fit, p10_fit, p01_fit, p20_fit, p11_fit)

    df_fitted_data_flatten = df_trainingData_flatten
    df_fitted_data_flatten[
        data['flatten_columns_names'][2]] = fitted_data_flatten

    df_fitted_data_pivot = df_fitted_data_flatten.pivot(
        index=data['flatten_columns_names'][1],
        columns=data['flatten_columns_names'][0],
        values=data['flatten_columns_names'][2])

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

    nRows = plots['nRoWs']

    DataToPlot = nRows*[None]
    DataToPlot[0] = [[df_pivot.columns,
                      df_pivot.index],
                     [df_pivot.values]]

    plotWhat = [True, False, False, False]

    plotting.plotData(DataToPlot, plotWhat, submodelName)
