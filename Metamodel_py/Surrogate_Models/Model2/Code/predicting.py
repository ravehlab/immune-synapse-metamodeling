#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 18:05:56 2022

@author: yair
"""

import numpy as np
import pandas as pd
import pymc3 as pm

import definitions
import training
import plotting

plots = definitions.plots
# Create a heatmap by running the trained model with a batch of x, y value:


def predict(df_model1_trainedTable):
    """
    Gets: df_model1_trainedTable.
    Returns: df_prediction_mean, df_prediction_std.
    Calling: training.get_pm_model1_trained.
    Called by: main
    Description: Calculating predictions based on
    the trained model parameters.
    """

    Xs = definitions.prediction['Xs']  # x values.
    Ys = definitions.prediction['Ys']  # y values.

    prediction_mean = np.zeros((len(Ys), len(Xs)))
    prediction_std = np.zeros((len(Ys), len(Xs)))

    for i, y in enumerate(Ys):
        for j, x in enumerate(Xs):

            current_model = training.get_pm_model2_trained(
                df_model1_trainedTable, observed_Diff=y, observed_Poff=x)

            with current_model:
                current_trace = pm.sample(2000, chains=4, progressbar=False)

            print(f"i,x={i, x}, j,y={j, y}")

            prediction_mean[i, j] =\
                current_trace.rv_output_Decaylength_LCKA2.mean()
            prediction_std[i, j] =\
                current_trace.rv_output_Decaylength_LCKA2.std()
            
            print(f"mean={prediction_mean[i, j]}, std={prediction_std[i, j]}")
            
    df_prediction_mean = pd.DataFrame(data=prediction_mean,
                                      index=Ys,
                                      columns=Xs,)

    df_prediction_std = pd.DataFrame(data=prediction_std,
                                     index=Ys,
                                     columns=Xs,)

    return df_prediction_mean, df_prediction_std

#################################################
# Plot prediction:


def plotPredictionData(df_pivot, definitions, submodelName):
    """
    Gets: df_pivot.
    Returns: None.
    Calling: plotData.
    Called by: main
    Description: Plotting a heatmap of the training data.
    """

    nRows = plots['nRoWs']

    DataToPlot = nRows*[None]
    DataToPlot[0] = [[df_pivot.columns,
                      df_pivot.index],
                     [df_pivot.values]]

    plotWhat = [False, False, False, True]

    plotting.plotData(DataToPlot, plotWhat, submodelName)
    
#################################################
