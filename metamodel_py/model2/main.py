# -*- coding: utf-8 -*-

"""
Created on Tue Jan 25 15:23:32 2022

@author: yairn

1. Pre process of data.
    1.1 Read raw training data for model1.
    1.2 Crop and scale data, assign values and units for x and y axes.
    1.3 Arange training data in 'pandas' dataFrame.
        1.3.1 Plot training data.
2. Pre modeling (finding initial fit parameters).
    2.1 Define fit equations and parameters.
    2.2 Get fit parameters.
    2.3 Create fitted data.
        2.3.1 Plot fitted data.
3. Create tables for the model.
    3.1 Define class RV.
    3.2  Define class Model.
    3.3 Get untrained model info.
        3.3.1 Display untrained table
    3.4 Save outputs.
4. Surrogate model with pymc3.
    4.1 Set untrained pymc3 model.
        4.1.1 Display and save graphviz of untrained model:
    4.2 Set trained pymc3 model.
        4.2.1 Plot trace_plots of the trained model:
    4.3 Set trained pymc3 model
        4.3.1 Display and save graphviz of trained model:
    4.4 Create a map with trained pymc3 model.
5. Outputs.
    5.1
    5.2
    5.3
"""

# import numpy as np
import pandas as pd
import pymc3 as pm
# from IPython.display import display

# 0. Import model1 packages:
import model2.styling
import model2.plotting
import model2.preprocessing
import model2.parametersfitting
import model2.model_info
import model2.modeling
import model2.trainedmodeltomesh

#################################################
# 1. Read and arange data:
# 1.1 Read raw training data for model2:
decaylength_raw_data = pd.read_csv(
    'model2/decaylength_raw_data.csv', header=None)

# 1.2 Crop and scale raw data to nanometers,
# assign values and units for x and y axes:
Poff_array, dl_array, decaylength_training_data_nm =\
    model2.preprocessing.cropAndScaleRawData(decaylength_raw_data.values)

# 1.3 Arange training data in pandas dataFrame (df):
df_decaylength_nm = model2.preprocessing.trainingDataToDataFrame(
    Poff_array, dl_array, decaylength_training_data_nm)

# 1.3.1 Plot training data:
nRows = 4

DataToPlot = nRows*[None]
DataToPlot[0] = [[df_decaylength_nm.columns,
                  df_decaylength_nm.index],
                 [df_decaylength_nm.values]]
plotWhat = [True, False, False, False]

model2.plotting.plotData(DataToPlot, plotWhat)
#################################################
# 2. Pre modeling (finding initial fit parameters):
# 2.1 Define fit equations and parameters:
df_trainingData_model2 = pd.read_csv('trainingData_model2.csv')

# 2.2 Get fit parameters:
df_fitParameters_DecayLength = model2.parametersfitting.setFitFunction(
    df_trainingData_model2)

# 2.3 Create fitted data from fit parameters:
df_fitted_DecayLength = model2.parametersfitting.fittedData(
    df_fitParameters_DecayLength, df_trainingData_model2)

# 2.4 Plot fitted data:
DataToPlot[1] = [[df_fitted_DecayLength.columns,
                  df_fitted_DecayLength.index],
                 [df_fitted_DecayLength.values]]

plotWhat = [True, True, False, False]

model2.plotting.plotData(DataToPlot, plotWhat)
#################################################
# 3. Create table for the model:
# 3.1 Define class RV
RV = model2.model_info.RV

# 3.2 Define class Model:
Model = model2.model_info.Model

# 3.3 Get untrained info:
model2_DL_info = model2.model_info.model2DL(df_fitParameters_DecayLength)
df_model2_untrainedTable = model2_DL_info.get_dataframe()

# 3.4 Display untrained table:
df_model2_untrainedTable = df_model2_untrainedTable.set_index('ID')
print(df_model2_untrainedTable)

#################################################
# 4. Modeling with pymc3:
# 4.1
pm_model2 = model2.modeling.get_pm_model2_untrained(
     df_trainingData_model2, df_model2_untrainedTable)

gv2 = pm.model_to_graphviz(pm_model2)
gv2

with pm_model2:
    trace2 = pm.sample(2000, chains=4)

pm.traceplot(trace2)

trace2_summary = pm.summary(trace2)
