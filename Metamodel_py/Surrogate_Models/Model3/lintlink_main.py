# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 15:23:32 2022

@author: yairn

1. Pre process of data.
    1.1 Read raw training data for model1.
    1.2 Crop and scale data, assign values and units for x and y axes.
    1.3 Arange training data in 'pandas' dataFrame.
    1.4 Plot training data.
2. Pre modeling (finding initial fit parameters).
    2.1 Define fit equations and parameters.
    2.2 Get fit parameters.
    2.3 Create fitted data.
    2.4 Plot fitted data.
3. Create tables for the model.
    3.1
    3.2
    3.3
    3.4
    3.5
4. Create surrogate model with pymc3.
    4.1 Create untrained pymc3 model.
    4.2 Create trained pymc3 model.
    4.3 Create a fine mesh surrogate model based on the trained parameters.
5. Outputs.
    5.1
    5.2
    5.3
"""

# import numpy as np
import pandas as pd
# import pymc3 as pm
# from IPython.display import display

# 0. Import model1 packages:
import model1.styling
import model1.plotting
import model1.preprocessing
import model1.parametersfitting
import model1.model_info
import model1.modeling

#################################################
# 1. Read and arange data:
# 1.1 Read raw training data for model1:
dep_raw_data = pd.read_csv('model1/dep_raw_data.csv', header=None)

# 1.2 Crop and scale raw data to nanometers,
# assign values and units for x and y axes:
t_array, k_array, dep_training_data_nm =\
    model1.preprocessing.cropAndScaleRawData(dep_raw_data)

# 1.3 Arange training data in pandas dataFrame (df):
df_dep_nm = model1.preprocessing.trainingDataToDataFrame(
    t_array, k_array, dep_training_data_nm)

# 1.4 Plot training data:
nRows = 4

DataToPlot = nRows*[None]
DataToPlot[0] = [[df_dep_nm.columns,
                  df_dep_nm.index],
                 [df_dep_nm.values]]
plotWhat = [True, False, False, False]

model1.plotting.plotData(DataToPlot, plotWhat)
#################################################
# 2. Pre modeling (finding initial fit parameters):
# 2.1 Define fit equations and parameters:
df_trainingData_model1 = pd.read_csv('trainingData_model1.csv')

# 2.2 Get fit parameters:
df_fitParameters_dep = model1.parametersfitting.setFitFunction(
    df_trainingData_model1)

# 2.3 Create fitted data from fit parameters:
df_fitted_dep = model1.parametersfitting.fittedData(
    df_fitParameters_dep, df_trainingData_model1)

# 2.4 Plot fitted data:
DataToPlot[1] = [[df_fitted_dep.columns,
                  df_fitted_dep.index],
                 [df_fitted_dep.values]]
plotWhat = [True, True, False, False]

model1.plotting.plotData(DataToPlot, plotWhat)
#################################################
# 3. Create table for the model:
# 3.1 Define class RV
RV = model1.model_info.RV

# 3.2 Define class Model:
Model = model1.model_info.Model

# 3.3 Get untrained info:
model1_dep_info = model1.model_info.model1Dep(df_fitParameters_dep)

# 3.4 Display untrained table:
model1.model_info.displayInfo(model1_dep_info)
#################################################
# 4. Modeling with pymc3:
# 4.1
pm_model1 = model1.modeling.get_pm_model1_untrained(
     df_trainingData_model1, model1_dep_info)

# gv1 = pm.model_to_graphviz(pm_model1)
# gv1
