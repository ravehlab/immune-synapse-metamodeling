# -*- coding: utf-8 -*-#

"""
Created on Tue Jan 25 15:23:32 2022

@author: yairn

1. Pre process of data.
    1.1 Get training data.
    1.2 Plot training data.
2. Pre modeling (finding initial fit parameters).
    2.1 Define fit equations and parameters.
    2.2 Get fit parameters.
    2.3 Create fitted data.
    2.4 Plot fitted data.
3. Create tables for the model.
    3.1 Define 'RV' (Random Variable) class.
    3.2 Define 'Model' class.
    3.3 Get untrained model info.
    3.4 Create table with untrained model info.
4. Create surrogate model with pymc3.
    4.1 Create untrained pymc3 model.
    4.2 Create trained pymc3 model.
    4.3 Create table with trained model info.
5. Predictions based on the trained parameters:
    5.1 Run prediction:
    5.2 Plot prediction data:
"""

import pandas as pd
import pymc3 as pm
from IPython.display import display
import os
import shutil
import sys

# Run from Directory: /home/yair/Documents/Git/Metamodel_py/
# Run command: run Surrogate_Models/Model1/Code/main

#################################################
# Import Model packages:
import definitions
import preProcessing
import parametersFitting
import createModelInfo
import training
import predicting

paths = definitions.paths
submodels = definitions.submodels

# Create the directory 'Output' in '/Metamodel_py/Surrogate_Models/Model1/'
submodelName = 'Depletion'
Output_path = paths['Output']
Input_path = paths['Input']

# Remove all Output directory content:
try:
    shutil.rmtree(Output_path)
except:
    print('Error deleting directory')

# os.rmdir(Output_path)
os.mkdir(Output_path)
print("Directory 'Output' created in '% s'" % (paths['Model']))

plot_things = False  # True  # 
#################################################
# 1.1 Read trainingData from Input/:
df_trainingData_depletion_pivot_r =\
    pd.read_csv(paths['Input']+"df_trainingData_depletion_pivot.csv",
                index_col=0)

df_trainingData_depletion_flatten =\
    pd.read_csv(paths['Input']+"df_trainingData_depletion_flatten.csv")

if plot_things:
    # 1.2 Plot training data:
    preProcessing.plotTrainingData(
        df_trainingData_depletion_pivot_r, submodelName)

#################################################
# 2. Parameters Fitting (to be used as initial parameters
# for the untrained model):
# 2.1 Get fit parameters:
df_fitParameters_depletion = parametersFitting.setFitFunction(
    df_trainingData_depletion_flatten)

# save df_fitParameters_depletion to Output/:
df_fitParameters_depletion.to_pickle(
    Output_path+"df_fitParameters_depletion")

# 2.2 Create fitted data from fit parameters:
df_fittedData_depletion_pivot = parametersFitting.getFittedData(
    df_trainingData_depletion_flatten, df_fitParameters_depletion)

if plot_things:
    # 2.3 Plot fitted data:
    parametersFitting.plotFittedData(df_fittedData_depletion_pivot, submodelName)

#################################################
# 3. Create table for model info:
# 3.1 Define class RV (Random variable).
createModelInfo.RV

# 3.2 Define class Model:
createModelInfo.Model

# 3.3 Get untrained info:
model1_depletion_info = createModelInfo.model1_depletion_info(
    df_fitParameters_depletion)

# Get untrained table:
df_untrainedTable = createModelInfo.model1_depletion.get_dataframe()

# Untrained table with 'ID' as index:
df_untrainedTable_ID = df_untrainedTable.set_index('ID')

# save df_untrainedTable_ID to Output/:
df_untrainedTable_ID.to_pickle(
    Output_path+"df_model1_untrainedTable_ID")

# Copy file to Coupled_model' Input/
src = Output_path+'df_model1_untrainedTable_ID'
dst = paths['Metamodel']+'Coupled_Model/'\
    'Input/df_model1_untrainedTable_ID'
shutil.copyfile(src, dst)

# 3.4 Display untrained table:
display(df_untrainedTable_ID.style.set_properties(
    **{'text-align': 'left',
       'background-color': submodels[submodelName]['tableBackgroundColor'],
       'border': '1px black solid'}))

print(df_untrainedTable_ID)

# 3.5 Output (temp) save displayed table as figure.

#################################################
# 4. Training with pymc3:

# 4.1 df_untrainedTable
pm_model_untrained = training.get_pm_model1_untrained(
     df_trainingData_depletion_flatten, df_untrainedTable_ID)

if plot_things:
    gv_untrained = pm.model_to_graphviz(pm_model_untrained)
    
    gv_untrained_filename =\
        gv_untrained.render(filename='gv_untrained',
                            directory=Output_path)

with pm_model_untrained:
    trace = pm.sample(2000, chains=4)  # ,return_inferencedata=True)

trace_summary = pm.summary(trace)

print(trace_summary)

trace_summary.to_pickle(Output_path+"trace_summary")

trace_summary_r = pd.read_pickle(Output_path+"trace_summary")

mean_sd_r = trace_summary_r.loc[:, ['mean', 'sd']]

df_trainedTable_ID = df_untrainedTable_ID

DP = 'Distribution parameters'
for rv in mean_sd_r.index:
    df_trainedTable_ID.loc[rv, DP]['mu'] =\
        str(mean_sd_r.loc[rv]['mean'])

    df_trainedTable_ID.loc[rv]['sd'] =\
        str(mean_sd_r.loc[rv]['sd'])

# Display trained table:.
display(df_trainedTable_ID.style.set_properties(
    **{'text-align': 'left',
       'background-color': submodels[submodelName]['tableBackgroundColor'],
       'border': '1px black solid'}))

# 4.3 Set trained model:
pm_model1_trained = training.get_pm_model1_trained(
    df_trainedTable_ID)

pm.plot_trace(trace)

gv_trained = pm.model_to_graphviz(pm_model1_trained)
gv_trained_filename =\
    gv_trained.render(filename='gv_trained', directory=Output_path)

#################################################
# 5 Predictions based on the trained parameters:
# 5.1 Run prediction:
# """
run_prediction = False  # True
prediction = definitions.prediction

if run_prediction:
    df_prediction_mean, df_prediction_std =\
        predicting.predict(df_trainedTable_ID)

    df_prediction_mean.to_pickle(
        Output_path+"/"+prediction['saveName_mean'])
    df_prediction_std.to_pickle(
        Output_path+"/"+prediction['saveName_std'])

    df_prediction_mean_r = pd.read_pickle(
        Output_path+"/"+prediction['saveName_mean'])
    df_prediction_std_r = pd.read_pickle(
        Output_path+"/"+prediction['saveName_std'])

    if plot_things:
        # 5.2 Plot prediction data:
        predicting.plotPredictionData(df_prediction_mean_r,
                                      definitions, submodelName)
# """
#################################################
