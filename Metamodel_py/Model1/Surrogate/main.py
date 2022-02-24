# -*- coding: utf-8 -*-#

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
    3.1 Define 'Random Variable' (RV) class.
    3.2 Define 'Model' class.
    3.3 Get untrained model info.
    3.4 Create table with untrained model info.
4. Create surrogate model with pymc3.
    4.1 Create untrained pymc3 model.
    4.2 Create trained pymc3 model.
    4.3 Create a fine mesh surrogate model based on the trained parameters.
5. Outputs.
    5.1
    5.2
    5.3
"""

import numpy as np
import pandas as pd
import pymc3 as pm
# from IPython.display import display

# Import Model1 packages:
from Model1.Surrogate import definitions
from Model1.Surrogate import plotting
from Model1.Surrogate import preProcessing
from Model1.Surrogate import parametersFitting
from Model1.Surrogate import createModelInfo
from Model1.Surrogate import training
from Model1.Surrogate import predicting

Metamodel_path = '/home/yair/Documents/Git/Metamodel_py/'
Model_path = Metamodel_path+'Model1/'
Input_path = Model_path+'Input/'
Output_path = Model_path+'Output/'
Input_data_name = 'df_trainingData_model1_pivot.csv'

#################################################
# 1. Get training data:

# 1.1 Read trainingData from Input/ folder:
df_trainingData_dep_pivot = pd.read_pickle(Input_path+Input_data_name)

# Get trainingDta aranged as dataFrame in columns (flatten):
df_trainingData_dep_flatten = preProcessing.pivotToFlatten(
    df_trainingData_dep_pivot)

save_name_pivot = 'df_trainingData_dep_pivot.csv'
save_name_flatten = 'df_trainingData_dep_flatten.csv'

# Save training data as pivot array:
df_trainingData_dep_pivot.to_pickle(
    Model_path+'/Processing/'+save_name_pivot)

# Save training data as flatten array:
df_trainingData_dep_flatten.to_pickle(
    Model_path+'/Processing/'+save_name_flatten)

# 1.3 Plot training data:
preProcessing.plotTrainingData(
    df_trainingData_dep_pivot, definitions)
#################################################

# 2. Parameters Fitting (to be used as initial parameters
# for the untrained model):

# Read df_trainingData_flatten:
df_trainingData_dep_flatten_r = pd.read_pickle(
    Model_path+'/Processing/'+save_name_flatten)

# 2.1 Define fit equations and parameters:




# 2.2 Get fit parameters:
df_fitParameters_dep = parametersFitting.setFitFunction(
    df_trainingData_dep_flatten_r)




# 2.3 Create fitted data from fit parameters:
df_fitted_dep = parametersFitting.fittedData(
    df_trainingData_dep_flatten_r, df_fitParameters_dep)

# 2.4 Plot fitted data:
preProcessing.plotFittedData(
    df_fittedData_dep_pivot, definitions)
#################################################
# 2. Get parameters fitting:






#################################################
# 3. Create table for model info:
# 3.1 Define class RV
createModelInfo.RV

# 3.2 Define class Model:
createModelInfo.Model

# 3.3 Get untrained info:
model1_dep_info = createModelInfo.model1_info(df_fitParameters_dep)
df_model1_untrainedTable = model1_dep_info.get_dataframe()

# 3.4 Display untrained table:
df_model1_untrainedTable = df_model1_untrainedTable.set_index('ID')
# model1.model_info.displayInfo(df_model1_untrainedTable)
print(df_model1_untrainedTable)

# 3.5 Output (temp)










#################################################
# 4. Modeling with pymc3:

# 4.1 df_model1_untrainedTabled
pm_model1_untrained = modeling.get_pm_model1_untrained(
     df_trainingData_model1, df_model1_untrainedTable)

gv1_untrained = pm.model_to_graphviz(pm_model1_untrained)
gv1_untrained_filename =\
    gv1_untrained.render(filename='gv1_untrained',
                         directory=Model1_path)

with pm_model1_untrained:
    trace1 = pm.sample(2000, chains=4)

pm.traceplot(trace1)

trace1_summary = pm.summary(trace1)

trace1_summary.to_pickle("trace1_summay")
# trace1_summary.to_pickle(metamodel_directoty+'/model1', "trace1_summay")
trace1_summary_r = pd.read_pickle("trace1_summay")

mean_sd_r = trace1_summary_r.loc[:, ['mean', 'sd']]

df_model1_trainedTable = df_model1_untrainedTable

DP = 'Distribution parameters'
for rv in mean_sd_r.index:
    df_model1_trainedTable.loc[rv, DP]['mu'] =\
        str(mean_sd_r.loc[rv]['mean'])
    df_model1_trainedTable.loc[rv]['sd'] =\
        str(mean_sd_r.loc[rv]['sd'])

# 4.3 Set trained model:
pm_model1_trained = training.get_pm_model1_trained(
    df_model1_trainedTable)

gv1_trained = pm.model_to_graphviz(pm_model1_trained)
gv1_trained_filename =\
    gv1_trained.render(filename='gv1_trained',
                       directory=Output_path)

#################################################
# 5 Predictions based on the trained parameters:

# 5.1 Run prediction:
run_prediction = False

if run_prediction:
    df_prediction_mean, df_prediction_std =\
        predicting.predict(df_model1_trainedTable)

    df_prediction_mean.to_pickle(Output_path+"/df_model1_predicted_dep_mean")
    df_prediction_std.to_pickle(Output_path+"/df_model1_predicted_dep_std")

df_prediction_mean_r = pd.read_pickle(
    Output_path+"/df_model1_predicted_dep_mean")
df_prediction_std_r = pd.read_pickle(
    Output_path+"/df_model1_predicted_dep_std")

# 5.2 Plot prediction data:
predicting.plotPredictionData(df_prediction_mean_r,
                              df_prediction_std_r,
                              definitions)
