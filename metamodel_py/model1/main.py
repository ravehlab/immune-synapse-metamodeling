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
6. git?
    First right click the tab corresponding to any file in your repository
    and click "set console working directory." Then go to the Ipython window
    in Spyder and simply type your git commands (assuming Git is installed
    and its paths are configured properly) but append a "!" to the beginning
    of your command:

    !git add "file.py"
    !git commit -m "My commit"
    !git push origin master
"""

import numpy as np
import pandas as pd
import pymc3 as pm
# from IPython.display import display

# 0. Import model1 packages:
import model1.styling
import model1.plotting
import model1.preprocessing
import model1.parametersfitting
import model1.model_info
import model1.modeling
import model1.trainedmodeltomesh

metamodel_directoty = '/home/yair/Documents/Git/metamodel_py'
model1_path = '/home/yair/Documents/Git/metamodel_py/model1'

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
# 3. Create table for model info:
# 3.1 Define class RV
model1.model_info.RV

# 3.2 Define class Model:
model1.model_info.Model

# 3.3 Get untrained info:
model1_dep_info = model1.model_info.model1_info(df_fitParameters_dep)
df_model1_untrainedTable = model1_dep_info.get_dataframe()

# 3.4 Display untrained table:
df_model1_untrainedTable = df_model1_untrainedTable.set_index('ID')
# model1.model_info.displayInfo(df_model1_untrainedTable)
print(df_model1_untrainedTable)

# 3.5 Output (temp)

#################################################
# 4. Modeling with pymc3:
# 4.1 df_model1_untrainedTabled
pm_model1_untrained = model1.modeling.get_pm_model1_untrained(
     df_trainingData_model1, df_model1_untrainedTable)

gv1_untrained = pm.model_to_graphviz(pm_model1_untrained)
gv1_untrained_filename =\
    gv1_untrained.render(filename='gv1_untrained',
                         directory=metamodel_directoty+'/model1')

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
pm_model1_trained = model1.modeling.get_pm_model1_trained(
    df_model1_trainedTable)

gv1_trained = pm.model_to_graphviz(pm_model1_trained)
gv1_trained_filename =\
    gv1_trained.render(filename='gv1_trained',
                       directory=metamodel_directoty+'/model1')

# 4.4 Trained_mesh:
n_t = 21
max_t = 100.
min_t = 0.
Ts = np.linspace(min_t, max_t, n_t)

n_k = 20
max_k = 100.
min_k = max_k/n_k
Ks = np.linspace(min_k, max_k, n_k)

if True:
    deps_mean, deps_std =\
        model1.trainedmodeltomesh.trained_mesh(min_t, max_t, n_t,
                                               min_k, max_k, n_k,
                                               df_model1_trainedTable)

    df_deps_mean = pd.DataFrame(data=deps_mean, index=Ks, columns=Ts)
    df_deps_std = pd.DataFrame(data=deps_std, index=Ks, columns=Ts)
#################################################
if False:
    # np.save("trained_dep_KSEG_mean_21x20", deps_mean)
    # np.save("trained_dep_KSEG_std_21x20", deps_std)

    df_deps_mean.to_pickle(model1_path+"/df_trained_dep_KSEG_mean_21x20")
    df_deps_std.to_pickle(model1_path+"/df_trained_dep_KSEG_std_21x20")

# import pickle

# deps_mean_r = np.load('trained_dep_KSEG_mean_21x20.npy')

df_deps_mean_r = pd.read_pickle(model1_path+"/df_trained_dep_KSEG_mean_21x20")
df_deps_std_r = pd.read_pickle(model1_path+"/df_trained_dep_KSEG_std_21x20")

if True:
    DataToPlot[3] = [[df_deps_mean_r.columns,
                      df_deps_mean_r.index],
                     [df_deps_mean_r.values]]
    plotWhat = [True, False, False, True]
    model1.plotting.plotData(DataToPlot, plotWhat)
