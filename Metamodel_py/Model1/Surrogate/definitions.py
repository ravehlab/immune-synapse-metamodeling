# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 10:58:29 2022

@author: yairn
"""

import numpy as np

"""
Contains:

Plotting:
Font sizes
Font type
model colors
Axes scales
Axes units
Axes ticks
Axes labels
Plots titles
rendering?

Fitting:
Fit equations
Fit parameters names

Creat Model Info:


Training:


Predicting:
"""

import pandas as pd

#################################################
# 0. Define data names and paths:
# Define pahts:
# '/home/yair/

paths = {}
paths['home'] = 'C:/Users/Owner/Documents/Git/immune-synapse-metamodeling/'
paths['Metamodel'] = paths['home']+'Metamodel_py/'
paths['Model'] = paths['Metamodel']+'Model1/'
paths['Input'] = paths['Model']+'Input/'
paths['Processing'] = paths['Model']+'Processing/'
paths['Output'] = paths['Model']+'Output/'

# Define data names
Input_data_name_pivot = 'df_trainingData_depletion_pivot.csv'

# Read trainingData aranged as dataFrame - pivot:
df_trainingData_depletion_pivot = pd.read_csv(
    paths['Input']+'df_trainingData_depletion_pivot.csv')
   
# Get trainingData aranged as dataFrame in columns (flatten):
df_trainingData_depletion_flatten =\
    pivotToFlatten(df_trainingData_depletion_pivot)

df_trainingData_depletion_flatten.to_csv(
    Input_path+"/df_trainingData_depletion_flatten.csv")    

# 1. Define plots:
plots = {}
    
axes_names_units = ['time_sec', 'k0_kTnm2', 'depletion_nm']

axes_labels = ["$t(sec)$", "$\kappa(kTnm^2)$"]

dep_Title = "Depletion"
colTitles = [dep_Title]

rowTitles = ["Training data", "Data fit", "Trained parameters", "Prediction"]

vmins = [0]
vmaxs = [250]

nRows = 4
nCols = 1

figsize = [4., 12.]

# Contour levels for depletion heatmap:
dep_contour_levels = np.arange(25., 250., 25.)

colormap = 'Purples'

fontsize1 = 10
fontsize2 = 12
fontsize3 = 14
fontsize4 = 16

#################################################
# 2.1 Define fit equations and parameters:


parametersNames_depletion = ['intercept', 'xSlope', 'ySlope']

equation_depletion = parametersNames_depletion[0] +\
                    "+" +\
                    parametersNames_depletion[1] +\
                    "*" +\
                    "x" +\
                    "+" +\
                    parametersNames_depletion[2] +\
                    "*" + \
                    "y"

# Get fit parameters:

#################################################
# 3. Create model info:

# dict_Model1 = {}
# dict_Model1['modelShortName'] = 'KSEG'

modelLongName = 'Kinetic segregation'
modelShortName = 'KSEG'
modelIndex = '1'
modelDescription = """Distributions and inter distances of TCR and CD45
molecules that result from the early contact of a T-cell and an activating
surface."""

shortName_x = 't'
shortName_y = 'k'

ID_x = 'fp_' + shortName_x + '_' + modelShortName+modelIndex
ID_y = 'fp_' + shortName_y + '_' + modelShortName+modelIndex

dict_x = {}
dict_x['ID'] = ID_x
dict_x['varType'] = 'Free parameter'
dict_x['shortName'] = 't'
dict_x['texName'] = "$$t^{KSEG}$$"
dict_x['units'] = '$$sec$$'
dict_x['description'] = 'Time'
dict_x['distribution'] = 'Uniform'
dict_x['distributionParameters'] = {'lower': str(0.),
                                    'upper': str(100.)}


dict_y = {}
dict_y['ID'] = ID_y
dict_y['varType'] = 'Free parameter'
dict_y['shortName'] = 'k'
dict_y['texName'] = "$$\kappa^{KSEG}$$"
dict_y['units'] = '$$kTnm^2$$'
dict_y['description'] = 'Membrane rigidity'
dict_y['distribution'] = 'Uniform'
dict_y['distributionParameters'] = {'lower': str(0.),
                                    'upper': str(100.)}

submodelName = 'depletion'

#############################
# y = pm.Normal.dist(mu=10, sd=0.5)
# y.random(size=20)

dict_depletion = {}
dict_depletion['ID_x'] = ID_x

depletion_background_color_rgba = (200, 150, 255, 0.65)
#############################


def model1_depletion_info(df_fitParameters_depletion):

    model1_depletion.add_rv(
        RV(id=dict_x['ID'],
           varType=dict_x['varType'],
           shortName=dict_x['shortName'],
           texName=dict_x['texName'],
           description=dict_x['description'],
           distribution=dict_x['distribution'],
           distributionParameters=dict_x['distributionParameters'],
           units=dict_x['units']))

    model1_depletion.add_rv(
        RV(id=dict_y['ID'],
           varType=dict_y['varType'],
           shortName=dict_y['shortName'],
           texName=dict_y['texName'],
           description=dict_y['description'],
           distribution=dict_y['distribution'],
           distributionParameters=dict_y['distributionParameters'],
           units=dict_y['units']))

    model1_depletion.add_rv(
        RV(id='rv_intercept_depletion_KSEG1',
            varType='Random variable',
            shortName='intercept',
            texName='$$dep^{KSEG}_{intercept}$$',
            description='Interception with z axis',
            distribution='Normal',
            distributionParameters={
                'mu': str(df_fitParameters_depletion.loc['intercept', 'mu']),
                'sd': str(df_fitParameters_depletion.loc['intercept', 'sd'])},
            units='$$nm$$'))

    model1_depletion.add_rv(
        RV(id='rv_tSlope_depletion_KSEG1',
            varType='Random variable',
            shortName='tSlope',
            texName='$$dep^{KSEG}_{tSlope}$$',
            description='Slope in t direction',
            distribution='Normal',
            distributionParameters={
                'mu': str(df_fitParameters_depletion.loc['tSlope', 'mu']),
                'sd': str(df_fitParameters_depletion.loc['tSlope', 'sd'])},
            units='$$sec$$'))

    model1_depletion.add_rv(
        RV(id='rv_kSlope_depletion_KSEG1',
            varType='Random variable',
            shortName='kSlope',
            texName='$$dep^{KSEG}_{kSlope}$$',
            description='Slope in k direction',
            distribution='Normal',
            distributionParameters={
                'mu': str(df_fitParameters_depletion.loc['kSlope', 'mu']),
                'sd': str(df_fitParameters_depletion.loc['kSlope', 'sd'])},
            units='$$kTnm^2$$'))

    model1_depletion.add_rv(
        RV(id='rv_output_depletion_KSEG1',
           varType='Random variable',
           shortName='output',
           texName='$$depletion^{KSEG}_{output}$$',
           description='depletion output',
           distribution='Normal',
           distributionParameters={'mu': '',
                                   'sd': str(20.)},
           units="$$nm$$"))

    model1_depletion_info.to_csv("Model1/Processing/Model1_Info_depletion.csv")

    return(model1_depletion_info)
#################################################

#################################################
# 4. Training:

#################################################
# 5. Prediction:
# Define parameters for running prediction:


n_x = 21  # number of points in x direction.
max_x = 100.
min_x = 0.
Xs = np.linspace(min_x, max_x, n_x)  # x values.

n_y = 20  # number of points in y direction.
max_y = 100.
min_y = max_y/n_y
Ys = np.linspace(min_y, max_y, n_y)
