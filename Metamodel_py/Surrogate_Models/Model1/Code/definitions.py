# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 10:58:29 2022
@author: yairn
"""

import numpy as np
# import pandas as pd
import os

"""
Contains definitions for:
paths
model
submodels
Fitting parameters
Creat Model Info
Training
Predicting
"""

#################################################
# Define model:
model = {}
model['LongName'] = 'Kinetic segregation'
model['ShortName'] = 'KSEG'
model['Index'] = '1'
model['Description'] = """Distributions and inter distances of TCR and CD45
molecules that result from the early contact of a T-cell and an activating
surface."""

#################################################
# Define paths:
modelIndex = '1'
paths = {}
# paths['Metamodel'] = os.getcwd()+'/'
# paths['Metamodel'] = '/home/yair/Documents/Git/Metamodel_py/'
paths['Metamodel'] = '/home/jonah/Yair/Git/immune-synapse-metamodeling/Metamodel_py/'
# paths['Metamodel'] = 'C://Users/Owner/Documents/Git/immune-synapse-metamodeling/Metamodel_py/'
paths['Surrogate'] = paths['Metamodel']+'Surrogate_Models/'
paths['Model'] = paths['Surrogate']+'Model'+modelIndex+'/'
paths['Input'] = paths['Metamodel']+'Input_Models/Model'+modelIndex+'/'
paths['Output'] = paths['Model']+'Output/'
paths['Processing'] = paths['Model']+'Processing/'
print(paths['Metamodel'])
#################################################
# Define data:
data = {}
data['flatten_columns_names'] = ['time_sec', 'k0_kTnm2', 'depletion_nm']
data['shortNames'] = ['t', 'k', 'dep']
data['units'] = ['sec', 'kTnm^2', 'nm']
data['description'] = ['Time',
                       'Membrane rigidity',
                       'Depletion between TCR and CD45']
data['x_min'] = str(0.)
data['x_max'] = str(100.)
data['y_min'] = str(0.)
data['y_max'] = str(100.)
data['distributions'] = ['Uniform', 'Uniform']

#################################################
# Define plots:
figSizeX = 4.  #
figSizeY = 4.  #
plots = {}
plots['figSize'] = [figSizeX, 1*figSizeY]
plots['colormap'] = 'Purples'
plots['xLabel'] = data['shortNames'][0]+'('+data['units'][0]+')'
plots['yLabel'] = data['shortNames'][1]+'('+data['units'][1]+')'
plots['rowTitles'] = ["Training data",
                      "Data fit",
                      "Trained parameters",
                      "Prediction"]
plots['fontSizes1'] = 10
plots['fontSizes2'] = 12
plots['fontSizes3'] = 14
plots['fontSizes4'] = 16
plots['nRoWs'] = 1  # len(plots['rowTitles'])
plots['nCols'] = 1  # len(submodelsNames)

#################################################
# Define free parameters for all submodels in the Model:
fp_x = {}
fp_x['varType'] = 'Free parameter'
fp_x['shortVarType'] = 'fp'
fp_x['shortName'] = data['shortNames'][0]
fp_x['description'] = data['description'][0]
fp_x['texName'] =\
    "$$" +\
    fp_x['shortName'] +\
    "^{" +\
    model['ShortName'] +\
    "}$$"
fp_x['units'] = data['units'][0]
fp_x['ID'] =\
    fp_x['shortVarType'] + '_' +\
    fp_x['shortName'] + '_' +\
    model['ShortName'] +\
    model['Index']
fp_x['distribution'] = 'Uniform'
fp_x['distributionParameters'] = {'lower': data['x_min'],
                                  'upper': data['x_max']}

fp_y = {}
fp_y['varType'] = 'Free parameter'
fp_y['shortVarType'] = 'fp'
fp_y['shortName'] = data['shortNames'][1]
fp_y['description'] = data['description'][1]
fp_y['texName'] =\
    "$$" +\
    fp_y['shortName'] +\
    "^{" +\
    model['ShortName'] +\
    "}$$"
fp_y['units'] = data['units'][1]
fp_y['ID'] =\
    fp_y['shortVarType'] + '_' +\
    fp_y['shortName'] + '_' +\
    model['ShortName'] +\
    model['Index']
fp_y['distribution'] = 'Uniform'
fp_y['distributionParameters'] = {'lower': data['y_min'],
                                  'upper': data['y_max']}

# %% ################################################
# Define submodels:
"""For every output of the same free parameters there is a different
submodel."""

submodelsNames = ['WTCR', 'WCD45', 'Depletion']
submodelName = submodelsNames[2]

submodels = {}
submodels[submodelName] = {}
submodels[submodelName]['fitParametersNames'] =\
    ['p00', 'p10', 'p01','p20', 'p11']

# Fit parameters description Depletion:
submodels[submodelName]['fitParametersDescriptions'] =\
    ['p00', 'p10', 'p01','p20', 'p11']

# Fit parameters units:
submodels[submodelName]['fitParametersUnits'] =\
    ["nm",
     "-",
     "-",
     "-",
     "-",
     "-"]

submodels[submodelName]['p0'] = [45., 1.7, 0.3, 0., 0.]
submodels[submodelName]['sd'] = [10., 0.5, 0.2, 0.1, 0.1]
submodels[submodelName]['tableBackgroundColor'] = 'rgba(200, 150, 255, 0.65)'

submodels[submodelName]['fitFunction'] = \
    'p00 + p10*x + p01*y + p20*x**2 + p11*x*y'


def poly21(xy, p00, p10, p01, p20, p11):

    x, y = xy
    f = eval(submodels[submodelName]['fitFunction'])

    return f

# %% ################################################
# Define fit parameters:
fitParameters = {}

for i, fitParametersName in enumerate(
        submodels[submodelName]['fitParametersNames']):
    #
    fitParameters[fitParametersName] = {}
    fitParameters[fitParametersName]['varType'] = 'Random variable'
    fitParameters[fitParametersName]['shortVarType'] = 'rv'
    fitParameters[fitParametersName]['shortName'] =\
        submodels[submodelName]['fitParametersNames'][i]
    fitParameters[fitParametersName]['description'] = \
        submodels[submodelName]['fitParametersDescriptions'][i]
    fitParameters[fitParametersName]['texName'] =\
        "$$" +\
        fitParameters[fitParametersName]['shortName'] +\
        "^{" +\
        model['ShortName'] +\
        submodelName +\
        "}$$"
    # fitParametersName['units'] = '$$kTnm^2$$'
    fitParameters[fitParametersName]['texName'] =\
        submodels[submodelName]['fitParametersUnits'][i]
    fitParameters[fitParametersName]['ID'] =\
        fitParameters[fitParametersName]['shortVarType'] + '_' +\
        fitParameters[fitParametersName]['shortName'] + '_' +\
        submodelName + '_' +\
        model['ShortName'] +\
        model['Index']
    fitParameters[fitParametersName]['distribution'] = 'Normal'
    fitParameters[fitParametersName]['distributionParameters'] = {
        'mu': str(0.),
        'sd': str(1.)}

    # print(fitParameters[fitParametersName])

################################################
# For Depletion plots:
plots[submodelName] = {}
plots[submodelName]['title'] = submodelName
plots[submodelName]['vmin'] = [0.]
plots[submodelName]['vmax'] = [250.]
plots[submodelName]['contourLevels'] = np.arange(25., 250., 25.)

#################################################
# crateModelInfo:

#################################################
# 4. Training:
# submodel random variables distributions:

#################################################
# 5. Prediction:
# Define parameters for running prediction:


prediction = {}
prediction['n_x'] = 5  # 21  # number of points in x direction.
prediction['max_x'] = 100.
prediction['min_x'] = 0.
prediction['Xs'] = np.linspace(prediction['min_x'],
                               prediction['max_x'],
                               prediction['n_x'])  # x values.

prediction['n_y'] = 4  # 20  # number of points in y direction.
prediction['max_y'] = 100.
prediction['min_y'] = prediction['max_y']/prediction['n_y']
prediction['Ys'] = np.linspace(prediction['min_y'],
                               prediction['max_y'],
                               prediction['n_y'])

prediction['saveName_mean'] = "df_predicted_depletion_mean"
prediction['saveName_std'] = "df_predicted_depletion_std"
#################################################
# y = pm.Normal.dist(mu=10, sd=0.5)
# y.random(size=20)
# rr = pm.distributions.Uniform.dist(0,1,5)