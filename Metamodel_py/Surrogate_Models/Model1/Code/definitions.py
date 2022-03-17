# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 10:58:29 2022
@author: yairn
"""

import numpy as np
# import pandas as pd

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
# '/home/yair/
# C:/Users/Owner/

paths = {}
paths['Git'] = '/home/yair/Documents/Git/'
paths['Metamodel'] = paths['Git']+'Metamodel_py/'
paths['Surrogate'] = paths['Metamodel']+'Surrogate_Models/'
paths['Model'] = paths['Surrogate']+'Model'+model['Index']+'/'
paths['Input'] = paths['Metamodel']+'Input_Models/Model'+model['Index']+'/'
paths['Output'] = paths['Model']+'Output/'
paths['Processing'] = paths['Model']+'Processing/'

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
plots['nRoWs'] = len(plots['rowTitles'])
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

#################################################
# Define submodels:
"""For every different output of the free parameters there is a different
submodel."""

submodelsNames = ['WTCR', 'WCD45', 'Depletion']
submodelName = submodelsNames[2]

submodels = {}
# submodels['names'] = ['Depletion']
submodels[submodelName] = {}
submodels[submodelName]['fitParametersNames'] =\
    ['intercept', 'xSlope', 'ySlope']

# Fit equation Depletion:
submodels[submodelName]['bareEquation'] = 'b + ax*x + ay*y'
###
b = submodels[submodelName]['fitParametersNames'][0]
ax = submodels[submodelName]['fitParametersNames'][1]
ay = submodels[submodelName]['fitParametersNames'][2]

###

submodels[submodelName]['equation'] =\
    submodels[submodelName]['fitParametersNames'][0] +\
    "+" +\
    submodels[submodelName]['fitParametersNames'][1] +\
    "*" +\
    "x" +\
    "+" +\
    submodels[submodelName]['fitParametersNames'][2] +\
    "*" +\
    "y"

# Fit parameters description Depletion:
submodels[submodelName]['fitParametersDescriptions'] =\
    ["Intersept with z axis (nm)",
     "Slope in x direction",
     "Slope in y direction"]

# Fit parameters units:
submodels[submodelName]['fitParametersUnits'] =\
    ["nm",
     "sec",
     "kTnm^2"]

# Initial fit parameters
submodels[submodelName]['p0'] = [100., 0., 0.]
submodels[submodelName]['tableBackgroundColor'] = 'rgba(200, 150, 255, 0.65)'

submodels[submodelName]['fitFunction'] = None

#################################################
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
        model['ShortName'] +\
        model['Index']
    fitParameters[fitParametersName]['distribution'] = 'Normal'
    fitParameters[fitParametersName]['distributionParameters'] = {
        'mu': str(0.),
        'sd': str(1.)}

    print(fitParameters[fitParametersName])

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
prediction['n_x'] = 21  # number of points in x direction.
prediction['max_x'] = 100.
prediction['min_x'] = 0.
prediction['Xs'] = np.linspace(prediction['min_x'],
                               prediction['max_x'],
                               prediction['n_x'])  # x values.

prediction['n_y'] = 20  # number of points in y direction.
prediction['max_y'] = 100.
prediction['min_y'] = prediction['max_y']/prediction['n_y']
prediction['Ys'] = np.linspace(prediction['min_y'],
                               prediction['max_y'],
                               prediction['n_y'])

#################################################
# y = pm.Normal.dist(mu=10, sd=0.5)
# y.random(size=20)
# rr = pm.distributions.Uniform.dist(0,1,5)