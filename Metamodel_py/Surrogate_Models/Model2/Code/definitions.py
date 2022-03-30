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
model['LongName'] = 'Lck activation'
model['ShortName'] = 'LCKA'
model['Index'] = '2'
model['Description'] = """Model2 description """

#################################################
# Define paths:
modelIndex = '2'
paths = {}
paths['Metamodel'] = os.getcwd()+'/'
paths['Surrogate'] = paths['Metamodel']+'Surrogate_Models/'
paths['Model'] = paths['Surrogate']+'Model'+modelIndex+'/'
paths['Input'] = paths['Metamodel']+'Input_Models/Model'+modelIndex+'/'
paths['Output'] = paths['Model']+'Output/'
paths['Processing'] = paths['Model']+'Processing/'

#################################################
# Define data:
data = {}
data['flatten_columns_names'] = ['Poff', 'Diff_um^2/sec', 'Decaylength_nm']
data['shortNames'] = ['Poff', 'Diff', 'Decaylength']
data['units'] = ['-', 'um^2/sec', 'nm']
data['description'] = ['Decay probability',
                       'Diffusion coefficient',
                       'Decay length of active lck']
data['x_mu'] = str(-2.)
data['x_sd'] = str(1.)
data['y_mu'] = str(-2.)
data['y_sd'] = str(1.)
data['distributions'] = ['Normal', 'Normal']

#################################################
# Define plots:
figSizeX = 4.  #
figSizeY = 4.  #
plots = {}
plots['figSize'] = [figSizeX, 1*figSizeY]
plots['colormap'] = 'Blues'
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
fp_x['distribution'] = data['distributions'][0]
fp_x['distributionParameters'] = {'mu': data['x_mu'],
                                  'sd': data['x_sd']}

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
fp_y['distribution'] = data['distributions'][1]
fp_y['distributionParameters'] = {'mu': data['y_mu'],
                                  'sd': data['y_sd']}

#################################################
# Define submodels:
"""For every output of the same free parameters there is a different
submodel."""

submodelsNames = ['Decaylength']
submodelName = submodelsNames[0]

submodels = {}
submodels[submodelName] = {}
submodels[submodelName]['fitParametersNames'] =\
    ['PoffScale', 'PoffMu', 'PoffSigma',
     'DiffScale', 'DiffMu', 'DiffSigma']

# Fit parameters description Depletion:
submodels[submodelName]['fitParametersDescriptions'] =\
    ['PoffScale',
     'PoffMu',
     'PoffSigma',
     'DiffScale',
     'DiffMu',
     'DiffSigma']

# Fit parameters units:
submodels[submodelName]['fitParametersUnits'] =\
    ["nm",
     "nm",
     "nm",
     "nm",
     "nm",
     "nm"]

# Initial fit parameters
#            mu	    sd
# xScale	1.908	0.103
# xMu	-4.074	0.109
# xSigma	2.480	0.162
# yScale	0.917	0.145
# yMu	-0.666	0.287
# ySigma	1.023	0.282
submodels[submodelName]['p0'] = [1.9, -4., 2.5, 0.9, -0.7, 1.]
submodels[submodelName]['sd'] = [1., 1., 1., 1., 1., 1.]
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

#################################################
# Define data names:
data = {}
data['flatten_columns_names'] = {}
data['flatten_columns_names']['x'] = 'Poff'
data['flatten_columns_names']['y'] = 'Diff'
data['flatten_columns_names']['z'] = 'Decaylength_nm'

#################################################
# Define plots:
plots = {}
plots['figSize'] = [4., 13.]
plots['colormap'] = 'Blues'
plots['xLabel'] = "$P_{off}()$"
plots['yLabel'] = "$Diffusion const.(\mu^2/sec)$"
plots['rowTitles'] = ["Training data",
                      "Data fit",
                      "Trained parameters",
                      "Prediction"]
plots['fontSizes1'] = 10
plots['fontSizes2'] = 12
plots['fontSizes3'] = 14
plots['fontSizes4'] = 16
plots['nRoWs'] = len(plots['rowTitles'])
plots['nCols'] = len(submodelsNames)

# For Depletion plots:
plots[submodelName] = {}
plots[submodelName]['title'] = submodelName
plots[submodelName]['vmin'] = [1.]
plots[submodelName]['vmax'] = [3.5]
plots[submodelName]['contourLevels'] = np.arange(1., 3.5, 0.25)

#################################################
# Define free parameters for all submodels in the Model:
x = {}
x['varType'] = 'Free parameter'
x['shortVarType'] = 'fp'
x['shortName'] = 'Poff'
x['description'] = 'Decay probability'
x['texName'] =\
    "$$" +\
    x['shortName'] +\
    "^{" +\
    model['ShortName'] +\
    "}$$"
x['units'] = '$$-$$'
x['ID'] =\
    x['shortVarType'] + '_' +\
    x['shortName'] + '_' +\
    model['ShortName'] +\
    model['Index']
x['distribution'] = 'Normal'
x['distributionParameters'] = {'mu': -2.,  # str(-2.)
                               'sd': 1.}  # str(1.)

y = {}
y['varType'] = 'Free parameter'
y['shortVarType'] = 'fp'
y['shortName'] = 'Diff'
y['description'] = 'Diffusion constant'
y['texName'] =\
    "$$" +\
    y['shortName'] +\
    "^{" +\
    model['ShortName'] +\
    "}$$"
y['units'] = '$$\mu^2/sec$$'
y['ID'] =\
    y['shortVarType'] + '_' +\
    y['shortName'] + '_' +\
    model['ShortName'] +\
    model['Index']
y['distribution'] = 'Normal'
y['distributionParameters'] = {'mu': str(-2.),
                               'sd': str(1.)}


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
        submodelName + '_' +\
        model['ShortName'] +\
        model['Index']
    fitParameters[fitParametersName]['distribution'] = 'Normal'
    fitParameters[fitParametersName]['distributionParameters'] = {
        'mu': str(2.),
        'sd': str(1.)}

    print(fitParameters[fitParametersName])


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
prediction['max_x'] = 0.
prediction['min_x'] = 1E-5
prediction['Xs'] = np.linspace(prediction['min_x'],
                               prediction['max_x'],
                               prediction['n_x'])  # x values.

prediction['n_y'] = 20  # number of points in y direction.
prediction['max_y'] = 1E-0
prediction['min_y'] = 1E-3
prediction['Ys'] = np.linspace(prediction['min_y'],
                               prediction['max_y'],
                               prediction['n_y'])

#################################################
