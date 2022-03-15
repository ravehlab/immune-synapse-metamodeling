# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 10:58:29 2022
@author: yairn
"""

import numpy as np
import pandas as pd

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
# 0. Define paths:
# Define pahts:
# '/home/yair/
# C:/Users/Owner/

paths = {}
paths['home'] = '/home/yair/Documents/Git/'
paths['Metamodel'] = paths['home']+'Metamodel_py/'
paths['Surrogate'] = paths['Metamodel']+'Surrogate_Models/'
paths['Model'] = paths['Surrogate']+'Model3/'
paths['Input'] = paths['Metamodel']+'Input_Models/Model3/'
paths['Processing'] = paths['Model']+'Processing/'

#################################################
# Model definitions:
model = {}
model['LongName'] = 'TCR phosphorylation'
model['ShortName'] = 'TCRP'
model['Index'] = '3'
model['Description'] = """Model3 description."""

#################################################
# Define data names:
data = {}
data['flatten_columns_names'] = {}
data['flatten_columns_names']['x'] = 'Decaylength_nm'
data['flatten_columns_names']['y'] = 'Depletion_nm'
# data['flatten_columns_names']['z'] = 'PhosRatio'
data['flatten_columns_names']['z'] = 'RgRatio'

#################################################
# Define plots:
figSize0 = 4.
plots = {}
plots['figSize'] = [figSize0, 2*figSize0]
plots['xLabel'] = "$Decaylength(nm)$"
plots['yLabel'] = "$Depletion(nm)$"
plots['rowTitles'] = ["Training data",
                      "Data fit",
                      "Trained parameters",
                      "Prediction"]
plots['nRows'] = len(plots['rowTitles'])
plots['nCols'] = 1

plots['fontSizes'] = {}
plots['fontSizes']['1'] = 10
plots['fontSizes']['2'] = 12
plots['fontSizes']['3'] = 14
plots['fontSizes']['4'] = 16


#################################################
# Define free parameters for all submodels in the Model:
x = {}
x['varType'] = 'Free parameter'
x['shortVarType'] = 'fp'
x['shortName'] = 'Decaylength'
x['description'] = 'Decay length of active Lck'
x['texName'] =\
    "$$" +\
    x['shortName'] +\
    "^{" +\
    model['ShortName'] +\
    "}$$"
x['units'] = '$$nm$$'
x['ID'] =\
    x['shortVarType'] + '_' +\
    x['shortName'] + '_' +\
    model['ShortName'] +\
    model['Index']
x['distribution'] = 'Normal'
x['distributionParameters'] = {'mu': str(100.),
                               'sd': str(50.)}

y = {}
y['varType'] = 'Free parameter'
y['shortVarType'] = 'fp'
y['shortName'] = 'Depletion'
y['description'] = 'Depletion distsance between TCR and CD45'
y['texName'] =\
    "$$" +\
    y['shortName'] +\
    "^{" +\
    model['ShortName'] +\
    "}$$"
y['units'] = '$$nm$$'
y['ID'] =\
    y['shortVarType'] + '_' +\
    y['shortName'] + '_' +\
    model['ShortName'] +\
    model['Index']
y['distribution'] = 'Normal'
y['distributionParameters'] = {'mu': str(100.),
                               'sd': str(50.)}
#################################################
# Define submodels:
"""For every output of the same free parameters there is a different
submodel."""

submodelsNames = ['PhosRatio', 'RgRatio']
#################################################
submodelName = submodelsNames[1]
submodels = {}
# submodels['names'] = ['PhosRatio']
submodels[submodelName] = {}
submodels[submodelName]['fitParametersNames'] =\
    ['intercept', 'xSlope', 'ySlope']

# Fit equation PhosRatio:
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
    ["Intersection with z axis (nm)",
     "Slope in x direction",
     "Slope in y direction"]

# Fit parameters units:
submodels[submodelName]['fitParametersUnits'] =\
    ["nm",
     "-",
     "-"]

# Initial fit parameters
submodels[submodelName]['p0'] = [0., 0., 0.]
submodels[submodelName]['tableBackgroundColor'] = 'rgba(200, 150, 0, 0.65)'

#################################################

# submodelName = submodelsNames[1]
# submodels[submodelName] = {}
# submodels[submodelName]['fitParametersNames'] =\
#     ['intercept', 'xSlope', 'ySlope']

# # Fit equation PhosRatio:
# submodels[submodelName]['equation'] =\
#     submodels[submodelName]['fitParametersNames'][0] +\
#     "+" +\
#     submodels[submodelName]['fitParametersNames'][1] +\
#     "*" +\
#     "x" +\
#     "+" +\
#     submodels[submodelName]['fitParametersNames'][2] +\
#     "*" +\
#     "y"

# # Fit parameters description Depletion:
# submodels[submodelName]['fitParametersDescriptions'] =\
#     ["Intersection with z axis (nm)",
#      "Slope in x direction",
#      "Slope in y direction"]

# # Fit parameters units:
# submodels[submodelName]['fitParametersUnits'] =\
#     ["nm",
#      "-",
#      "-"]

# # Initial fit parameters
# submodels[submodelName]['p0'] = [0., 0., 0.]
# submodels[submodelName]['tableBackgroundColor'] = 'rgba(200, 150, 0, 0.65)'

# For Depletion plots:
submodelName = submodelsNames[0]
plots[submodelName] = {}
plots[submodelName]['title'] = submodelName
plots[submodelName]['vmin'] = [0.]
plots[submodelName]['vmax'] = [1.]
plots[submodelName]['contourLevels'] = np.arange(0.1, 1., 0.1)
plots[submodelName]['colormap'] = 'Oranges'


submodelName = submodelsNames[1]
plots[submodelName] = {}
plots[submodelName]['title'] = submodelName
plots[submodelName]['vmin'] = [0.]
plots[submodelName]['vmax'] = [1.5]
plots[submodelName]['contourLevels'] = np.arange(0.1, 1.5, 0.1)
plots[submodelName]['colormap'] = 'Oranges'

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
    # fitParametersName['distribution'] = 'Uniform'
    # fitParametersName['distributionParameters'] = {'lower': str(0.),
    #                                'upper': str(100.)}

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