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
paths['Model'] = paths['Surrogate']+'Model2/'
paths['Input'] = paths['Metamodel']+'Input_Models/Model2/'
paths['Processing'] = paths['Model']+'Processing/'

#################################################
# Define model:

model = {}
model['LongName'] = 'Lck activation'
model['ShortName'] = 'LCKA'
model['Index'] = '2'
model['Description'] = """Model2 description """

#################################################
# Define submodels:
"""For every output of the same free parameters there is a different
submodel."""

submodelsNames = ['DecayLength']
submodelName = submodelsNames[0]
submodels = {}
# submodels['names'] = ['Depletion']
submodels[submodelName] = {}
submodels[submodelName]['fitParametersNames'] =\
    ['intercept', 'xSlope', 'ySlope']

# Fit equation Depletion:
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
    ["-",
     "mum^2/sec",
     "nm"]

# Initial fit parameters
submodels[submodelName]['p0'] = [0., 0., 0.]
submodels[submodelName]['tableBackgroundColor'] = 'rgba(200, 150, 0, 0.65)'

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
plots['xLabel'] = "$P_{off}()$"
plots['yLabel'] = "$Diffusion const.(\mu m^2/sec)$"
plots['rowTitles'] = ["Training data",
                      "Data fit",
                      "Trained parameters",
                      "Prediction"]
plots['fontSizes'] = {}
plots['fontSizes']['1'] = 10
plots['fontSizes']['2'] = 12
plots['fontSizes']['3'] = 14
plots['fontSizes']['4'] = 16
plots['nRoWs'] = len(plots['rowTitles'])
plots['nCols'] = len(submodelsNames)

# For Depletion plots:
plots[submodelName] = {}
plots[submodelName]['title'] = submodelName
plots[submodelName]['vmin'] = [1.]
plots[submodelName]['vmax'] = [3.5]
plots[submodelName]['contourLevels'] = np.arange(1., 3.5, 0.25)
plots[submodelName]['colormap'] = 'Blues'

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
x['distributionParameters'] = {'mu': str(-2.),
                               'sd': str(1.)}

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
y['units'] = '$$\mu m^2/sec$$'
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
