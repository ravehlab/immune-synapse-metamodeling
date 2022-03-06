# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 10:58:29 2022
@author: yairn
"""

import numpy as np
import pandas as pd

"""
Contains:
Plotting:
Font type

Axes scales
Axes units
Axes ticks
Axes labels
Plots titles
rendering?
Fitting:

Creat Model Info:
Training:
Predicting:
"""

#################################################
# 0. Define data names and paths:
# Define pahts:
# '/home/yair/
# C:/Users/Owner/

paths = {}
paths['home'] = '/home/yair/Documents/Git/'
paths['Metamodel'] = paths['home']+'Metamodel_py/'
paths['Surrogate'] = paths['Metamodel']+'Surrogte_Models/'
paths['Model'] = paths['Surrogate']+'Model1/'
paths['Input'] = paths['Metamodel']+'Input_Models/Model1/Input/'
paths['Processing'] = paths['Model']+'Processing/'

#################################################
# Define model:

model = {}
model['LongName'] = 'Kinetic segregation'
model['ShortName'] = 'KSEG'
model['Index'] = '1'
model['Description'] = """Distributions and inter distances of TCR and CD45
molecules that result from the early contact of a T-cell and an activating
surface."""
# Define submodels:

# model['submodels'] = ['Depletion']
#################################################
# Define submodels:

submodels = {}
submodels['names'] = ['Depletion']
submodels['names'][0] = {}
submodels['names'][0]['fitParametersNames'] =\
    ['intercept', 'xSlope', 'ySlope']

# Fit equation Depletion:
submodels['names'][0]['equation'] =\
    submodels['names'][0]['fitParametersNames'][0] +\
    "+" +\
    submodels['names'][0]['fitParametersNames'][1] +\
    "*" +\
    "x" +\
    "+" +\
    submodels['names'][0]['fitParametersNames'][2] +\
    "*" +\
    "y"

# Fit parameters description Depletion:
submodels['names'][0]['fitParametersDescriptions'] =\
    ["Intersection with z axis (nm)",
     "Slope in x direction",
     "Slope in y direction"]

# Fit parameters units:
submodels['names'][0]['fitParametersUnits'] =\
    ["nm",
     "sec",
     "kTnm^2"]

# Initial fit parameters
submodels['names'][0]['p0'] = [100., 0., 0.]
submodels['names'][0]['tableBackgroundColor'] = 'rgba(200, 150, 255, 0.65)'

#################################################
# Define data names:
data = {}
data['flatten_columns_names'] = {}
data['flatten_columns_names']['x'] = 'time_sec'
data['flatten_columns_names']['y'] = 'k0_kTnm2'
data['flatten_columns_names']['z'] = 'depletion_nm'

#################################################
# Define plots:
plots = {}
plots['figSize'] = [4., len(submodels['names'])*4.]
plots['xLabel'] = "$t(sec)$"
plots['yLabel'] = "$\kappa(kTnm^2)$"
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
plots['nCols'] = len(submodels['names'])

# For Depletion plots:
plots['Depletion'] = {}
plots['Depletion']['title'] = 'Depletion'
plots['Depletion']['vmin'] = [0]
plots['Depletion']['vmax'] = [250]
plots['Depletion']['contourLevels'] = np.arange(25., 250., 25.)
plots['colormap'] = 'Purples'

#################################################
# Define free parameters for all submodels in the Model:
x = {}
x['varType'] = 'Free parameter'
x['shortVarType'] = 'fp'
x['shortName'] = 't'
x['description'] = 'Time'
x['texName'] =\
    "$$" +\
    x['shortName'] +\
    "^{" +\
    model['ShortName'] +\
    "}$$"
x['units'] = '$$sec$$'
x['ID'] =\
    x['shortVarType'] + '_' +\
    x['shortName'] + '_' +\
    model['ShortName'] +\
    model['Index']
x['distribution'] = 'Uniform'
x['distributionParameters'] = {'lower': str(0.),
                               'upper': str(100.)}

y = {}
y['varType'] = 'Free parameter'
y['shortVarType'] = 'fp'
y['shortName'] = 'k'
y['description'] = 'Membrane rigidity'
y['texName'] =\
    "$$" +\
    y['shortName'] +\
    "^{" +\
    model['ShortName'] +\
    "}$$"
y['units'] = '$$kTnm^2$$'
y['ID'] =\
    y['shortVarType'] + '_' +\
    y['shortName'] + '_' +\
    model['ShortName'] +\
    model['Index']
y['distribution'] = 'Uniform'
y['distributionParameters'] = {'lower': str(0.),
                               'upper': str(100.)}


#################################################
# Define fit parameters:

fitParameters = {}

for i, fitParametersName in enumerate(
        submodels['names'][0]['fitParametersNames']):
    #
    fitParameters[fitParametersName] = {}
    fitParameters[fitParametersName]['varType'] = 'Random variable'
    fitParameters[fitParametersName]['shortVarType'] = 'rv'
    fitParameters[fitParametersName]['shortName'] =\
        submodels['names'][0]['fitParametersNames'][i]
    fitParameters[fitParametersName]['description'] = \
        submodels['names'][0]['fitParametersDescriptions'][i]
    fitParameters[fitParametersName]['texName'] =\
        "$$" +\
        fitParameters[fitParametersName]['shortName'] +\
        "^{" +\
        model['ShortName'] +\
        "}$$"
    # fitParametersName['units'] = '$$kTnm^2$$'
    fitParameters[fitParametersName]['texName'] =\
        submodels['names'][0]['fitParametersUnits'][i]
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
