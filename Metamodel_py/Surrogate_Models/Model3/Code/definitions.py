# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 10:58:29 2022
@author: yairn
"""

import numpy as np
import pandas as pd
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
model['LongName'] = 'TCR phosphorylation'
model['ShortName'] = 'TCRP'
model['Index'] = '3'
model['Description'] = """Model3 description."""

#################################################
# Define paths:
modelIndex = '3'
paths = {}
paths['Metamodel'] = os.getcwd()+'/'
paths['Surrogate'] = paths['Metamodel']+'Surrogate_Models/'
paths['Model'] = paths['Surrogate']+'Model'+modelIndex+'/'
paths['Input'] = paths['Metamodel']+'Input_Models/Model'+modelIndex+'/'
paths['Output'] = paths['Model']+'Output/'
paths['Processing'] = paths['Model']+'Processing/'

#################################################
# Define data:  (Specific)
data = {}
data['flatten_columns_names'] = ['Decaylength_nm',
                                 'Depletion_nm',
                                 'PhosRatio',
                                 'RgRatio']

data['shortNames'] = ['Decaylength',
                      'Depletion',
                      'PhosRatio',
                      'RgRatio']
data['units'] = ['nm', 'nm', '-', '-']
data['description'] = ['Decay length of active Lck in nm.',
                       'Depletion distance between TCR and CD45',
                       'Ratio between the number of phosphorylated TCRs'
                       'to the total number of TCRs',
                       'Ratio between the radii of gyration of the'
                       'phosphorylated TCRs and all the TCRs']
data['x_mu'] = str(100.)
data['x_sd'] = str(50.)
data['y_mu'] = str(100.)
data['y_sd'] = str(50.)
data['distributions'] = ['Normal', 'Normal']

#################################################
# Define plots: (General)
figSizeX = 4.  #
figSizeY = 13.  #
plots = {}
plots['figSize'] = [figSizeX, 1*figSizeY]
plots['colormap'] = 'Oranges'
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
plots['nRows'] = len(plots['rowTitles'])
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
fp_x['distribution'] = 'Normal'
fp_x['distributionParameters'] = {'mu': data['x_mu'],
                                  'sd': data['x_sd']}

fp_y = {}
fp_y['varType'] = 'Free parameter'
fp_y['shortVarType'] = 'fp'
fp_y['shortName'] = 'Depletion'
fp_y['description'] = 'Depletion distsance between TCR and CD45'
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
fp_y['distribution'] = 'Normal'
fp_y['distributionParameters'] = {'mu': data['y_mu'],
                                  'sd': data['y_sd']}
#################################################
# Define submodels:
"""For every output of the same free parameters there is a different
submodel."""

submodelsNames = ['PhosRatio', 'RgRatio']
#################################################
submodelName = submodelsNames[0]
submodels = {}
# submodels['names'] = ['PhosRatio']
submodels[submodelName] = {}
submodels[submodelName]['fitParametersNames'] =\
    ['DecaylengthScale',
     'DecaylengthMu',
     'DecaylengthSigma',
     'DepletionScale',
     'DepletionMu',
     'DepletionSigma']

# Fit parameters description Depletion:
submodels[submodelName]['fitParametersDescriptions'] =\
    ['Decaylengthcale',
     'DecaylengthMu',
     'DecaylengthSigma',
     'DepletionScale',
     'DepletionMu',
     'DepletionSigma']

# Fit parameters units:
submodels[submodelName]['fitParametersUnits'] =\
    ["-",
     "nm",
     "nm",
     "-",
     "nm",
     "nm"]

# Initial fit parameters
# p0_phosRatio = 1., 100., 100., 1., 100., 100.
# mu	sd
# xScale	0.959	0.065
# xMu	263.251	13.893
# xSigma	113.371	7.837
# yScale	-0.228	0.018
# yMu	192.868	11.107
# ySigma	85.407	12.065
submodels[submodelName]['p0'] = [1., 260., 100, 0.2, 200., 100.]
submodels[submodelName]['sd'] = [1., 50., 50., 1., 50., 50.]
submodels[submodelName]['tableBackgroundColor'] = 'rgba(200, 150, 0, 0.65)'

#################################################
# For plots:
submodelName = submodelsNames[0]
plots[submodelName] = {}
plots[submodelName]['title'] = submodelName
plots[submodelName]['vmin'] = [0.]
plots[submodelName]['vmax'] = [1.]
plots[submodelName]['contourLevels'] = np.arange(0.1, 1., 0.1)


# submodelName = submodelsNames[1]
# plots[submodelName] = {}
# plots[submodelName]['title'] = submodelName
# plots[submodelName]['vmin'] = [1.0]
# plots[submodelName]['vmax'] = [1.5]
# plots[submodelName]['contourLevels'] = np.arange(1.0, 1.5, 0.1)

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
        submodelName + '_' +\
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
    fitParameters[fitParametersName]['distributionParameters'] =\
        {'mu': submodels[submodelName]['p0'][i],
         'sd': submodels[submodelName]['sd'][i]}

    print(fitParameters[fitParametersName])


# p0_RgRatio = 1., 200., 100., 1., 0.
# mu	sd
# xScale	-0.564	0.023
# xMu	197.729	4.257
# xSigma	96.988	5.777
# yIntercept	1.516	0.021
# ySlope	0.001	0.000

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