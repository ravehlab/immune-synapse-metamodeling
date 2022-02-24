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

axes_names_units = ['time_sec', 'k0_kTnm2', 'dep_nm']

axes_labels = ["$t(sec)$", "$\kappa(kTnm^2)$"]

dep_Title = "Depletion \n"
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

# equation_dep = parametersNames_dep[0] + \
#                 "+" + \
#                 parametersNames_dep[1] + \
#                 "*" + \
#                 "t" + \
#                 "+" + \
#                 parametersNames_dep[2] +\
#                 "*" + \
#                 "k"

# Get fit parameters:
parametersNames_dep = ['intercept', 'xSlope', 'ySlope']
#################################################
# 4.4
n_x = 21  # number of points in x direction.
max_x = 100.
min_x = 0.
Xs = np.linspace(min_x, max_x, n_x)  # x values.

n_y = 20
max_y = 100.
min_y = max_y/n_y
Ys = np.linspace(min_y, max_y, n_y)
