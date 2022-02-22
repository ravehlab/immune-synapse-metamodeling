#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 18:05:56 2022

@author: yair
"""

import numpy as np
import pymc3 as pm

from Model1.Surrogate import modeling

# Create a heatmap by running the trained model with a batch of x, y value:


def trained_mesh(min_x, max_x, n_x, min_y, max_y, n_y,
                 df_model1_trainedTable):

    Xs = np.linspace(min_x, max_x, n_x)
    Ys = np.linspace(min_y, max_y, n_y)
    outputs_mean = np.zeros((n_y, n_x))
    outputs_std = np.zeros((n_y, n_x))

    for i, y in enumerate(Ys):
        for j, x in enumerate(Xs):
            current_model = modeling.get_pm_model1_trained(
                df_model1_trainedTable, observed_k=y, observed_t=x)
            with current_model:
                current_trace = pm.sample(2000, chains=4, progressbar=False)
            print(f"i,x={i, x}, j,y={j, y}")
            outputs_mean[i, j] = current_trace.rv_output_dep_KSEG1.mean()
            outputs_std[i, j] = current_trace.rv_output_dep_KSEG1.std()
    return outputs_mean, outputs_std
