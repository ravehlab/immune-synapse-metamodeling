#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 08:01:39 2022

@author: yair
"""

import pandas as pd



uploaded_trace1_summary = pd.read_pickle('trace1_summary')
print(uploaded_trace1_summary)



model1_rv_names = uploaded_trace1_summary.index[:]
print(model1_rv_names.shape)
### mean:
for model1_rv_name in model1_rv_names:
    # print(model1_rv_name)
    mean_model1_rv_name = 'mean_' + model1_rv_name
    std_model1_rv_name = 'std_' + model1_rv_name

    mean_model1_rv_name = uploaded_trace1_summary.loc[model1_rv_name,'mean']
    std_model1_rv_name = uploaded_trace1_summary.loc[model1_rv_name,'sd']
    print(std_model1_rv_name)