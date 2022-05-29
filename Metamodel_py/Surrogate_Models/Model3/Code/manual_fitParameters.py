#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 20:53:06 2022

@author: jonah
"""

import numpy as np
import pandas as pd

parametersNames = ['a', 'xScale', 'xCen', 'xDev', 'yScale', 'yCen', 'yDev']
mu = [0.1, 0.9, 110., 40.0, -0.5, 100., 80.]
sd = [0.1, 0.2, 30., 10., 0.2, 20., 20.]


data = {'mu': mu, 'sd': sd}
index = parametersNames

df_fitParameters = pd.DataFrame(data, index=index)