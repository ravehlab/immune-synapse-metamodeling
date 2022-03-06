#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 15:19:06 2022

@author: yair
"""

import os
from Code import definitions

paths = definitions.paths

def makeDirectory(path, dirName)
# Directory
directory = "Output"

# Parent Directory path
parent_dir = "D:/Pycharm projects/"

# paths['Output'] = paths['Model']+'Output/'
# Path
path = os.path.join(parent_dir, directory)

# Create the directory 'Output' in '/home / User / Documents'
os.mkdir(path)
print("Directory '% s' created" % directory)
