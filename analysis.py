# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 16:23:21 2021

@author: thesq
"""

import numpy as np
import tools
import animation

outputDirectory = "" # choose the directory in which the output files are to be found in format "path/to/file/from/code/directory/"
quiver = True
animate = True

data = np.load(f"{outputDirectory}positions.npy") # loads in position and direction data from file
dirData = np.load(f"{outputDirectory}directions.npy")

if quiver == True:
    tools.quiver(data,dirData)
    
if animate == True:
    animation.main(data.reshape(data.shape[0],data.shape[1],-1))