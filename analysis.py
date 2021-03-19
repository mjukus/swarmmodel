# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 11:50:51 2021

@author: mawga
"""

import numpy as np
import tools
import animation

outputDirectory = "" # choose the directory in which the output files are to be found in format "path/to/file/from/code/directory/"
quiver = True
animate = False
Crystal_order = True
data = np.load(f"{outputDirectory}positions.npy") # loads in position and direction data from file
dirData = np.load(f"{outputDirectory}directions.npy")

if quiver == True:
    tools.quiver(data,dirData,10000)
    
if animate == True:
    animation.main(data.reshape(data.shape[0],data.shape[1],-1))
          

if Crystal_order == True:
    tools.crystal_order(dirData,10000,4**3)   

