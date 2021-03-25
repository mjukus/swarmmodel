# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 11:50:51 2021

@author: mawga
"""

import numpy as np
import tools
import animation

outputDirectory = "" # choose the directory in which the output files are to be found in format "path/to/file/from/code/directory/"
quiver = False
animate = False
Crystal_order = False
Histogram = False
kMeans = False
AggHierarchy = False
BestClusterNumber = True

data = np.load(f"{outputDirectory}positions.npy") # loads in position and direction data from file
dirData = np.load(f"{outputDirectory}directions.npy")

if quiver == True:                 #Creates Quiver plot, input form (data,dirData,Nt)
    tools.quiver(data,dirData,1000)
    
if animate == True:         
    animation.main(data.reshape(data.shape[0],data.shape[1],-1))
          

if Crystal_order == True:          #Creates plot of crystal order over time, input form (dirData,Nt,axisN**3)
    tools.crystal_order(dirData,2000,7**3)   
    
if Histogram == True:              #Creates a histogram of pos at a given time, input form (data,Nt,axisN**3,nRod)
    tools.histogram(data,1400,7**3,4)    

if kMeans == True:                 #Kmeans cluster plot at a given time, input form (data,Nt,axisN**3,number of clusters)
    tools.kMeans(data,1500,7**3,3)    

if AggHierarchy == True:           #Agg cluster plot at a given time, input form (data,Nt,axisN**3,number of clusters)
    tools.AggHierarchy(data,0,7**3,4)  
    
if BestClusterNumber == True:      #Gives the number of clusters that best fits the data using silhoutte score and plots
    tools.BestClusterNumber(data,1500,7**3)    