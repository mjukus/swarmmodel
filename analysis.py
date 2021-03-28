# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 16:23:21 2021

@author: thesq
"""

import numpy as np
import tools
import animation

outputDirectory = "2021-03-28_1000_4_10000/" # choose the directory in which the output files are to be found in format "path/to/file/from/code/directory/"
quiver = False
animate = False
Crystal_order = False
Histogram = False
kMeans = False
AggHierarchy = True
BestClusterNumber = False

data = np.load(f"{outputDirectory}positions.npy") # loads in position and direction data from file
dirData = np.load(f"{outputDirectory}directions.npy")

if quiver == True:                 #Creates Quiver plot, input form (data,dirData,Nt)
    tools.quiver(data,dirData,100)
    
if animate == True:         
    animation.main(data.reshape(data.shape[0],data.shape[1],-1))
          

if Crystal_order == True:          #Creates plot of crystal order over time, input form (dirData,Nt,axisN**3)
    tools.crystal_order(dirData,100,9**3)   
    
if Histogram == True:              #Creates a histogram of pos at a given time, input form (data,Nt,axisN**3,nRod)
    tools.histogram(data,90,9**3,4)    

if kMeans == True:                 #Kmeans cluster plot at a given time, input form (data,Nt,axisN**3,number of clusters)
    tools.kMeans(data,90,9**3,2)    

if AggHierarchy == True:           #Agg cluster plot at a given time, input form (data,Nt,axisN**3,number of clusters)
    tools.AggHierarchy(data,90,9**3,2)  
    
if BestClusterNumber == True:      #Gives the number of clusters that best fits the data using silhoutte score and plots
    tools.BestClusterNumber(data,90,9**3)    