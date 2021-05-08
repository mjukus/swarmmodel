# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 16:23:21 2021

@author: thesq
"""

import numpy as np
import tools
import animation
import matplotlib.pyplot as plt

outputDirectory = "Output/2021-05-08_125_4_1000000/" # choose the directory in which the output files are to be found in format "path/to/file/from/code/directory/"
quiver = False
animate = False
Crystal_order = True
Histogram = False
kMeans = False
AggHierarchy = False
BestClusterNumber = True
trajectory = True

data = np.load(f"{outputDirectory}positions.npy") # loads in position and direction data from file
dirData = np.load(f"{outputDirectory}directions.npy")

if quiver == True:                 #Creates Quiver plot, input form (data,dirData,Nt)
    tools.quiver(data,dirData,100)
    
if animate == True:         
    animation.main(data.reshape(data.shape[0],data.shape[1],-1))
          

if Crystal_order == True:          #Creates plot of crystal order over time, input form (dirData,Nt,axisN**3)
    tools.crystal_order(dirData,10000,5**3)   
    
if Histogram == True:              #Creates a histogram of pos at a given time, input form (data,Nt,axisN**3,nRod)
    tools.histogram(data,99,10**3,4)    

if kMeans == True:                 #Kmeans cluster plot at a given time, input form (data,Nt,axisN**3,number of clusters)
    tools.kMeans(data,10000,5**3,2)    

if AggHierarchy == True:           #Agg cluster plot at a given time, input form (data,Nt,axisN**3,number of clusters)
    tools.AggHierarchy(data,90,9**3,2)  
    
if BestClusterNumber == True:      #Gives the number of clusters that best fits the data using silhoutte score and plots
    tools.BestClusterNumber(data,10000,5**3)

if trajectory == True:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.set_zlabel("z / m")
    for i in range(data.shape[2]):
        data_reshape = data[:,:,i,1].reshape((data.shape[0],data.shape[1]))
        ax.plot(data_reshape[:,0],data_reshape[:,1],data_reshape[:,2])