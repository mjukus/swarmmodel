# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 09:43:12 2021

@author: thesq
"""

import numpy as np
import tools
#import matplotlib.pyplot as plt
#from numba import jit

def init(axisN,partAxisSep,nRod,bondLength,centre=0):
    '''
    This function generates starting positions and orientations of N rod-like particles.
    It begins by producing a 3D grid of N evenly-spaced particles, before generating a random
    orientation vector for each and attaching further points at a set bond length to these according
    to the orientation vectors to produce rods.

    Parameters
    ----------
    axisN : integer
        The number of particles on each axis of the cube. Cube root of the total number of particles in the system.
    partAxisSep : float
        The axial separation between two adjacent particles in the cube.
    nRod : integer
        The number of points in each rod-like particle.
    bondLength : float
        The length of each bond in a particle; the distance between each point in the particle.
    centre : float, optional
        The position of the centre of the cube in all axes. The default is 0.

    Returns
    -------
    pos : N x nRod x 3 array
        The positions of all the points in the particles in the system.

    '''
    N = axisN ** 3 #the number of rod-like particles in the system.
    
    gridWidth = (axisN - 1) * partAxisSep # width of the cube
    axisPos = np.linspace(centre-gridWidth/2,centre+gridWidth/2,axisN) # an array of particle positions on each axis
    grid = np.array(np.meshgrid(axisPos,axisPos,axisPos)).T.reshape(-1,3) # the full 3d cube of particle coordinates as an N x 3 array
    randOrientation = 2*np.pi*np.random.rand(N,2) # an N x 2 array of particle orientations [[theta1,phi1],...]
    
    bondDir = np.hstack((np.sin(randOrientation[:,1:2])*np.cos(randOrientation[:,0:1]),
                        np.sin(randOrientation[:,1:2])*np.sin(randOrientation[:,0:1]),
                        np.cos(randOrientation[:,1:2])))
        # an array containing the orientational vectors for each particle
    
    pos = tools.bondVectorGen(grid,bondDir,bondLength,nRod)
    # calls tools.bondVectorGen, which generates an array of position vectors for all the points in all the
    # particles
    
    #tools.plot(np.vstack(pos)[:,0:1],np.vstack(pos)[:,1:2],np.vstack(pos)[:,2:3])
    # plots the starting state of the system in 3 dimensions
    
    return pos, bondDir