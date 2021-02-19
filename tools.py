# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 00:48:17 2021

@author: thesq
"""

import numpy as np
import matplotlib.pyplot as plt

def bondVectorGen(grid,bondDir,bondLength,nRod):
    '''
    Generates an N x nRod x 3 array containing the positions of all points in all particles from a grid of
    particle starting positions, the bond directions, the bond length, and the number of points in the rod.

    Parameters
    ----------
    grid : N x 3 array
        The positions of the particles, specifically the starting point or tail of each particle.
    bondDir : N x 3 array
        Unit vectors describing the direction of the particles, and hence the bonds.
    bondLength : float
        The length of each bond in a particle.
    nRod : integer
        The number of points in a particle.

    Returns
    -------
    pos : N x nRod x 3 array
        DESCRIPTION.

    '''
    N = len(bondDir) # the total number of rod-like particles is taken from the length of bondDir
    bondVector = bondLength * bondDir # bond vectors are calculated
    
    allBonds = np.stack(np.tensordot(np.linspace(0,nRod-1,nRod),bondVector,0),1)
    # this array contains the displacements of all the points in the rod from the first point generated
    # above.
    stack = np.repeat([grid],nRod,axis=1).reshape(N,nRod,3)
    pos = stack + allBonds
    # adds the bond lengths to the grid of particles to produce an N x nRod x 3 array describing
    # the positions of each interacting point in each particle
    
    return pos

def plot(x,y,z=0):
    '''
    Simple 3D scatter plotting function. Hope to replace/improve with an animated thing.

    Parameters
    ----------
    x : array
    y : array
    z : array, optional
        The default is 0.

    Returns
    -------
    Produces a 3D scatter plot of x, y and z.

    '''
    plotFig = plt.figure()
    plotAx = plotFig.add_subplot(111,projection="3d")
    plotAx.scatter(x,y,z)