# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 00:48:17 2021

@author: thesq
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import jit

@jit(forceobj=True) # np.tensordot not supported by numba. Pain to remove.
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

#@jit # jit is currently being a bit problematic
def separation(pos,N,nRod):
    
    x = pos[:,:,0:1].copy()
    x = x.reshape((N,nRod))
    dx = x.T - x[:,:,np.newaxis,np.newaxis] #creates 4D! tensors of x, y and z separations
    y = pos[:,:,1:2].copy()
    y = y.reshape((N,nRod))
    dy = y.T - y[:,:,np.newaxis,np.newaxis] #let me take a moment to apologise for my constant reshaping
    z = pos[:,:,2:3].copy()
    z = z.reshape((N,nRod))
    dz = z.T - z[:,:,np.newaxis,np.newaxis] #it cannot possibly be efficient
    
    for i in range(N):
        dx[i,:,:,i] = 0 # ensures that every point in a particle has zero separation
        dy[i,:,:,i] = 0 # from every other in the same particle.
        dz[i,:,:,i] = 0
    
    r = (dx**2 + dy**2 + dz**2)**0.5 # calculate magnitude of separations
    
    #dx = dx.flatten() # flatten so numba works
    #dy = dy.flatten()
    #dz = dz.flatten()
    #r = r.flatten()
    
    dx = dx[r != 0].reshape(N,nRod,nRod,N-1)
    dy = dy[r != 0].reshape(N,nRod,nRod,N-1)
    dz = dz[r != 0].reshape(N,nRod,nRod,N-1)
    r = r[r != 0].reshape(N,nRod,nRod,N-1) # remove all zeros to avoid nan in force
    
    sepDir = np.array([dx * r**-1, dy * r**-1, dz * r**-1]) # array of separation directions
    return r, sepDir

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
    
    plotAx.set_xlabel("x")
    plotAx.set_ylabel("y")
    plotAx.set_zlabel("z")
    
    plot = plotAx.scatter(x,y,z)
    
    return plotFig, plotAx, plot