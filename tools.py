# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 04:15:23 2021
@author: mawga
"""
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import matplotlib.pyplot as plt
from numba import jit

@jit(forceobj=True) # np.tensordot not supported by numba. Pain to remove, not worth.
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

def findCentre(pos):
    particleTails = pos[:,0] # the positions of the two ends of the particles - the heads and tails
    particleHeads = pos[:,-1]
    centre = 0.5 * (particleHeads - particleTails)
    # the mean of the head and tail position relative to the tail - the centre of the particle
    
    return particleTails, centre

def plot(x,y,z=0):
    '''
    Simple 3D scatter plotting function.
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

def quiver (data,dirData,timestep=-1):
    ''' quiver plot innit '''
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    N = data.shape[2]
    data2 = np.moveaxis(data,3,2)
   
    dirData2 = np.sqrt(dirData**2)
    dirData2 = np.moveaxis(dirData,2,1)
    
    colour = np.zeros([N,3])
    for i in range (N):
        colour[i] = np.array([dirData2[0,i,0],dirData2[0,i,1],dirData2[0,i,2]])
        #colour[N+2*i] = np.array([dirData2[0,i,0],dirData2[0,i,1],dirData2[0,i,2]])  not working
        #colour[N+1+(2*i)] = np.array([dirData2[0,i,0],dirData2[0,i,1],dirData2[0,i,2]])  not working    
    colour[colour < 0 ] = 0.5 * colour[colour<0]
    colour = np.abs(colour) 
    
    ax.quiver(data2[timestep,0],data2[timestep,1],data2[timestep,2],dirData[timestep,0],dirData[timestep,1],dirData[timestep,2], colors=colour, length=5E-7)
    plt.show()
    
def crystal_order(dirData,Nt,N):  
    crystal_order_tensor = np.zeros((Nt,3,3)) #Sets up all necessary arrays
    eigenvalue_matrix = np.zeros((Nt,3))
    eigenvalue_matrix_max = np.zeros((Nt))
    eigenvector_matrix = np.zeros((Nt,3,3))
    dirData = np.moveaxis(dirData,2,1)
    for i in range (Nt):            # Calculates the order parameter at each timestep, diagonilizes the matrix
        for j in range (N):         # and finds the maximum eigenvalue  
            crystal_order_tensor[i] += ((3 * np.array([(dirData[i][j][0]*dirData[i][j][0], 
                                                    dirData[i][j][0]*dirData[i][j][1],
                                                    dirData[i][j][0]*dirData[i][j][2]),
                                                   (dirData[i][j][1]*dirData[i][j][0], 
                                                    dirData[i][j][1]*dirData[i][j][1],
                                                    dirData[i][j][1]*dirData[i][j][2]),
                                                   (dirData[i][j][2]*dirData[i][j][0], 
                                                    dirData[i][j][2]*dirData[i][j][1],
                                                    dirData[i][j][2]*dirData[i][j][2])
                                                 ])) - (np.identity(3)))
        
        crystal_order_tensor = crystal_order_tensor / (2*N) 
       
        eigenvalue_matrix[i], eigenvector_matrix[i] = np.linalg.eig(crystal_order_tensor[i]) 
        
        eigenvalue_matrix_max[i] = np.amax(eigenvalue_matrix[i])        
        
     
    fig, ax = plt.subplots()
    t = np.arange(0.0, Nt, 1)
    ax.plot(t, eigenvalue_matrix_max)  
    
    ax.set(xlabel='timestep', ylabel='Order parameter',
       title='Order of whole swarm over time')
    ax.grid()
    plt.show()                                                     
    return eigenvalue_matrix,eigenvector_matrix     