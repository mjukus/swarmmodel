# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 12:14:27 2020

@author: thesq
"""

import numpy as np
import matplotlib.pyplot as plt

'''
KEY PARAMETERS
--------------
These parameters are used by almost all the functions in the program and define the system.
'''

centre = 0 # the centre position of the cube of N particles
axisN =  2 # the number of particles on each axis of the cube (recommended odd)
N = axisN ** 3 # total number of particles N
partAxisSep = 5 # the axial separation between two adjacent particles in the cube

rodLength = 1 # rod length
nRod = 5 # number of interaction points in rod. Must be greater than 1
bondLength = rodLength/nRod # bond length between adjacent points in rod

partMass = 1 # the mass of each whole rod-like particle


def init():
    '''
    This function generates starting positions and orientations of N rod-like particles.
    It begins by producing a 3D grid of N evenly-spaced particles, before generating a random
    orientation vector for each and attaching further points at a set bond length to these according
    to the orientation vectors to produce rods. Finally a plot is produced of initial state of the
    system.

    Returns
    -------
    pos : N x nRod x 3 array
        The initial positions of every point on every particle in the system.

    '''
    
    gridWidth = (axisN - 1) * partAxisSep # width of the cube
    axisPos = np.linspace(centre-gridWidth/2,centre+gridWidth/2,axisN) # an array of particle positions on each axis
    grid = np.array(np.meshgrid(axisPos,axisPos,axisPos)).T.reshape(-1,3) # the full 3d cube of particle coordinates as an N x 3 array
    
    randOrientation = 2*np.pi*np.random.rand(N,2) # an N x 2 array of particle orientations [[theta1,phi1],...]
    
    bondDir = np.hstack((np.sin(randOrientation[:,1:2])*np.cos(randOrientation[:,0:1]),
                        np.sin(randOrientation[:,1:2])*np.sin(randOrientation[:,0:1]),
                        np.cos(randOrientation[:,1:2])))
        # an array containing the orientational vectors for each particle
    
    bondVector = bondLength * bondDir # an array containing the bond vectors for each particle
    allBonds = np.stack(np.tensordot(np.linspace(0,nRod-1,nRod),bondVector,0),1)
    # this array contains the displacements of all the points in the rod from the first point generated
    # above
    stack = np.repeat([grid[:]],nRod,axis=1).reshape(N,nRod,3)
    pos = stack + allBonds
    # adds the bond lengths to the cubic grid of particles to produce an N x nRod x 3 array describing
    # the starting positions of each interacting point in each particle
    
    plotFig = plt.figure()
    plotAx = plotFig.add_subplot(111,projection="3d")
    plotAx.scatter(np.vstack(pos)[:,0:1],np.vstack(pos)[:,1:2],np.vstack(pos)[:,2:3])
    # plots all the lads
    
    return pos

def lincs(oldPos,newPos,truncOrder=1):
    
    S = np.sqrt(partMass)/np.sqrt(2)
    #Sdiag from the literature. Rather than a diagonal matrix, it is constant when all partMass are the
    #same, as they are
    K = nRod - 1 #number of constraints; one less than number of particles (for a rod)
    
    B = np.zeros([N,K,3]) #will be an array of the directions of the constraints; zeros now
    for i in range(K):
        #we don't like for loops, but this one saved me a headache
        B[:,i] = oldPos[:,i+1] - oldPos[:,i]
    B = B / bondLength #Gets direction from the vectors above. It may be that this becomes poor after
    #multiple iterations, because it uses the original bond length parameter. There is an "easy" solution
    
    coef = 0.5 #a prefactor, greatly simplified because rod
    A = np.zeros([N,K,K]) #the normalised constraint coupling matrix. I - B * B^T
    for i in range(N):
        Bmult = B[i] @ B[i].T #matrix product of B and its transpose
        A[i] = np.identity(K) - (coef * Bmult) #fills A with coupling coefficients following A = 
        '''THE FUDGE ZONE - A should have zeros on the diagonal and be zero where coupling between
        unconnected bonds. The above fails in both these respects.'''
        np.fill_diagonal(A[i],0) #fudge to make the diagonal zero
        for j in range(2,K):
            #for all diagonals except the main diagonal and first offset diagonals
            np.fill_diagonal(A[i,j:],0) #fills the jth offset diagonal with zeros
            np.fill_diagonal(A[i,:,j:],0) #in both directions
        #rhs[1,i] = S * (B[i] @ newBondVector - bondLength)
    print(A)
    
    def solve():
        #this function will do the actual legwork after the normalised constraint coupling matrix A
        #has been generated. It does not yet exist.
        w = 2
        for rec in range(truncOrder):
            for i in range(K):
                placeholder = True
pos = init()
lincs(pos,pos)