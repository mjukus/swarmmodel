# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 12:14:27 2020

@author: thesq
"""

import numpy as np
#import matplotlib.pyplot as plt
import tools
from numba import jit
    
#def angleCon(pos,bondStiffness):
#    # UNIMPLEMENTED. Not necessary, hopefully.
#    #problem is, will have to do this for each set of three points in the rod
#    particleTails = pos[:,0]
#    particleHeads = pos[:,-1]
#    centre = 0.5 * (particleHeads - particleTails)
#    N = len(particleTails)
#    
#    rij = centre
#    rik = 2 * centre
#    rij_abs = np.linalg.norm(rij,axis=1)
#    rik_abs = np.linalg.norm(rik,axis=1)
#    rijrik = rij_abs*rik_abs
#    rij2 = rij_abs*rij_abs
#    rik2 = rik_abs*rik_abs
#    rij2 = rij2[:,np.newaxis]
#    rik2 = rik2[:,np.newaxis]
#    
#    costhetajik = np.diagonal(np.dot(rij,rik.T))/rijrik
#    costhetajik = costhetajik[:,np.newaxis]
#    rijrik = rijrik[:,np.newaxis]
#    Force = np.zeros([N,3,3])
#    i=1
#    Force[:,i] = bondStiffness*((rik+rij)/rijrik-costhetajik*(rij/rij2+rik/rik2))
#    Force[:,i-1] = bondStiffness*(costhetajik*rij/rij2-rik/rijrik)
#    Force[:,i+1] = bondStiffness*(costhetajik*rik/rik2-rij/rijrik)
#    print(Force)
#    return Force

@jit(forceobj=True)
def bondCon(pos,bondLength,nRod,tumbleProb):
    N = len(pos)
    randNum = np.random.rand(N) # random numbers to compare with tumbleProb and decide whether to tumble
    randOrientation = np.random.rand(N,2)*2*np.pi # random orientations as 2 angles
    randBondDir = np.hstack((np.sin(randOrientation[:,1:2])*np.cos(randOrientation[:,0:1]), # produces an array f random bond directions
                        np.sin(randOrientation[:,1:2])*np.sin(randOrientation[:,0:1]),
                        np.cos(randOrientation[:,1:2])))
    
    particleTails, centre = tools.findCentre(pos)
    centreMag = np.linalg.norm(centre,axis=1)
    #centreMag = np.empty(N)
    #for i in range(N):
    #    centreMag[i] = np.linalg.norm(centre[i])
    centreMag = centreMag.reshape(-1,1)
    # the magnitude of the vectors describing the middle of the particles
    bondDir = centreMag**-1 * centre
    # calculates the unit vectors describing particle direction using the centres
    for i in range(N):
        if randNum[i] <= tumbleProb:
            bondDir[i] = randBondDir[i] # new bondDir for particles which pass probability check
    particleTails += centre - (0.5 * bondLength * (nRod-1) * bondDir)
    # finds the new positions of the particle tails post-constraint
    pos = tools.bondVectorGen(particleTails,bondDir,bondLength,nRod)
    # calls tools.bondVectorGen, which generates an array of position vectors for all the points in all the
    # particles
    
    return pos, bondDir
