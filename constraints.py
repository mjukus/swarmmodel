# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 12:14:27 2020

@author: thesq
"""

import numpy as np
import matplotlib.pyplot as plt
import tools
    
def angleCon(pos,bondStiffness):
    #problem is, will have to do this for each set of three points in the rod
    particleTails = pos[:,0]
    particleHeads = pos[:,-1]
    centre = 0.5 * (particleHeads - particleTails)
    N = len(particleTails)
    
    rij = centre
    rik = 2 * centre
    rij_abs = np.linalg.norm(rij,axis=1)
    rik_abs = np.linalg.norm(rik,axis=1)
    rijrik = rij_abs*rik_abs
    rij2 = rij_abs*rij_abs
    rik2 = rik_abs*rik_abs
    rij2 = rij2[:,np.newaxis]
    rik2 = rik2[:,np.newaxis]
    
    costhetajik = np.diagonal(np.dot(rij,rik.T))/rijrik
    costhetajik = costhetajik[:,np.newaxis]
    rijrik = rijrik[:,np.newaxis]
    Force = np.zeros([N,3,3])
    i=1
    Force[:,i] = bondStiffness*((rik+rij)/rijrik-costhetajik*(rij/rij2+rik/rik2))
    Force[:,i-1] = bondStiffness*(costhetajik*rij/rij2-rik/rijrik)
    Force[:,i+1] = bondStiffness*(costhetajik*rik/rik2-rij/rijrik)
#    print(Force)
    return Force

def bondCon(pos,bondLength,nRod):
    particleTails = pos[:,0] # the positions of the two ends of the particles - the heads and tails
    particleHeads = pos[:,-1]
    centre = 0.5 * (particleHeads - particleTails)
    # the mean of the head and tail position relative to the tail - the centre of the particle
    centreMag = np.linalg.norm(centre, axis=1)
    # the magnitude of the vectors describing the middle of the particles
    
    bondDir = centreMag[:,np.newaxis]**-1 * centre
    # calculates the unit vectors describing particle direction using the centres
    
    pos = tools.bondVectorGen(particleTails,bondDir,bondLength,nRod)
    # calls tools.bondVectorGen, which generates an array of position vectors for all the points in all the
    # particles
    
    return pos