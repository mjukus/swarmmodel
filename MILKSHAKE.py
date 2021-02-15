# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 16:28:50 2020

@author: mawga
"""
import numpy as np
import matplotlib as plt
import scipy 
from numba import jit, f8

## Tri Diagonal Matrix Algorithm(a.k.a Thomas algorithm) solver

def TDMAsolver(a, b, c, d):
    '''
    TDMA solver, a b c d can be NumPy array type or Python list type.
    refer to http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    and to http://www.cfd-online.com/Wiki/Tridiagonal_matrix_algorithm_-_TDMA_(Thomas_algorithm)
    '''
    nf = len(d) # number of equations
    ac, bc, cc, dc = map(np.array, (a, b, c, d)) # copy arrays
    for it in range(1, nf):
        mc = ac[it-1]/bc[it-1]
        bc[it] = bc[it] - mc*cc[it-1] 
        dc[it] = dc[it] - mc*dc[it-1]
        	    
    xc = bc
    xc[-1] = dc[-1]/bc[-1]

    for il in range(nf-2, -1, -1):
        xc[il] = (dc[il]-cc[il]*xc[il+1])/bc[il]

    return xc
# VARIABLES
# NS,NL                 Number of sites, links
# TOL                   The predefined error
# M (NS)                Masses of the sites
# DSQ (NL)              Squared link lengths
# DT                    Time step
# F,V,R (NS,3)          Forces, velocities, positions
# DONE                  Indicates when the tolerance has been achieved
# A,B,IT                Indices
# ERROR                 Calculated error
# MAX                   Maximum error
# RIJ (NL,3)            Link vectors using R, defined R(I)-R(J)
# P (NS,3)              Updated positions prior to constraints
# RNCIJ (NL,3)          Link vectors using P
# N (NS,3)              Current approximation to constrained positions
# NIJ (NL,3)            Link vectors using N
# RIJ2,RNCIJ2,NIJ2 (NL) Link vectors squared
# DL,DU (NL-1)          Subdiagonal and superdiagonal vectors of the Jacobian
# D (NL)                Diagonal vector of the Jacobian
# LAMBDA (NL)           Undetermined Lagrange multipliers
# RHS,RHSOLD (NL)       Constraint conditions

# FUNCTIONS
#   SQRT(...)           Calculates the square root
#   DOT(VECT1,VECT2)    Calculates the inner product of VECT1 and VECT2
#   MU(M1,M2)           Calculates the reduced mass of M1 and M2
#   TRIDIAG(...)        Solves the tridiagonal matrix

# Values used will be based on a 3 site, 2 link chain, link length 0.6E-6 and centred around 0 on axis
# Verlet integration will be used to find the updated constrained positions
NS = 3                                         
NL = 2  
X = np.random.random() * 1E-15
Y = np.random.random() * 1E-15
Z = np.random.random() * 1E-15                                    
F = np.random.rand(3,NS) * 1E-15
V = np.array([[X,Y,Z],[X,Y,Z],[X,Y,Z]])     #All in same direction i.e swimming
R = np.array([[-0.6E-6,0,0],[0,0,0],[0.6E-6,0,0]])
P = np.array([[-0.6E-6,0,0],[0,0,0],[0.6E-6,0,0]])
TOL = 1E-8
M = 0.3E-15
MU = M/2
DSQ = np.array([3.6E-13,3.6E-13])
DT = 0.01
N = np.zeros([NS,3])
RIJ = np.zeros([NL,3])
RNCIJ = np.zeros([NL,3])
RIJ2 = np.zeros(NL)
RNCIJ2 = np.zeros(NL)
NIJ2 = np.zeros(NL)
D = np.zeros(NL)
DU = np.zeros(NL-1)
DL = np.zeros(NL-1)
RHS = np.zeros(NL)
RHSOLD = np.zeros(NL)

def MILCSHAKEa (R,V,F,DT,DSQ,M,TOL,NS,NL):
    
    #First step in Verlet integration
    
    for A in range (NS):
        P[A] = R[A] + (DT*V[A]) + (DT*DT/2) * (F[A]/M) 
        V[A] = V[A] + (DT/2) * (F[A]/M)
        #print(P)
        #print(V)
        
    #Set up matrix equation
            
    for A in range (NL):
        B = A + 1
        
        #link vectors using R (actual positions)
        RIJ[A] = R[A] - R[B]
        RIJ2[A] = np.dot(RIJ[A],RIJ[A])
        
        #link vectors using P (Unconstrained new positions)
        RNCIJ[A] = P[A] - P[B]
        RNCIJ2[A] = np.dot(RNCIJ[A],RNCIJ[A])
        
        #print(RNCIJ,RNCIJ2)
        #print (RNCIJ.shape)
        
    D[0] = 2 * (np.dot(RIJ[0],RNCIJ[0]) / MU)   
    DU[0] = -2 * (np.dot(RIJ[1],RNCIJ[0]) / M) 
    DL[0] = -2 * (np.dot(RIJ[0],RNCIJ[1]) / M)    
    #DO A =2, NL -1 is the same as [1] to [0] for us. Anyway A+1 term is zero thus term dissapears, same for NL-2
    D[1] = 2 * (np.dot(RIJ[1],RNCIJ[1]) / MU) 
    print(DL,D,DU)
    
    RHS = DSQ - RNCIJ2
    RHSOLD = RHS
    print (RHS)
    
    #iterative solution using tridiagonal matrix solver
    
    #DONE == false
    #IT == 1
    xc = TDMAsolver(DL, D, DU, RHS)
    

    
    return (P,V,RHS,xc)

    
    
                