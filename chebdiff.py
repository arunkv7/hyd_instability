# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 12:45:26 2017

@author: ani
"""

import math
import numpy as np
from scipy.linalg import toeplitz

def chebdiff(N,M):
    #  The function DM =  chebdif(N,M) computes the differentiation 
    #  matrices D1, D2, ..., DM on Chebyshev nodes. 
    # 
    #  Input:
    #  N:        Size of differentiation matrix.        
    #  M:        Number of derivatives required (integer).
    #  Note:     0 < M <= N-1.
    #
    #  Output:
    #  DM:       DM(1:N,1:N,ell) contains ell-th derivative matrix, ell=1..M.
    #
    #  The code implements two strategies for enhanced 
    #  accuracy suggested by W. Don and S. Solomonoff in 
    #  SIAM J. Sci. Comp. Vol. 6, pp. 1253--1268 (1994).
    #  The two strategies are (a) the use of trigonometric 
    #  identities to avoid the computation of differences 
    #  x(k)-x(j) and (b) the use of the "flipping trick"
    #  which is necessary since sin t can be computed to high
    #  relative precision when t is small whereas sin (pi-t) cannot.
        
    #  J.A.C. Weideman, S.C. Reddy 1998.

    n1 = int(math.floor(N/2.)) # Index used for flipping trick
    n2 = int(math.ceil(N/2.)) # Index used for flipping trick
    k = np.arange(N).reshape(N,1) #theta vector
    th = k*math.pi/(N-1)

    #   'N' Chebyshev collocation points are obtained by cosines of 
    #   uniformly-spaced points:
    #       x_k = cos((k-1)pi/(N-1)),  k = 1, ... , N.
    #   To ensure symmetry in spite of round-off errors, WR00 suggest using
    #   the following trig identity:
    #       cos(th) = sin(pi/2-th).
    x = np.sin(math.pi*np.arange(N-1,-1-N,-2)/(2.*(N-1))); #Chebyshev points

    #   Z contains entries 1/(x(k)-x(j)) with zeros on the diagonal
    T = np.tile(th/2.,N)    #Start off with square matrix T
    DX = 2*np.sin(T.T+T)*np.sin(T.T-T) # Use trig identity to find x(k)-x(j)
    DX[n1:,:] = -np.flipud(np.fliplr(DX[:n2,:])) # Flipping trick for accuracy
    np.fill_diagonal(DX,1.)  # Put 1's on the main diagonal of DX
    Z = 1./DX;  np.fill_diagonal(Z,0.)  #Finalize Z as 1/DX with 0's on diagonal

    #   C is the matrix with entries c(k)/c(j)
    C = toeplitz((-1.)**k)
    C[0,:] = C[0,:]*2;		C[-1,:] = C[-1,:]*2
    C[:,0] = C[:,0]/2;		C[:,-1] = C[:,-1]/2

    DM = []
    D = np.eye(N) # D contains diff. matrices
    for ell in range(M):
        diagD = np.diag(D).reshape(N,1)
        D = (ell+1)*Z*(C*np.tile(diagD,N) - D) # Off-diagonals
        np.fill_diagonal(D,-np.sum(D.T,axis=0))
        DM.append(D)

    return x.reshape(N), DM
