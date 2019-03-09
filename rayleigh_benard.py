"""
Computes natural frequencies of a beam, formulated as an eigenvalue problem,
using Chebychev collocation method.
The code is writen in primitive variable y (vertical position fluctuations).

The eigenvalue problem is
    (D**4 + s**2)*Y = 0,
where
	D	:	differential operator with respect to position on beam, x
	s	:	temporal exponent
	Y	:	vertical position fluctuation normal mode

Constraints (hinged, clamped or free) can be applied at either boundary:
    Hinged b.c.  : y = D2y = 0 (no displacement or moment).
    Clamped b.c. : y = Dy = 0 (no displacement or slope).
    Free b.c.    : D2y = D3y = 0 (no moment or shear force).
These boundary constraints are used to remove certain degrees-of-freedom of Y.
DOFs removed are the first two from either end (4 b.c.'s in 4th order problem).
Let the resulting eigenvalue problem be denoted by:
    A0 * Y_k = s**2 * Y_k,
where
    Y_k = 'kept' portion of Y
    A0  = -D**4, actually a submatrix of it, with the constraint applied.

This particular quadratic eigenvalue problem is actually easily reformulated as
a linear one simply by defining
    S := s**2.
Then, the new (linear) eigenvalue problem is
    A0 * Y_k = S * Y_k.
This procedure is demonstrated in comments.

However, to demonstrate the procedure for linearizing arbitrary polynomial
eigenvalue problems, we take the standard approach:
    Let the problem be:     A0 * Y_k + s * A1 * Y_k + s**2 * A2 * Y_k = 0.
    Define:     Z_k = s * Y_k
    Then:       A * X_k = s * B * X_k, is the generalized eigenvalue problem
    where:
        X_k = [[Y_k], [Z_k]],
        A = [[0, I], [A0, A1]]
        B = [[I,0],[0,-A2]].
"""

import pylab, time
import numpy as np
from scipy.linalg import eig

from chebdiff import chebdiff

def BeamEig(N,L,bc1,bc2):
    
    """
    INPUTS:
    N : number of collocation nodes
    L : domain from -L to L
    bc1: boundary condition type at x = L in {'hinged', 'clamped' or 'free'}
    bc2: boundary condition type at x = -L (same scheme as bc1)
    
    OUTPUTS:
    S : 1d-array of leading-order eigenvalues
    Y : 2d-array of corresponding eigenvectors
    x : 1d-array of position along beam
    """

    #### Differentiation matrices
    x, DM = chebdiff(N,4)
    D1 = DM[0]; D2 = DM[1]; D3 = DM[2]; D4 = DM[3]

    #### Scale domain to [-L,L]
    scal = L;   x = x*scal
    D1 = D1/scal; D2 = D2/scal**2; D3 = D3/scal**3; D4 = D4/scal**4
    
    #### Eigenvalue problem kernel ('A0' in our notation)
    LHS = -D4
    
    #### Impose boundary conditions at x = L
    C1 = np.zeros((2,N))
    I = np.eye(N)
    if (bc1.lower() == 'hinged'):
        C1[0,:] = I[0,:]    #y = 0 at x = L
        C1[1,:] = D2[0,:]   #D2y = 0 at x = L
    elif (bc1.lower() == 'clamped'):
        C1[0,:] = I[0,:]    #y = 0 at x = L
        C1[1,:] = D1[0,:]   #Dy = 0 at x = L
    elif (bc1.lower() == 'free'):
        C1[0,:] = D2[0,:]   #D2y = 0 at x = L
        C1[1,:] = D3[0,:]   #D3y = 0 at x = L
    else:
        raise Exception('bc1 = %s not coded' % bc1)
    rr1 = np.array([0,1])   #Removed dofs

    #### Impose boundary conditions at x = -L
    C2 = np.zeros((2,N))
    if (bc2.lower() == 'hinged'):
        C2[0,:] = I[-1,:]   #y = 0 at x = L
        C2[1,:] = D2[-1,:]  #D2y = 0 at x = L
    elif (bc2.lower() == 'clamped'):
        C2[0,:] = I[-1,:]   #y = 0 at x = L
        C2[1,:] = D1[-1,:]  #Dy = 0 at x = L
    elif (bc2.lower() == 'free'):
        C2[0,:] = D2[-1,:]  #D2y = 0 at x = L
        C2[1,:] = D3[-1,:]  #D3y = 0 at x = L
    else:
        raise Exception('bc2 = %s not coded' % bc2)
    rr2 = np.array([N-1,N-2])   #Removed dofs
    
    #### Collate all boundary conditions
    C = np.vstack((C1,C2))                  #All constraints
    
    #### Collate all removed dofs, and determine kept dofs
    rr = np.concatenate([rr1,rr2])          #Removed dofs
    kk = np.setdiff1d(np.arange(N),rr)    #Kept dofs

    #### Give-back matrix, such that U_r = G*U_k, where U_r are removed dofs of
    #### solution U and U_k are its kept dofs
    G = - np.linalg.solve(C[:,rr],C[:,kk])

    #### Constrained system matrix
    LHS_k = LHS[np.ix_(kk,kk)] + np.dot(LHS[np.ix_(kk,rr)],G)
    
#    #### Solve for eigenvalues, noting that the problem is actually linear if
#    #### 's**2' is denoted by 'S'.
#    print 'computing eigenvalues ...'
#    t = time.time()
#    S, Y_k = eig(LHS_k)
#    s = np.sqrt(S)
#    print 'elapsed time is ',time.time() - t,' seconds'
    
    #### Solve for eigenvalues in linearized problem
    print('computing eigenvalues ...')
    t = time.time()
    nLHS_k = np.shape(LHS_k)[0] #Size of original eigenvalue problem (w/ b.c.'s)
    # Compose augmented kernel ('A' in our notation)
    aLHS_k = np.zeros((2*nLHS_k,2*nLHS_k))
    aLHS_k[:nLHS_k,(nLHS_k):] = np.eye(nLHS_k)
    aLHS_k[(nLHS_k):,:nLHS_k] = LHS_k
    s, aY_k = eig(aLHS_k)   #Solve augmented eigenvalue problem
    Y_k = aY_k[:nLHS_k,:]
    print('elapsed time is ',time.time() - t,' seconds')
    
    #### Remove very large eigenmodes (assuming that they are stable)
    ret = np.where((np.abs(s) < 200))[0]
    s = s[ret]
    Y_k = Y_k[:,ret]

    #### Retrieve full solution from its kept dofs
    Y = np.zeros((N,np.shape(Y_k)[1]),dtype=np.complex)
    Y[kk,:] = Y_k
    Y[rr,:] = np.dot(G,Y_k)
    
    return s, Y, x

# this is only accessed if running directly
if __name__ == '__main__':
    s, Y, x = BeamEig(100,0.5,'clamped','clamped')
    
    print ('Eigenvalues are: \n',s)

    pylab.figure();
    aa = np.argmin(np.abs(np.imag(s)))
    pylab.plot(x,np.real(Y[:,aa]),'-.')
    pylab.plot(x,np.imag(Y[:,aa]),'-')
    pylab.legend(['Yreal','Yimag']);
    pylab.xlabel('x');pylab.ylabel('Y')
    pylab.title('Eigenfunction for '+str(s[aa].real)+' '+str(s[aa].imag)+'j')

    pylab.figure();
    pylab.plot(x,np.real(Y),'-.')
    pylab.plot(x,np.imag(Y),'-')
    pylab.xlabel('x');pylab.ylabel('Y')
    pylab.title('All eigenfunctions')
    pylab.show() 
