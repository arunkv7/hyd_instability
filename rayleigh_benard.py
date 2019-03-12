# import pylab, time
import numpy as np
from scipy.linalg import eig

from chebdiff import chebdiff


class EigValue():
    def __init__(self, N, L, bc1, bc2, Ra, Pr, k):
        self.N = N
        self.L = L
        self.bc1 = bc1
        self.bc2 = bc2
        self.Ra = Ra
        self.Pr = Pr
        self.k = k

        self.A = None
        self.B = None
        self.C = None
        self.rr = None
        self.kk = None

        self.G = None
        self.A_k = None
        self.B_k = None

        self.s = None
        self.W = None
        self.T = None

        self.x, self.DM = chebdiff(N, 4)
        self.x = self.x / L
        for i in range(len(self.DM)):
            self.DM[i] = self.DM[i]/L**(i+1)

    def eval_AB(self):
        Ra = self.Ra
        Pr = self.Pr
        k = self.k
        N = self.N
        D2 = self.DM[1]
        D4 = self.DM[3]

        K2 = np.eye(N) * k**2
        K4 = np.eye(N) * k**4

        LHS00 = D4 + K4 - 2 * np.dot(D2, K2)
        LHS01 = -Ra * K2
        LHS10 = np.eye(N)
        LHS11 = D2 - K2

        self.A = np.vstack((np.hstack((LHS00, LHS01)),
                            np.hstack((LHS10, LHS11))))

        RHS00 = (D2 - K2)/Pr
        RHS01 = np.zeros_like(RHS00)
        RHS10 = np.zeros_like(RHS00)
        RHS11 = np.eye(N)

        self.B = np.vstack((np.hstack((RHS00, RHS01)),
                            np.hstack((RHS10, RHS11))))

    def setBC(self):
        N = self.N
        bc1 = self.bc1
        bc2 = self.bc2
        D2 = self.DM[1]
        I = np.eye(N)

        C1 = np.zeros((3, 2*N))
        if (bc1.lower() == 'free'):
            C1[0, 0:N] = I[0, :]  # w = 0 at z = 0
            C1[1, 0:N] = D2[0, :]  # D2w = 0 at z = 0
            C1[2, N:2*N] = I[0, :]  # T = 0 at z = 0
        else:
            raise Exception('bc1 = %s not coded' % bc1)
        rr1 = np.array([0, 1, N])

        C2 = np.zeros((3, 2*N))
        if (bc2.lower() == 'free'):
            C2[0, 0:N] = I[N-1, :]  # w = 0 at z = L
            C2[1, 0:N] = D2[N-1, :]  # D2w = 0 at z = L
            C2[2, N:2*N] = I[N-1, :]  # T = 0 at z = L
        else:
            raise Exception('bc1 = %s not coded' % bc1)
        rr2 = np.array([N-1, N-2, 2*N-1])

        self.C = np.vstack((C1, C2))
        print(self.C)
        self.rr = np.concatenate([rr1, rr2])
        self.kk = np.setdiff1d(np.arange(2*N), self.rr)

    def eval_constrained_mat(self):
        rr = self.rr
        kk = self.kk
        print(rr, kk)
        C = self.C
        print(C[:,rr])
        self.G = G = - np.linalg.solve(C[:, rr], C[:, kk])
        self.A_k = self.A[np.ix_(kk, kk)] + np.dot(self.A[np.ix_(kk, rr)], G)
        self.B_k = self.B[np.ix_(kk, kk)] + np.dot(self.B[np.ix_(kk, rr)], G)

    def solve(self):
        self.eval_AB()
        self.setBC()
        self.eval_constrained_mat()
        N = self.N
        A_k = self.A_k
        B_k = self.B_k

        self.s, WT_k = eig(A_k, B_k)

        WT = np.zeros((2*N, np.shape(WT_k)[1]), dtype=np.complex)
        WT[self.kk, :] = WT_k
        WT[self.rr, :] = np.dot(self.G, WT_k)

        self.W = WT[:, 0:N]
        self.T = WT[:, N:2*N]


if __name__ == '__main__':
    RB = EigValue(10, 0.5, 'free', 'free', 867., 0.02, 1.0)
    RB.solve()
    print(len(RB.s))
