from matplotlib import pyplot as plt
import numpy as np
from scipy.linalg import eig

from chebdiff import chebdiff


class EigValueProblem():
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

        self.sW = None
        self.sT = None
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
        D1 = self.DM[0]
        D2 = self.DM[1]
        I = np.eye(N)

        C1 = np.zeros((3, 2*N))
        if (bc1.lower() == 'free'):
            C1[0, 0:N] = I[0, :]  # w = 0 at z = 0
            C1[1, 0:N] = D2[0, :]  # D2w = 0 at z = 0
            C1[2, N:2*N] = I[0, :]  # T = 0 at z = 0
        elif (bc1.lower() == 'rigid'):
            C1[0, 0:N] = I[0, :]  # w = 0 at z = 0
            C1[1, 0:N] = D1[0, :]  # D2w = 0 at z = 0
            C1[2, N:2*N] = I[0, :]  # T = 0 at z = 0
        else:
            raise Exception('bc1 = %s not coded' % bc1)
        rr1 = np.array([0, 1, N])

        C2 = np.zeros((3, 2*N))
        if (bc2.lower() == 'free'):
            C2[0, 0:N] = I[N-1, :]  # w = 0 at z = L
            C2[1, 0:N] = D2[N-1, :]  # D2w = 0 at z = L
            C2[2, N:2*N] = I[N-1, :]  # T = 0 at z = L
        elif (bc2.lower() == 'rigid'):
            C2[0, 0:N] = I[N-1, :]  # w = 0 at z = L
            C2[1, 0:N] = D1[N-1, :]  # D2w = 0 at z = L
            C2[2, N:2*N] = I[N-1, :]  # T = 0 at z = L
        else:
            raise Exception('bc1 = %s not coded' % bc1)
        rr2 = np.array([N-1, N-2, 2*N-1])

        self.C = np.vstack((C1, C2))
        self.rr = np.concatenate([rr1, rr2])
        self.kk = np.setdiff1d(np.arange(2*N), self.rr)

    def eval_constrained_mat(self):
        rr = self.rr
        kk = self.kk
        C = self.C
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

        s, WT_k = eig(A_k, B_k)
        
        n_eigvecs = np.shape(WT_k)[1]
        WT = np.zeros((2*N, n_eigvecs), dtype=np.complex)
        WT[self.kk, :] = WT_k
        WT[self.rr, :] = np.dot(self.G, WT_k)

        n = (int)(n_eigvecs/2)
        self.W = WT[0:N, :n]
        self.T = WT[N:, n:]
        self.sW = s[0:n]
        self.sT = s[n:]

        self.plot_mins_mode('W')
        self.plot_mins_mode('T')


    def plot_mins_mode(self, var):
        index = 0
        plt.figure()
        if var == 'W':
            index = np.argmin(np.abs(self.sW))
            plt.plot(self.x, np.real(self.W[:,index]))
            plt.plot(self.x, np.imag(self.W[:,index]))
            plt.title('%f+i%f mode plot of %s for %s - %s'
                    %(np.real(self.sW[index]),np.imag(self.sW[index]),
                    var, self.bc1, self.bc2))
        elif var == 'T':
            index = np.argmin(np.abs(self.sT))
            plt.plot(self.x, np.real(self.T[:,index]))
            plt.plot(self.x, np.imag(self.T[:,index]))
            plt.title('%f+i%f mode plot of %s for %s-%s'
                    %(np.real(self.sT[index]),np.imag(self.sT[index]),
                        var, self.bc1, self.bc2))
        plt.xlabel('x')
        plt.ylabel(var)

        plt.savefig('%s-%s_%s_plot.pdf'%(self.bc1, self.bc2, var))


def plotStreamLine():
    k = 2.221
    x = np.linspace(0, 2 * 2*np.pi/k, 100)
    z = np.linspace(0, 1, 100)
    X , Z = np.meshgrid(x, z)
    f = np.sin(np.pi*0.5)* np.sin(k*0.5) - np.sin(np.pi*Z)* np.sin(k*X)
    plt.figure()
    plt.contour(X, Z, f)   
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('z')
    plt.title('Streamlines for Rayleigh Benard free-free problem')
    plt.savefig("streamlines.pdf")
    pass

if __name__ == '__main__':
    rb1 = EigValueProblem(200, 0.5, 'free', 'free', 657.5, 7.56, 2.221)
    rb1.solve()

    rb1 = EigValueProblem(200, 0.5, 'rigid', 'rigid', 1708, 7.56, 3.117)
    rb1.solve()

    rb1 = EigValueProblem(200, 0.5, 'free', 'rigid', 1101, 7.56, 2.682)
    rb1.solve()
    
    rb1 = EigValueProblem(200, 0.5, 'rigid', 'free', 1101, 7.56, 2.682)
    rb1.solve()

    plotStreamLine()
