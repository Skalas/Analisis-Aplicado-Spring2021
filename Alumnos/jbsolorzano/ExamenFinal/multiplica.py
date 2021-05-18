import numpy as np


def multiplica(A,x):
    [n,m]= np.asarray(A.shape)
    Ax = np.asarray(np.zeros(n))
    for i in range(n):
        j = 1
        while j < m and abs(A[i,j+1]) < 1e-15:
            Ax[i] = Ax[i] + A[i,j] * x[ int(A[i,j+1])]
            j += 2
    return Ax




if __name__ == '__main__':
    n=15
    A = np.matrix(np.eye(n)).T
    b = np.zeros(n)+1
    x0 = np.matrix(np.ones(n)).T
    print(multiplica(A,x0))
