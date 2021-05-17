import numpy as np
from numpy import linalg as la

class Optimizador:
    epsilon = 0.00001
    max_iter = 100000
    def __init__(self, epsilon = 0.00001, max_iter = 10000):
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.printInstrucciones()
    def printInstrucciones(self):
        print('Para optimizar una función usa el método optimiza, donde f representa la función a optimizar, dimensión\nrepresenta el tamaño del dominio de f, y Tipo representa el método a utilizar.')

        print('Tipo == 0 utilizará BFGS')
        print('Tipo == 1 utilizará una búsqueda lineal base')
        print('Tipo == 2 utilizará el método de Newton con modificación a la Hessiana')
        print('Tipo == 3 utilizará el método de Newton')
    def optimiza(self, f, dimension, tipo):
        x = np.zeros(dimension)
        if tipo == 0 : # BFGS
            return self.BFGS(f,x)
        elif tipo == 1: #Line Base
            return self.newton_line_base(f,x)
        elif tipo == 2: #Newton with modification
            return self.newton_mH(f,x)
        elif tipo == 3:
            return self.newton(f,x)
        else:
            print('\nTipo no valido')
            self.printInstrucciones()
    def derivada_parcial(self,f, xk, pos):
        eps = 0.0001
        n = xk.size
        h = np.zeros(n)
        h[pos] += eps
        return (f(xk + h) - f(xk)) / eps

    def derivada(self, f, xk):
        return (f(xk + self.epsilon) - f(xk))/self.epsilon

    def gradiente(self, f, xk):
        n = xk.size
        res = np.zeros(n)
        for i in range(n):
            res[i] = self.derivada_parcial(f, xk, i)
        return res

    def segunda_derivada(self, f, xk, pos1, pos2):
        n = xk.size
        h = np.zeros(n)
        h[pos2] += self.epsilon

        def f_prima(x):
            return self.derivada_parcial(f, x, pos1)

        return self.derivada_parcial(f_prima, xk, pos2)

    def hessiana(self, f, xk):
        n = xk.size
        res = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                res[i][j] = self.segunda_derivada(f, xk, i, j)
                res[j][i] = res[i][j]
        return res

    def is_pos_def(self, H):
        return np.all(np.linalg.eigvals(H) > self.epsilon)

    def condiciones_optimalidad(self, f, xk):
        if (np.all(self.gradiente(f, xk) >= self.epsilon)):
            return self.is_pos_def(self.hessiana(f, xk))
        return False

    def mk(self, f, xk, p):
        pt = p.transpose()
        return f(xk) + pt.dot(self.gradiente(f, xk)) + .5 * (pt.dot((self.hessiana(f, xk)).dot(p)))


    def backtracking_line_search(self, f, xk, p, c=.001, ro=.5):
        alpha = 1;
        f_k = f(xk)
        gr = self.gradiente(f, xk)
        while f(xk + alpha * p) > f_k + c * alpha * gr.dot(p):
            alpha *= ro;
        return alpha
    def make_pos_def(self,H):
        n = H.shape[0]
        Id = np.identity(n)
        t = self.epsilon
        while(not self.is_pos_def(H)):
            H = H + t*Id
            t= 2*t
        return H

    def zoom(self, a_low, a_high, phi, num=0, c1=0.0001, c2=0.9):
        a_mid = (a_high + a_low)/2
        if 20 < num:
            return a_mid
        if(phi(0) + c1*a_mid*(self.derivada(phi,0)) < phi(a_mid)) or (phi(a_low) <= phi(a_mid)):
            return self.zoom(a_low, a_mid, phi, num+1)
        else:
            if abs(self.derivada(phi, a_mid)) <= (-1)*c2*self.derivada(phi,0):
                return a_mid
            if 0 <= self.derivada(phi, a_mid)*(a_high - a_low):
                a_high = a_low
            return self.zoom(a_mid, a_high, phi, num+1)


    def BFGS(self, f, xk):
        #tomamos H0 simplemente como la identidad, como sugiere el Nocedal.
        n = xk.size
        Id = np.identity(n)
        Hk = Id
        gr = self.gradiente(f,xk)
        count = 0
        while self.epsilon < la.norm(gr) and count < self.max_iter:
            count = count +1
            print(count)
            pk = (-1)*Hk.dot(gr)
            al = self.backtracking_line_search(f,xk,pk)
            xk_n = xk + al*pk
            gr_n = self.gradiente(f,xk_n)
            sk = np.array([xk_n - xk])
            yk = np.array([gr_n-gr]) #hago yk una matriz de n*1
            yk = yk.T #transpongo para que sean vector columna
            sk = sk.T
            if ((yk.T).dot(sk))[0][0] < self.epsilon:
                ro = 1/self.epsilon
            else:
                ro = 1 /(((yk.T).dot(sk))[0][0])
            Hk = (Id + ((-1)*ro*sk).dot(yk.T)).dot(Hk).dot(Id + ((-1)*ro*yk).dot(sk.T)) + (ro*sk).dot(sk.T)
            xk = xk_n
            gr = gr_n
        return xk



    def newton_mH(self, f, x):
        eps = 0.0001
        n = 1000
        for i in range(self.max_iter):
            B = self.make_pos_def(self.hessiana(f, x))
            gr = self.gradiente(f, x)
            p = np.linalg.solve(B, -1 * gr)

            alpha = self.backtracking_line_search(f, x, p)
            x += alpha * p
        return x

    def newton (self, f, x):
        for i in range(self.max_iter):
            B = self.hessiana(f, x)
            gr = self.gradiente(f, x)
            p = np.linalg.solve(B, -1 * gr)
            alpha = self.backtracking_line_search(f, x, p)
            x += alpha * p
        return x

    def line_base(self, f, xk, p, c1=0.0001, c2=0.9):
        def phi(al):
           return f(xk + al*p)
        al1 = 0
        al_max = 100
        al2 = al_max /2
        for i in range (50):
            if (phi(0) + c1*al2*self.derivada(phi,0) < phi(al2))  or  (0 < i and phi(al1) <= phi(al2)):
                return self.zoom(al1, al2, phi)
            if abs(self.derivada(phi, al2)) <= (-1)*c2*self.derivada(phi,0):
                return al2
            if(0 <= self.derivada(phi,al2)):
                return self.zoom(al2, al1, phi)
            al1 = al2
            al2 = (al_max + al1)/2
        return al2

    def newton_line_base(self, f, x):
        eps = 0.0001
        n = 1000
        for i in range(self.max_iter):
            B = self.hessiana(f, x)
            gr = self.gradiente(f, x)
            p = np.linalg.solve(B, -1 * gr)
            alpha = self.line_base(f, x, p)
            x += alpha * p
        return x


o = Optimizador()
def rosenbrock(x, a = 1.5, b = 20):
    # x pertenece a R2
    return (a - x[0])**2 + b*(x[1] - x[0]**2)**2;

print(o.optimiza(rosenbrock,2,1))