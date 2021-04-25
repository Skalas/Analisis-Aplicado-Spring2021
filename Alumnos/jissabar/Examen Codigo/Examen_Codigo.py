import numpy as np

b = 10
f = lambda x,y: (x-1)**2 + b*(y-x**2)**2

df = lambda x,y: np.array([2*(x-1) - 4*b*(y - x**2)*x,2 *b*(y-x**2)])

F = lambda X: f(X[0],X[1])

dF = lambda X: df(X[0],X[1])

x0 = np.array([-1.4,1.1])

fx = F(x0)

gx = dF(x0)

s = -gx

al = np.linspace(0,0.1,101)

z = [F(x0+a*s) for a in al]

theta = 0.1
alpha = 1
tol = 1e-10
d = theta*np.dot(gx,s)

for i in range(10):
    if F(x0+alpha*s) < (fx + alpha*d):
        break
    alpha = alpha/2




