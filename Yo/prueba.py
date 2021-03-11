#Aqui voy a hacer mis notas

import numpy as np 
import matplotlib
import matplotlib.pyplot as plt

def f(x):
    return(np.exp((-1)/(x**2)))

t = np.linspace(-2, 2, 100)

# Data for plotting
t = np.arange(0.0, 2.0, 0.01)
s = 1 + np.sin(2 * np.pi * t)

fig, ax = plt.subplots()
ax.plot(t, s)

ax.set(xlabel='time (s)', ylabel='voltage (mV)',
       title='About as simple as it gets, folks')
ax.grid()

fig.savefig("test.png")
plt.show()