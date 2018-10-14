import numpy as np
import matplotlib.pyplot as plt

N = 100

def summation(f, n0, N):
    s = 0
    for n in range(n0, N+1):
        s += f(n)
    return s

def random_fourier(x, nmax):
    a = np.random.random(nmax+1)-0.5
    b = np.random.random(nmax+1)-0.5
    return summation(lambda n: a[n]*np.sin(n*np.pi*x) + b[n]*np.cos(n*np.pi*x), 0, nmax)

x = np.linspace(0, 1, N+1)

while True:
    plt.plot(x, random_fourier(x, np.random.randint(0, 32)))
    plt.show()
