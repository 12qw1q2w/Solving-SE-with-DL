import numpy as np
import matplotlib.pyplot as plt

N = 100

def tensor_sum(f, n0, N):
    s = np.zeros(f(n0).shape)
    for n in range(n0, N+1):
        s += f(n)
    return s

def random_fourier(x, nmax=np.random.randint(0, 32)):
    a = np.random.random(nmax+1)-0.5
    b = np.random.random(nmax+1)-0.5
    f = lambda x: tensor_sum(lambda n: a[n]*np.sin(n*np.pi*x) + b[n]*np.cos(n*np.pi*x), 0, nmax)
    return f(x)

x = np.linspace(0, 1, N+1)
plt.plot(x, random_fourier(x))
plt.show()
