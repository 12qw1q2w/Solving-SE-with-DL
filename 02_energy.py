import numpy as np

summation = lambda f, n0, N: np.sum(f(np.arange(n0, N+1)))

def integral(y, x):
    n = len(x) - 1
    dx = (x[-1] - x[0]) / n
    s1 = summation(lambda i: y[2*i-1], 1, n//2)
    s2 = summation(lambda i: y[4*i-2], 1, n//4)
    s3 = summation(lambda i: y[4*i], 1, n//4)
    return ( 28*y[0] - 28*y[-1] + 128*s1 + 48*s2 + 56*s3 ) * dx / 90

def energy(V, psi, x):
    psi1 = np.gradient(psi, x)
    psi2 = np.gradient(psi1, x)
    return integral( - psi2 * psi / 2 + V * psi * psi , x)

x = np.linspace(0, 1, 10**6+1)
V = (lambda x: np.zeros_like(x))(x)
psi = (lambda x: np.sqrt(2)*np.sin(np.pi*x))(x)

print(energy(V, psi, x))
print(np.pi**2/2)
