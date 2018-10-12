import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

N = 100

summation = lambda n0, N, f: sum([f(n) for n in range(n0, N+1)])

def derivative(f, x):
    n = len(x) - 1
    h = (x[-1] - x[0]) / n
    f1 = [0 for i in range(n+1)]
    f1[0] = ( f[1] - f[0] ) / h
    for i in range(1, n):
        f1[i] = (f[i+1] - f[i-1]) / (2*h)
    f1[n] = ( f[-1] - f[-2] ) / h
    return np.array(f1)

def integral(y, x):
    n = len(x) - 1
    dx = (x[-1] - x[0]) / n
    return ((y[0] - y[-1]) / 2 + summation(1, n, lambda i: y[i])) * dx

def energy(V, psi, x):
    psi = [psi[i] for i in range(len(x))]
    psi1 = derivative(psi, x)
    psi2 = derivative(psi1, x)
    return integral(psi*(- psi2*psi/2+V*psi), x)

x = np.linspace(0, 1, N+1)
V = (lambda x: np.zeros(len(x)))(x)
psi = tf.Variable(np.ones(len(x)))
Energy = energy(V, psi, x)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1/(2*N))
train = optimizer.minimize(Energy)
boundary_condition_0 = tf.assign(psi[0], 0)
boundary_condition_1 = tf.assign(psi[-1], 0)
normalization = tf.assign(psi, psi / tf.sqrt(integral(psi*psi, x)))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(1001):
    sess.run(boundary_condition_0)
    sess.run(boundary_condition_1)
    sess.run(normalization)
    sess.run(train)
    if step % 10 == 0:
        print(step, sess.run(Energy))
    if step % 100 == 0:
        plt.plot(x, np.sqrt(2)*np.sin(np.pi*x))
        plt.plot(x, sess.run(psi))
        plt.show()

print(sess.run(Energy))
print(np.pi**2/2)
