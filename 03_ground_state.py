import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

steps = 10000
N = 128 # The number of the samples.

# summation(n0, N, f) = f(n0) + f(n0+1) + ... + f(N)
summation = lambda n0, N, f: sum([f(n) for n in range(n0, N+1)])

'''
derivative(f, x):
    It returns the array of the derivatives.
    f: An array of the value of the function.
    x: An array of the variables.
'''
def derivative(f, x):
    n = len(x) - 1
    h = (x[-1] - x[0]) / n
    f1 = [0 for i in range(n+1)]
    f1[0] = ( f[1] - f[0] ) / h
    for i in range(1, n):
        f1[i] = (f[i+1] - f[i-1]) / (2*h)
    f1[n] = ( f[-1] - f[-2] ) / h
    return np.array(f1)

'''
integral(y, x):
    It returns the integral value.
    y: The array of the value of a function.
    x: The array of the variables.
'''
def integral(y, x):
    n = len(x) - 1
    dx = (x[-1] - x[0]) / n
    return ((y[0] - y[-1]) / 2 + summation(1, n, lambda i: y[i])) * dx

'''
energy(V, psi, x):
    It returns the quantum mechanical energy of the particle.
    We assume that the mass of the particle and the Dirac's constant are 1.
    V  : The array of the values of the potential function.
    psi: The array of the values of the wave function.
    x  : The array of the variables(position).
'''
def energy(V, psi, x):
    psi = [psi[i] for i in range(len(x))]
    psi1 = derivative(psi, x)
    psi2 = derivative(psi1, x)
    return integral(psi*(- psi2*psi/2+V*psi), x)

x = np.linspace(0, 1, N+1)                  # The array of the variables.
V = (lambda x: np.zeros_like(x))(x)         # The array of the potential funcion.
psi = tf.Variable(np.random.rand(len(x)))   # The initial wave function: random numbers on [0, 1]
#psi = tf.Variable(np.ones(len(x)))
Energy = energy(V, psi, x)                  # The energy.

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.004)  # The optimizer
bc0 = tf.assign(psi[0], 0)                                          # The boundary condition psi(0) = 0
bc1 = tf.assign(psi[-1], 0)                                         # The boundary condition psi(1) = 0
norm = tf.assign(psi, psi / tf.sqrt(integral(psi*psi, x)))          # Normalization
train = optimizer.minimize(Energy)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

energys = np.zeros(steps+1)

for step in range(steps+1):
    sess.run([bc0, bc1, norm, train])
    energys[step] = sess.run(Energy)
    if step % 10 == 0:
        print(step, sess.run(Energy))

fig = plt.figure(figsize=(12.8, 6.4))           # 그림 크기 설정
grid_size = [1, 2]                              # 화면을 1행 2열의 격자로 나눈다.

axis00 = plt.subplot2grid(grid_size, (0, 0))    # (0, 0) 위치
plt.title('Wave Function')                      # 첫번째 그림 제목
plt.xlabel('x')                                 # 첫번째 그림 가로축 이름
plt.ylabel('wave function')                     # 두번째 그림 세로축 이름

axis01 = plt.subplot2grid(grid_size, (0, 1))    # (0, 1) 위치
plt.title('Energy')                             # 두번째 그림 제목
plt.xlabel('steps')                             # 두번째 그림 가로축 이름
plt.ylabel('energy')                            # 두번째 그림 가로축 이름

# 이미 알고 있는 파동함수, 검은색 선
axis00.plot(x, np.sqrt(2)*np.sin(np.pi*x),
            color='k',
            linewidth=0.5,
            label = 'True wave function'
            )

# 프로그램이 구한 파동함수, 빨간색 점
axis00.plot(x, sess.run(psi),
               color='r',
               lw=0.5,
               label = 'The wave function found by the program'
            )
axis00.scatter(x, sess.run(psi),
               color='r',
               s=1
               )

# 이미 알고 있는 에너지, 검은색 선
axis01.plot([0, steps], [np.pi**2/2, np.pi**2/2], lw=0.5, color='k', label = 'True energy')
axis01.text(steps, np.pi**2/2, f'{np.pi**2/2}', ha='center', va='top')

# 프로그램이 구한 에너지, 빨간색 점
axis01.scatter(range(steps+1), energys, s=1, color='r', label = 'The energy found by the program')
axis01.annotate(
    f'{energys[-1]}',
    xytext = (steps, energys[-1]+(energys[0]-energys[-1])/10),
    xy = (steps, energys[-1]),
    color='r',
    ha = 'center',
    arrowprops={'facecolor':'w', 'edgecolor':'r'}
    )

axis00.legend(loc='lower center')   # 첫번째 그림 범례
axis01.legend(loc='upper right')    # 두번째 그림 범례

plt.show()
