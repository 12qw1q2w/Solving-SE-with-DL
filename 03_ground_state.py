import time
initial_time = time.time()
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

steps = 2000
step = 0
N = 128 # The number of the samples.

psis = np.zeros((steps+1, N+1))
energys = np.zeros(steps+1)

# summation(f, n0, N) = f(n0) + f(n0+1) + ... + f(N)
def summation(f, n0, N):
    s = 0
    for i in range(n0, N+1):
        s += f(i)
    return s

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
    return ((y[0] - y[-1]) / 2 + summation(lambda i: y[i], 1, n)) * dx

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
    return integral(psi*(-psi2/2 + V*psi), x)

def update(step):
    print(step, energys[step])
    line0.set_ydata(psis[step])
    fig.canvas.draw()

def press(event):
    global step
    if step > 0 and event.key == '-':
        if step <= 10:
            step -= 1
        elif step <= 100:
            step -= 10
        elif step < 10000:
            step -= 100
        else:
            step -= 1000
        update(step)
    elif step < steps and event.key == '+':
        if step < 10:
            step += 1
        elif step < 100:
            step += 10
        elif step < 10000:
            step += 100
        else:
            step += 1000
        update(step)

x = np.linspace(0, 1, N+1)                  # The array of the variables.
V = (lambda x: np.zeros_like(x))(x)         # The array of the potential funcion.
#psi = tf.Variable(np.random.rand(len(x)))   # The initial wave function: random numbers on [0, 1]
psi = tf.Variable(np.ones(len(x)))
Energy = energy(V, psi, x)                  # The energy.

optimizer = tf.train.GradientDescentOptimizer(learning_rate=2**-8)  # The optimizer
bc0 = tf.assign(psi[0], 0)                                          # The boundary condition psi(0) = 0
bc1 = tf.assign(psi[-1], 0)                                         # The boundary condition psi(1) = 0
norm = tf.assign(psi, psi / tf.sqrt(integral(psi*psi, x)))          # Normalization
train = optimizer.minimize(Energy)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(steps+1):
    sess.run([train, bc0, bc1, norm])
    psis[step] = sess.run(psi)
    energys[step] = sess.run(Energy)
    if step % 10 == 0:
        print(step, energys[step])

fig = plt.figure(figsize=(12.8, 6.4))           # 그림 크기 설정
fig.canvas.mpl_connect('key_press_event', press)

grid_size = [1, 2]                              # 화면을 1행 2열의 격자로 나눈다.

axis00 = plt.subplot2grid(grid_size, (0, 0))    # (0, 0) 위치
plt.title('Wave Function')                      # 첫번째 그림 제목
plt.xlabel('x')                                 # 첫번째 그림 가로축 이름
plt.ylabel('wave function')                     # 두번째 그림 세로축 이름

axis01 = plt.subplot2grid(grid_size, (0, 1))    # (0, 1) 위치
plt.title('Energy')                             # 두번째 그림 제목
plt.xlabel('steps')                             # 두번째 그림 가로축 이름
plt.ylabel('energy')                            # 두번째 그림 세로축 이름

# 이미 알고 있는 파동함수, 검은색 선
axis00.plot(x, np.sqrt(2)*np.sin(np.pi*x), color='k', linewidth=0.5, label='True wave function')

# 프로그램이 구한 파동함수, 빨간색 점
line0, = axis00.plot(x, psis[step], color='r', lw=0.5, label='The wave function found by the program')

# 이미 알고 있는 에너지, 검은색 선
axis01.plot([0, steps], [np.pi**2/2, np.pi**2/2], lw=0.5, color='k', label = 'True energy')
axis01.text(steps, np.pi**2/2, f'{np.pi**2/2}', ha='center', va='top')

# 프로그램이 구한 에너지, 빨간색 점
axis01.scatter(range(steps+1), energys, s=1, color='r', label = 'The energy found by the program')
untitled0 = axis01.annotate(f'{energys[-1]}', xytext = (step, energys[-1]+(energys[0]-energys[-1])/10),
    xy = (step, energys[-1]), color='r', ha = 'center', arrowprops={'facecolor':'w', 'edgecolor':'r'})

axis00.legend(loc='lower center')   # 첫번째 그림 범례
axis01.legend(loc='upper right')    # 두번째 그림 범례

final_time = time.time()
time_taken = divmod(final_time - initial_time, 60)
print('Timme taken:', int(time_taken[0]), 'min', int(time_taken[1]), 's')

plt.show()
