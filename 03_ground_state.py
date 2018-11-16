import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from math import pi

N = 256
steps = 10000
step = 0
learning_rate = 2**-9

psis = np.zeros((steps+1, N+1))
energys = np.zeros(steps+1)

def derivative2(y, x):
    h = (x[-1] - x[0]) / (int(x.shape[0]) - 1)
    y2 = (y[2:] + y[:-2] - 2*y[1:-1]) / (h*h)
    y2 = tf.concat([[2*y2[0] - y2[1]], y2, [2*y2[-1] - y2[-2]]], 0)
    return y2

def integral(y, x):
    dx = (x[-1] - x[0]) / (int(x.shape[0]) - 1)
    return ((y[0] + y[-1])/2 + tf.reduce_sum(y[1:-1])) * dx

def energy(V, psi, x):
    psi2 = derivative2(psi, x)
    return integral(psi*((-0.5)*psi2 + V*psi), x)

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

x = tf.linspace(0.0, 1.0, N+1)
V = tf.zeros_like(x)
psi0 = tf.Variable(tf.ones(N-1))
#psi0 = tf.Variable(tf.random_uniform((N-1,), 0, 1))
psi = tf.concat([[0.0], psi0, [0.0]], 0)
E = energy(V, psi, x)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
norm = tf.assign(psi0, psi0 / tf.sqrt(integral(psi*psi, x)))
train = optimizer.minimize(E)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(steps+1):
    sess.run(train)
    sess.run(norm)
    psis[step] = sess.run(psi)
    energys[step] = sess.run(E)
    if step % 50 == 0:
        print(step, energys[step])

x = sess.run(x)

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
axis00.plot(x, np.sqrt(2)*np.sin(pi*x), color='k', linewidth=0.5, label='True wave function')

# 프로그램이 구한 파동함수, 빨간색 점
line0, = axis00.plot(x, psis[step], color='r', lw=0.5, label='The wave function found by the program')

# 이미 알고 있는 에너지, 검은색 선
axis01.plot([0, steps], [pi**2/2, pi**2/2], lw=0.5, color='k', label = 'True energy')
axis01.text(steps, pi**2/2, f'{pi**2/2}', ha='center', va='top')

# 프로그램이 구한 에너지, 빨간색 점
axis01.scatter(range(steps+1), energys, s=1, color='r', label = 'The energy found by the program')
untitled0 = axis01.annotate(f'{energys[-1]}', xytext = (steps, energys[-1]+(energys[0]-energys[-1])/10),
    xy = (steps, energys[-1]), color='r', ha = 'center', arrowprops={'facecolor':'w', 'edgecolor':'r'})

axis00.legend(loc='lower center')   # 첫번째 그림 범례
axis01.legend(loc='upper right')    # 두번째 그림 범례

plt.show()
