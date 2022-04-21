from turtle import color
import numpy as np
import matplotlib.pyplot as plt

def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)

t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)

fig, ax = plt.subplots()

ax.set_xlabel('time')
ax.set_ylabel('value')

ax.plot(t1, f(t1), 'o', color='limegreen')
ax.plot(t2, f(t2), color='blue')

#plt.show()
plt.savefig('sample.eps')
