import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

im = plt.imshow(np.random.randn(10,10))

def update(i):
    A = np.random.randn(10,10)
    im.set_array(A)
    return im

ani = FuncAnimation(plt.gcf(), update, frames=range(100), interval=5, blit=False)

plt.show()