import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import data
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage.util import crop
import os

os.chdir(r"D:\Old hard\tbia\4th year\2019\2nd term\2019,2nd semester\Computer vision\Tasks\task4\snake")
#img = data.astronaut()
img = scipy.misc.imread("astronout_2.PNG", mode=None)

img = rgb2gray(img)
B = crop(img, ((10, 25), (35, 20)), copy=False)

plt.imshow(B)
print(img.shape, B.shape)

s = np.linspace(0, 2*np.pi, 400)
x = 200 + 100*np.cos(s)
y = 100 + 100*np.sin(s)
init = np.array([x, y]).T


snake = active_contour(gaussian(img, 3),
                       init, alpha=a, beta=b, gamma=g)

fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(img, cmap=plt.cm.gray)
ax.plot(init[:, 0], init[:, 1], '--r', lw=3)
ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3)
ax.set_xticks([]), ax.set_yticks([])
ax.axis([0, img.shape[1], img.shape[0], 0])