
import numpy as np
from PIL import Image 
import scipy.misc
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def rgb2gray(rgb):
   return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

kernel=[[1/9,1/9,1/9],
        [1/9,1/9,1/9],
        [1/9,1/9,1/9]]

img = Image.open("w1024.jpg")
pix=img.load()
imge= mpimg.imread("w1024.jpg")
loaded_image2 = scipy.misc.imread("w1024.jpg")#, flatten=True)

plt.imshow(loaded_image2)
imggrey=rgb2gray(loaded_image2)
plt.imshow(imggrey)
print(5//2)



#for i in range (0,img.height):
 #   for j in range (0,img.width):
        
  #    loaded_image2.itemset((i,j), Grey[j])


    
plt.show()
