import matplotlib.pyplot as plt
from skimage import data
from skimage import morphology
from skimage import transform
from skimage.color import rgb2hsv
import scipy
import os 
import numpy as np
import scipy
from PIL import Image, ImageDraw
from math import sqrt, pi, cos, sin
from canny import canny_edge_detector
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as patches

os.chdir(r"D:\Old hard\tbia\4th year\2019\2nd term\2019,2nd semester\Computer vision\Tasks\final project") 


rgb_img  = scipy.misc.imread("rbcV2.jpg")

hsv_img   = rgb2hsv(rgb_img)
hue_img   = hsv_img[:, :, 0]
sat_img   = hsv_img[:, :, 1]
value_img = hsv_img[:, :, 2]

plt.subplots(nrows=2, ncols=2, figsize=(8, 8))

plt.subplot(2,2,1)    
plt.imshow(rgb_img)
plt.title("RGB image")   

plt.subplot(2,2,2)    
plt.imshow(hue_img, cmap='hsv')
plt.title("Hue channel")   

plt.subplot(2,2,3)    
plt.imshow(value_img)
plt.title("Value channel")   

plt.subplot(2,2,4)    
plt.imshow(sat_img)
plt.title("saturation channel")   


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(17, 8))
saturaion=hsv_img[...,1].flatten()
plt.subplot(2,2,1)    
points = np.linspace(0.0,.5,100)
range_  = range=(0.0,.5)
plt.hist(saturaion,bins=100,range=(0.0,.5),histtype='stepfilled', color='g', label='Saturation')
plt.title("Histogram")   
plt.xlabel("Value")    
plt.ylabel("Frequency")
plt.legend()
        

plt.subplots(nrows=2, ncols=2, figsize=(8, 8))

sat_img_low_value = sat_img.copy()
for i in np.arange(sat_img.shape[0]):
    for j in np.arange(sat_img.shape[1]):
        if(sat_img_low_value[i,j] > .08 ):
            sat_img_low_value[i,j] = 0
        else:       
            sat_img_low_value[i,j] = 1
#sat_img_low_value = morphology.binary_dilation(sat_img_low_value, selem=np.ones((4, 4)))
            
plt.subplot(2,2,1)    
plt.imshow(sat_img_low_value)
plt.title("Saturation low binary")   
plt.xlabel("low Value")    
plt.ylabel("Frequency")

#High threshold value to prevent WBCS
sat_img_High_value = sat_img.copy()
for i in np.arange(sat_img.shape[0]):
    for j in np.arange(sat_img.shape[1]):
        if(sat_img_High_value[i,j] > .16 ):
            sat_img_High_value[i,j] = 0
        else:       
            sat_img_High_value[i,j] = 1
            
sat_img_High_value = morphology.binary_dilation(sat_img_High_value, selem=np.ones((4, 4)))
plt.subplot(2,2,2)    
plt.imshow(sat_img_High_value)
plt.title("Saturation high binary")   
plt.xlabel("higher Value")    
plt.ylabel("Frequency")
scipy.misc.imsave("WBCS.png",sat_img_High_value)

# In[11]:

sat_img_xor_value = sat_img.copy()
for i in np.arange(sat_img.shape[0]):
    for j in np.arange(sat_img.shape[1]):
        sat_img_xor_value[i,j] = int(sat_img_low_value[i,j]) ^ int(sat_img_High_value[i,j])

plt.subplot(2,2,3)    
plt.imshow(sat_img_xor_value)
plt.title("XOR high binary")   


#smoothed with median filetr after XOR operation
img = sat_img_xor_value.copy()
img = scipy.signal.medfilt(sat_img_xor_value, kernel_size=3)
plt.subplot(2,2,4)    
plt.imshow(sat_img_xor_value)
plt.title("Saturation high binary")   
scipy.misc.imsave("RBCS_XOR_smoothed_img.png",img)

# In[11]:
#Hough transform Circle Red Blood Cells
input_image = Image.open("RBCS_XOR_smoothed_img.png")
print("input_image.size  :",input_image.size)
print("")

rgb_img = Image.open("rbcV2.jpg")

# Output image:
output_image = Image.new("RGB", rgb_img.size )
output_image.paste(rgb_img)
draw_result = ImageDraw.Draw(output_image)

# Find circles
rmin = 55
rmax = 65
steps = 100
threshold = 0.235

points = []
for r in np.arange(rmin, rmax + 1 , 1):
    for t in np.arange(0,steps,1):
        points.append((r, int(r * cos(2 * pi * t / steps)), int(r * sin(2 * pi * t / steps))))

acc = defaultdict(int)
for x, y in canny_edge_detector(input_image):
    for r, dx, dy in points:
        a = x - dx
        b = y - dy
        acc[(a, b, r)] += 1

circles = []
for k, v in sorted(acc.items(), key=lambda i: -i[1]):
    x, y, r = k
    if v / steps >= threshold and all((x - xc) ** 2 + (y - yc) ** 2 > rc ** 2 for xc, yc, rc in circles):
        print(v / steps, x, y, r)
        circles.append((x, y, r))

for x, y, r in circles:
    draw_result.ellipse((x-r, y-r, x+r, y+r), outline=(255,0,0,0))


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 8))
plt.subplot(1,1,1)    

plt.imshow(output_image)
plt.title("Hough Circle RBCS transform")   
# Save output image
output_image.save("result_high_low_dilated.png")


# In[11]:

input_image = Image.open("bloodcells.jpg")
input_image = np.invert(input_image)
input_image = scipy.signal.medfilt(input_image, kernel_size=51)
input_image = morphology.binary_dilation(input_image, selem=np.ones((40, 40)))
#input_image = morphology.binary_erosion(input_image, selem=np.ones((4, 4)))

scipy.misc.imsave("WBCs.jpg",input_image)
input_image = Image.open("WBCs.jpg")


# Output image:
output_image = Image.new("RGB", input_image.size )
output_image.paste(input_image)
draw_result = ImageDraw.Draw(output_image)

# Find circles
rmin = 50
rmax = 100
steps = 100
threshold = 0.235

points = []
for r in np.arange(rmin, rmax + 1 , 1):
    for t in np.arange(0,steps,1):
        points.append((r, int(r * cos(2 * pi * t / steps)), int(r * sin(2 * pi * t / steps))))

acc = defaultdict(int)
for x, y in canny_edge_detector(input_image):
    for r, dx, dy in points:
        a = x - dx
        b = y - dy
        acc[(a, b, r)] += 1

circles = []
for k, v in sorted(acc.items(), key=lambda i: -i[1]):
    x, y, r = k
    if v / steps >= threshold and all((x - xc) ** 2 + (y - yc) ** 2 > rc ** 2 for xc, yc, rc in circles):
        print(v / steps, x, y, r)
        circles.append((x, y, r))

for x, y, r in circles:
    draw_result.ellipse((x-r, y-r, x+r, y+r), outline=(0,255,0,0))


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
plt.subplot(1,1,1)    
plt.imshow(output_image)
plt.title("Hough Circle WBCS transform")   
# Save output image
output_image.save("Hough Circle WBCS transform.png")

loaded_image = scipy.misc.imread("WBCs.png", flatten=True, mode=None)
draw_result = ImageDraw.Draw(loaded_image)

im = np.array(Image.open('WBCs.png'), dtype=np.uint8)
fig,ax = plt.subplots(1)
ax.imshow(im)
rect = patches.Rectangle((50,100),x,y,linewidth=1,edgecolor='r',facecolor='none')
ax.add_patch(rect)

plt.show()

#draw_result.Draw.rectangle(x,y, fill=None, outline=None)



# In[11]:


