
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance

img=Image.open("original_image-1.png")
pixels=img.load()
height,width=img.size
print(height,width)
members = [(0,0)] * 9 
w, h = height, width;
redFeature = [[0 for x in range(h)] for y in range(w)] #2D matrix for feature space
greenFeature = [[0 for x in range(h)] for y in range(w)] #2D matrix for feature space
threshold=100
output_image = Image.new('I',(height,width))

print("calculating feature space...")

        
for i in range (1,height-1):
    for j in range (1,width-1): 
        redFeature[i][j] = pixels[i,j][0]
        greenFeature[i][j] = pixels[i,j][1]
        

redFlattend = np.asarray(redFeature).reshape(-1)
greenFlattend = np.asarray(greenFeature).reshape(-1)
featureSpace=plt.plot(redFlattend, greenFlattend, 'ro')
plt.show()
print(featureSpace)
#for i in range (len(stdFlattend)):
 # dst = distance.euclidean(meanFlattend[i], meanFlattend[i])
  
    
    



#stdSorted = np.sort(stdFlattend)
#meanSorted = np.sort(meanFlattend)
#print(len(meanSorted))
#FeatureSpace = [[0 for x in range(len(stdFlattend))] for y in range(len(stdFlattend))] #2D matrix for feature space
#meanFeatureSpcae = [[0 for x in range(w)] for y in range(h)] #2D matrix for feature space

#print("Finished successfully!!")        
#np.savetxt('stdtext.txt',stdSorted,fmt='%.2f')#print feature space matrix to a text file
#np.savetxt('meantext.txt',meanSorted,fmt='%.2f')#print feature space matrix to a text file