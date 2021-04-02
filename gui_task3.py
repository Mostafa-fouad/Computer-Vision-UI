import sys
from PyQt5.uic import loadUi
from PyQt5 import QtCore , QtGui ,QtWidgets
from PIL import Image,ImageDraw,ImageOps
from math import sqrt
from PyQt5.QtWidgets import QDialog, QMainWindow
from PIL.ImageQt import ImageQt
from PyQt5.QtWidgets import QMessageBox, QFileDialog, QApplication, QLCDNumber, QTextEdit, QLabel, QProgressBar
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import scipy.ndimage.filters
from PIL import Image , ImageOps,ImageDraw
from scipy.ndimage import gaussian_filter
import matplotlib.image as mpimg
from skimage import data, feature, color, filters
from skimage.io import imread
from skimage.color import rgb2gray
from scipy import signal as sig
import math
from skimage.transform import (hough_line, hough_line_peaks,probabilistic_hough_line)
from skimage.feature import canny
from matplotlib import cm
from canny_task3 import canny_edge_detector
from collections import defaultdict
from math import  pi, cos, sin,tanh
import random


global seed , opencvImagePix , segmentation_value
Window = True
event = True
global condition
condition = True
x=0
y=0
textboxValue=''
selectedKernel=""
index=0
selectedFilter=""
limg_back = None
himg_back = None
fshift = None
fshift_copy = None
magnitude_spectrum = None




class Main(QMainWindow):
   
          def __init__(self):
                  list1=[ "   ",
                          "Gaussian filter",
                          "Box filter",
                          "Median filter",
                          "Sharpening filter",
                          "Prewitt",
                          "Sobel",
                          "Laplacian",
                          "Laplacian of Gaussian (LoG)",
                          "Difference of Gaussian (DoG)"
                          ]
                  segList=[ "   ",
                           "Region Growing",
                           "Mean Shift",
                           
                          ]
          
                  
                  super(Main, self).__init__()
                  loadUi('mainwindow.ui', self)   
                  self.comboBox.addItems(list1)
                  self.Segmentation_comboBox.addItems(segList)
                  self.progressBar.setValue(0)
                  self.pushButton_filters_load.clicked.connect(self.openImage)#browse to image for spatial filters
                  self.pushButton_histograms_load.clicked.connect(self.histImage)#browse to image for histogram_equalization
                  self.pushButton_histograms_load_target.clicked.connect(self.histImageMatch)#browse to image for matching
                  self.pushButton_FourierTransform.clicked.connect(self.fourierTransform)#browse to image for fourier filters
                  self.pushButton_corners_load.clicked.connect(self.cornerImage)#browse to image for conrer detection image
                  self.Segmentation_pushButton.clicked.connect(self.segmentationImage)#browse to image for segmentation
                  self.pushButton_corners_load_2.clicked.connect(self.findCorners)#get corners of image
                  self.pushButton_lines_load.clicked.connect(self.openImage_Hough_Lines)
                  self.pushButton_circles_load.clicked.connect(self.openImage_Hough_Circles)
                  self.comboBox.activated.connect(self.handleActivated)
                  self.Segmentation_comboBox.activated.connect(self.segActivated)
                  self.comboBox_2.activated.connect(self.ffthandleActivated)
                  self.radioButton.clicked.connect(self.histogramEqualization)
                  self.radioButton_2.clicked.connect(self.histogramMatching)
                  self.pushButton_histograms_load.setCheckable(True)  
                  self.pushButton_histograms_load_target.setCheckable(True)  
                  self.Segmentation_input_image.mousePressEvent = self.getPixel
                  #self.Stop.clicked.connect(self.stop)
                  
                  
#_______________________________________________combox_1 filters(spatial domain)______________________________________________                   
          def handleActivated(self,text):
              global selectedKernel 
              
              selectedFilter=self.comboBox.itemText(text)
              
              selectedKernel=selectedFilter
              if selectedKernel in ["Gaussian filter","Box filter","Sharpening filter"]:
                  self.Filtering()
              elif selectedKernel in ["Sobel","Prewitt","Laplacian","Laplacian of Gaussian (LoG)","Difference of Gaussian (DoG)"]:
                  self.Edge_detection()
              elif selectedKernel=="Median filter":
                  self.Median_filter()
              
                 
          
              
          def segActivated(self,text):       
                  global selectedSegmentation
                  selectedSegmentation=self.Segmentation_comboBox.itemText(text)
                  self.segment(selectedSegmentation)
#_______________________________________________Segmentation Part______________________________________________                   
          def segmentationImage(self):
                    global segmap ,segmapCopy ,imageName ,loadedImage,loadedPixels,loaded_image,rows,cols
                    imageName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image", "", "Image Files (*.png *.jpg *jpeg *.bmp)") # Ask for file
                    loaded_image = scipy.misc.imread(imageName, flatten=True, mode=None)
                    rows, cols = loaded_image.shape
                    loadedImage = Image.open(imageName)
                    loadedPixels = loadedImage.load()
                    segmap = QtGui.QPixmap(imageName) # Setup pixmap with the provided image
                    segmap = segmap.scaled(self.Segmentation_input_image.width(), self.Segmentation_input_image.height(), QtCore.Qt.KeepAspectRatio) # Scale pixmap
                    segmapCopy=segmap.copy();
                    self.Segmentation_input_image.setPixmap(segmap) # Set the pixmap onto the label  
                    self.Segmentation_input_image.setAlignment(QtCore.Qt.AlignCenter) # Align the label to center""          
                
            
          def segment(self,selectedSegmentation):
#_______________________________________________ 1st_ Mean shift segmentation ______________________________________________                   
              
              if selectedSegmentation=="Mean Shift":
                    
                    K=loadedImage.load() #load image as a pixel map
                    height,width=loadedImage.size
                    row=width
                    col=height
                   
                    J= row * col
                    #print(J)
                    # preparation for mean-shift
                    
                    Size = row,col,3
                    outputImage = np.zeros(Size, dtype= np.uint8) 
                    D=np.zeros((J,5))
                    arr=np.array((1,3)) #array to save the RGB of each pixel and pass it to the feature space image
                    
                    
                    
                    counter=0  
                    iter=int(self.iter_value.text())   #get the ecludian distance from the text box    
                    
                    
                    threshold=int(self.Threshold_value.text()) #get the threshold from the text box  
                    current_mean_random = True
                    current_mean_arr = np.zeros((1,5))
                    below_threshold_arr=[]
                    
                    # convert the image K[rows][col] into a feature space D. The dimensions of D are [rows*col][5]
                    for i in range(0,row):
                        for j in range(0,col):      
                            
                            arr= K[j,i]
                            #print (K[j,i])
                            
                            for n in range(0,5): 
                               
                                if(n>=0) & (n <=2):
                                   
                                    D[counter][n]=arr[n] # assign RGB values as features of the image
                                else:
                                    if(n==3):
                                        D[counter][n]=i # x coordinates
                                    else:
                                        D[counter][n]=j # y coordinates
                                        
                            counter+=1
                            
                            
                    while(len(D)>0):
                        val=int(((J-len(D))/J)*100)
                        self.progressBar.setValue(val+1)
                        
                    #selecting a random row from the feature space and assigning it as the current mean    
                    
                        if(current_mean_random):
                            current_mean= random.randint(0,len(D)-1) # pick any random point as an initial mean
                            for i in range(0,5):
                                current_mean_arr[0][i] = D[current_mean][i] # assign this random feature value to this variable
                        below_threshold_arr=[]
                        for i in range(0,len(D)):
                            
                            ecl_dist = 0
                            
                    #Finding the eucledian distance of the randomly selected row i.e. current mean with all the other rows
                            for j in range(0,5):
                                ecl_dist += ((current_mean_arr[0][j] - D[i][j])**2)
                                    
                            ecl_dist = ecl_dist**0.5
                    
                    #Checking if the distance calculated is within the threshold. If yes taking those rows and adding 
                    #them to a list below_threshold_arr
                          
                            if(ecl_dist < threshold):
                                below_threshold_arr.append(i) # save the detected feature row that is less 
                                                              # than the ecludian distance threshold
                                
                                
                                
                        Rvalue = 0
                        Gvalue = 0
                        Bvalue = 0
                        x_index = 0
                        y_index = 0
                        Rmean = 0
                        Gmean = 0
                        Bmean = 0
                        xmean = 0
                        ymean = 0
                        current_mean = 0
                        
                        
                    #For all the rows found and placed in below_threshold_arr list, calculating the average of 
                    #Red, Green, Blue and index positions.
                        
                        for i in range(0, len(below_threshold_arr)):
                            Rvalue += D[below_threshold_arr[i]][0]
                            Gvalue += D[below_threshold_arr[i]][1]
                            Bvalue += D[below_threshold_arr[i]][2]
                            x_index += D[below_threshold_arr[i]][3]
                            y_index += D[below_threshold_arr[i]][4]   
                        
                        Rmean = Rvalue / len(below_threshold_arr)
                        Gmean = Gvalue / len(below_threshold_arr)
                        Bmean = Bvalue / len(below_threshold_arr)
                        xmean = x_index / len(below_threshold_arr)
                        ymean = y_index / len(below_threshold_arr)
                        
                    #Finding the distance of these average values with the current mean and comparing it with iter
                    
                        mean_ec_distance = ((Rmean - current_mean_arr[0][0])**2 + (Gmean - current_mean_arr[0][1])**2 + (Bmean - current_mean_arr[0][2])**2 + (xmean - current_mean_arr[0][3])**2 + (ymean - current_mean_arr[0][4])**2)
                        
                        mean_ec_distance = mean_ec_distance**0.5
                        
                        
                    
                        
                    # If less than iter, find the row in below_threshold_arr that has i,j nearest to mean_i and mean_j
                    #This is because mean_i and mean_j could be decimal values which do not correspond
                    #to actual pixel in the Image array.
                    
                        if(mean_ec_distance < iter):
                         
                            new_arr = np.zeros((1,3))
                            new_arr[0][0] = Rmean
                            new_arr[0][1] = Gmean
                            new_arr[0][2] = Bmean
                            
                    # When found, color all the rows in below_threshold_arr with 
                    #the color of the row in below_threshold_arr that has i,j nearest to mean_i and mean_j
                            for i in range(0, len(below_threshold_arr)):    
                                outputImage[int(D[below_threshold_arr[i]][3])][int(D[below_threshold_arr[i]][4])] = new_arr #draw the RGB of the features in the new image
                                
                    # Also now don't use those rows that have been colored once.
                                
                                D[below_threshold_arr[i]][0] = -1 # just a flag to exclude the used row
                                 
                            current_mean_random = True
                            new_D=np.zeros((len(D),5))
                            counter_i = 0
                            
                            for i in range(0, len(D)):
                                if(D[i][0] != -1):
                                    new_D[counter_i][0] = D[i][0]
                                    new_D[counter_i][1] = D[i][1]
                                    new_D[counter_i][2] = D[i][2]
                                    new_D[counter_i][3] = D[i][3]
                                    new_D[counter_i][4] = D[i][4]
                                    counter_i += 1
                                
                            
                            D=np.zeros((counter_i,5))
                            
                            counter_i -= 1
                            for i in range(0, counter_i):
                                D[i][0] = new_D[i][0]
                                D[i][1] = new_D[i][1]
                                D[i][2] = new_D[i][2]
                                D[i][3] = new_D[i][3]
                                D[i][4] = new_D[i][4]
                            
                        else:
                            current_mean_random = False
                             
                            current_mean_arr[0][0] = Rmean
                            current_mean_arr[0][1] = Gmean
                            current_mean_arr[0][2] = Bmean
                            current_mean_arr[0][3] = xmean
                            current_mean_arr[0][4] = ymean
                                
                        
                          
                    scipy.misc.imsave("Mean_shift_output.png",outputImage)
                    segmap2 = QtGui.QPixmap("Mean_shift_output.png") # Setup pixmap with the provided image
                    segmap2 = segmap2.scaled(self.Segmentation_output_image.width(), self.Segmentation_output_image.height(), QtCore.Qt.KeepAspectRatio) # Scale pixmap
                    self.Segmentation_output_image.setPixmap(segmap2) # Set the pixmap onto the label
                    self.Segmentation_output_image.setAlignment(QtCore.Qt.AlignCenter) # Align the label to center""    
                    
#_______________________________________________ 2nd_ Region Growing segmentation ______________________________________________                   
    
              elif   selectedSegmentation=="Region Growing":
    
                  savedImage = loadedImage.resize((301,221),Image.ANTIALIAS)
                  scipy.misc.imsave("ready.png",savedImage)
                  segmap = QtGui.QPixmap("ready.png")
                  segmap = segmap.scaled(self.Segmentation_input_image.width(), self.Segmentation_input_image.height(), QtCore.Qt.KeepAspectRatio) # Scale pixmap
                  self.Segmentation_input_image.setPixmap(segmap) # Set the pixmap onto the label  
                  self.Segmentation_input_image.setAlignment(QtCore.Qt.AlignCenter) # Align the label to center""          
                  
          def getPixel(self, event):
              
            self.img = QtGui.QImage('ready.png')
            x = event.pos().x()
            y = event.pos().y() 
            print ("x=",x)
            print ("y=",y)
            c = self.img.pixel(x,y)  # color code (integer): 3235912
            c_rgb = QtGui.QColor(c).getRgb()  # 8bit [Red,Green,Blue]
            seed=x,y,c_rgb[0],c_rgb[1],c_rgb[2]
            
            opencvImage=Image.open("ready.png")
            opencvImage=opencvImage.resize((301,221),Image.ANTIALIAS)
            scipy.misc.imsave("ready1.png",opencvImage)
            opencvImage=Image.open("ready1.png")
            
            out = self.region_growing(opencvImage, seed)
            
            scipy.misc.imsave("Region_Growing_output.png",out)
            out=Image.open("Region_Growing_output.png")
            outSaved = out.resize((301,221),Image.ANTIALIAS)
            scipy.misc.imsave("Region_Growing_output21.png",outSaved)
            segmap2 = QtGui.QPixmap("Region_Growing_output21.png") # Setup pixmap with the provided image
            segmap2 = segmap2.scaled(self.Segmentation_output_image.width(), self.Segmentation_output_image.height(), QtCore.Qt.KeepAspectRatio) # Scale pixmap
            self.Segmentation_output_image.setPixmap(segmap2) # Set the pixmap onto the label
            self.Segmentation_output_image.setAlignment(QtCore.Qt.AlignCenter) # Align the label to center""    

         
                    
                    
          def region_growing(self,img, seed):
            list = []
           
            height,width=img.size
            output_image = Image.new('I',(height,width))
            imgPix=img.load()
            h,w=img.size
            segmentation_value = int(self.Threshold_value_2.text())
            
            
            list.append((seed[0], seed[1]))#if the current pixel has the same feature add it to the region
            processed = []
            
            while(len(list) > 0 ):
                        pix = list[0]
                        for coord in self.getAdjacencies(pix[0], pix[1]):
                          print(imgPix[coord[0], coord[1]],len(list))
                          if (imgPix[coord[0], coord[1]]) <=((seed[2],seed[3],seed[4])) and (imgPix[coord[0], coord[1]]) >=((seed[2]-segmentation_value,seed[3]-segmentation_value,seed[4]-segmentation_value)) :
                            output_image.putpixel((coord[0], coord[1]),255)
                            
                            if not coord in processed:
                                list.append(coord)
                            processed.append(coord)
                            
                        list.pop(0)
                       
                        
                  
                
            return output_image       
      
    
    
          def getAdjacencies(self,x, y):
                            out = []
                            opencvImage= Image.open("ready.png")
                            h,w=opencvImage.size
                            maxx = h
                            maxy = w
                        #if min(max(y-1,0),maxy) != 0 or min(max(x-1,0),maxx) != 0 or min(max(x+1,0),maxx) != w or min(max(y+1,0),maxy) != h:
                        #top left
                            outx = min(max(x-1,0),maxx)
                            outy = min(max(y-1,0),maxy)
                            out.append((outx,outy))
                        
                            #top center
                            outx = x
                            outy = min(max(y-1,0),maxy)
                            out.append((outx,outy))
                           
                        
                            #top right
                            outx = min(max(x+1,0),maxx)
                            outy = min(max(y-1,0),maxy)
                            out.append((outx,outy))
                        
                            #left
                            outx = min(max(x-1,0),maxx)
                            outy = y
                            out.append((outx,outy))
                        
                            #right
                            outx = min(max(x+1,0),maxx)
                            outy = y
                            out.append((outx,outy))
                        
                            #bottom left
                            outx = min(max(x-1,0),maxx)
                            outy = min(max(y+1,0),maxy)
                            out.append((outx,outy))
                        
                            #bottom center
                            outx = x
                            outy = min(max(y+1,0),maxy)
                            out.append((outx,outy))
                        
                            #bottom right
                            outx = min(max(x+1,0),maxx)
                            outy = min(max(y+1,0),maxy)
                            out.append((outx,outy))
                        
                       
                            return out 
#_______________________________________________combox_2 filters(fourier domain)______________________________________________                                 
        
          def ffthandleActivated(self,text):
             global selectedfftFilter
             selectedfftFilter=self.comboBox_2.itemText(text)
             if selectedfftFilter == "Low Pass Filter":
                 self.fftlowpass()
             elif selectedfftFilter=="High Pass Filter":
                 self.ffthighpass()
                 
          

#_______________________________________________Fourier transform_________________________________________________________________                         
            
            
          def fourierTransform(self):
                    global LPKernel,HPKernel
                    f = np.fft.fft2(loaded_image)
                    self.fshift = np.fft.fftshift(f)
                    self.fshift_copy = self.fshift.copy()
                    self.magnitude_spectrum = 20*np.log(np.abs(self.fshift))
                    plt.imsave('fft.png', self.magnitude_spectrum , cmap='gray', format='png')
                    pixmap1 = QtGui.QPixmap("fft.png") # Setup pixmap with the provided image
                    pixmap1 = pixmap1.scaled(self.label_filters_output.width(), self.label_filters_output.height(), QtCore.Qt.KeepAspectRatio) # Scale pixmap
                    self.label_filters_output.setPixmap(pixmap1) # Set the pixmap onto the label  
                    self.label_filters_output.setAlignment(QtCore.Qt.AlignCenter) # Align the label to center""
                   
                    

#_______________________________________________low pass filter in Fourier transform_________________________________________________________________                         
          
          def fftlowpass(self):
              
                    LPKernel = int(self.lineEdit_4.text())
                    crow,ccol = rows/2 , cols/2
                    # create a mask first, center square is 1, remaining all zeros
                    mask = np.zeros((rows,cols),np.uint8)
                    mask[int(crow-LPKernel):int(crow+LPKernel), int(ccol-LPKernel):int(ccol+LPKernel)] = 1
                    np.savetxt('text.txt',mask,fmt='%d')
                    lpf = mask*self.fshift_copy
                    lpf = np.fft.ifftshift(lpf)
                    limg_back = np.fft.ifft2(lpf)
                    limg_back = np.abs(limg_back)
                    plt.imsave('lowfftt.png', limg_back , cmap='gray', format='png')
                    pixmap2 = QtGui.QPixmap("lowfftt.png") # Setup pixmap with the provided image
                    pixmap2 = pixmap2.scaled(self.label_filters_output.width(), self.label_filters_output.height(), QtCore.Qt.KeepAspectRatio) # Scale pixmap
                    self.label_filters_output.setPixmap(pixmap2) # Set the pixmap onto the label  
                    self.label_filters_output.setAlignment(QtCore.Qt.AlignCenter) # Align the label to center""

#_______________________________________________high pass filter in Fourier transform_________________________________________________________________                         
            
          def ffthighpass(self):
              
                        HPKernel = int(self.lineEdit_5.text())
                        crow,ccol = rows/2 , cols/2
                        self.fshift[int(crow-HPKernel):int(crow+HPKernel), int(ccol-HPKernel):int(ccol+HPKernel)] = 0
                        hpf = np.fft.ifftshift(self.fshift)
                        himg_back = np.fft.ifft2(hpf)
                        himg_back = np.abs(himg_back)  
                        plt.imsave('highfft.png', himg_back, cmap='gray', format='png')
                        pixmap3 = QtGui.QPixmap("highfft.png") # Setup pixmap with the provided image
                        pixmap3 = pixmap3.scaled(self.label_filters_output.width(), self.label_filters_output.height(), QtCore.Qt.KeepAspectRatio) # Scale pixmap
                        self.label_filters_output.setPixmap(pixmap3) # Set the pixmap onto the label  
                        self.label_filters_output.setAlignment(QtCore.Qt.AlignCenter) # Align the label to center""

#_______________________________________________Corner detection_________________________________________________________________                         
           
          def cornerImage(self):
                   
                    global pixmap ,pixmapCopy ,fileName1 ,input_image,input_pixels
                    fileName1, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image", "", "Image Files (*.png *.jpg *jpeg *.bmp)") # Ask for file
                    input_image = Image.open(fileName1)
                    input_pixels = input_image.load()
                    pixmap = QtGui.QPixmap(fileName1) # Setup pixmap with the provided image
                    pixmap = pixmap.scaled(self.label_corners_corners_output.width(), self.label_corners_corners_output.height(), QtCore.Qt.KeepAspectRatio) # Scale pixmap
                    pixmapCopy=pixmap.copy();
                    self.label_corners_corners_output.setPixmap(pixmap) # Set the pixmap onto the label  
                    self.label_corners_corners_output.setAlignment(QtCore.Qt.AlignCenter) # Align the label to center""
                    
                    
          def findCorners(self):     
              img = imread(fileName1)
              imggray = rgb2gray(img)
              
              def gradient_x(imggray):
                ##Sobel operator kernels.
                kernel_x = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
                return sig.convolve2d(imggray, kernel_x, mode='same')
              def gradient_y(imggray):
                kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
                return sig.convolve2d(imggray, kernel_y, mode='same')
              Ix = gradient_x(imggray)
              Iy = gradient_y(imggray)
              Ixx = Ix**2
              Ixy = Iy*Ix
              Iyy = Iy**2
              k = float(self.corner_threshold.text())
              
              height, width = imggray.shape
              print(height,width)
              harris_response = []
             
              offset = 1
              for y in range(offset, height-offset):
                  for x in range(offset, width-offset):
                    Sxx = np.sum(Ixx[y-offset:y+1+offset, x-offset:x+1+offset])
                    Syy = np.sum(Iyy[y-offset:y+1+offset, x-offset:x+1+offset])
                    Sxy = np.sum(Ixy[y-offset:y+1+offset, x-offset:x+1+offset])
                    
                    #Find determinant and trace, use to get corner response
                    
                    det = (Sxx * Syy) - (Sxy**2)
                    trace = Sxx + Syy
                    r = det - k*(trace**2)
                    
                    harris_response.append([x,y,r])
                    img_copy = np.copy(img)
            
              for response in harris_response:
                x, y, r = response
                if r > 0:
                    img_copy[y,x] = [255,0,0]
              
              
              scipy.misc.imsave("corner_detection_output.png",img_copy)
              
              original_image = Image.open("corner_detection_output.png")
              
              size = (700, 700)
              fit_and_resized_image = ImageOps.fit(original_image, size, Image.ANTIALIAS)
              fit_and_resized_image.save("corner_detection_last_output.jpg")
              pixmap2 = QtGui.QPixmap("corner_detection_last_output.jpg") # Setup pixmap with the provided image
              pixmap2 = pixmap2.scaled(self.label_corners_corners.width(), self.label_corners_corners.height(), QtCore.Qt.KeepAspectRatio) # Scale pixmap
              self.label_corners_corners.setPixmap(pixmap2) # Set the pixmap onto the label
              self.label_corners_corners.setAlignment(QtCore.Qt.AlignCenter) # Align the label to center""
              print("Corners have been detected!!")
#_______________________________________________browse to an image on computer_________________________________________________________________                         
                     
          def openImage(self):
                    global pixmap ,pixmapCopy ,fileName ,input_image,input_pixels,loaded_image,rows,cols
                    fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image", "", "Image Files (*.png *.jpg *jpeg *.bmp)") # Ask for file
                    loaded_image = scipy.misc.imread(fileName, flatten=True, mode=None)
                    rows, cols = loaded_image.shape
                    input_image = Image.open(fileName)
                    input_pixels = input_image.load()
                    pixmap = QtGui.QPixmap(fileName) # Setup pixmap with the provided image
                    pixmap = pixmap.scaled(self.label_filters_input.width(), self.label_filters_input.height(), QtCore.Qt.KeepAspectRatio) # Scale pixmap
                    pixmapCopy=pixmap.copy();
                    self.label_filters_input.setPixmap(pixmap) # Set the pixmap onto the label  
                    self.label_filters_input.setAlignment(QtCore.Qt.AlignCenter) # Align the label to center""
         
          
                    
          def rgb2gray(rgb):
                      return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
                  
          def convolution (self,image,kernel):
             
                    offset = len(kernel) // 2
                
                    # Create output image
                    output = Image.new("RGB", image.size)
                    draw = ImageDraw.Draw(output)
                    
                    # Compute convolution between intensity and kernels
                    for x in range(offset, image.width - offset):#for image indices
                        for y in range(offset, image.height - offset):
                            acc = [0, 0, 0]
                           
                            for a in range(len(kernel)):
                                for b in range(len(kernel)):
                                    xn = x + a - offset
                                    yn = y + b - offset
                                    pixel = input_pixels[xn, yn]
                                    acc[0] += pixel[0] * kernel[a][b]
                                    acc[1] += pixel[1] * kernel[a][b]
                                    acc[2] += pixel[2] * kernel[a][b]
                    
                            draw.point((x, y), (int(acc[0]), int(acc[1]), int(acc[2])))
                    return output
#===========================================(smoothing and sharpening filters)====================================================              
         
          def Filtering(self):
             #============================================(Box filter)============================================================       
                 
                box_kernel = [[1 / 25, 1 / 25, 1 / 25, 1/25, 1/ 25],
                              [1 / 25, 1 / 25, 1 / 25, 1/25, 1/ 25],
                              [1 / 25, 1 / 25, 1 / 25, 1/25, 1/ 25],
                              [1 / 25, 1 / 25, 1 / 25, 1/25, 1/ 25],
                              [1 / 25, 1 / 25, 1 / 25, 1/25, 1/ 25]]
                
             #============================================(Gaussian filter)=======================================================
               
                gaussian_kernel =  [[6  / 600, 12/ 600, 24 / 600,  12 / 600, 6  / 600],
                                    [12 / 600, 24/ 600, 48 / 600,  24 / 600, 12 / 600],
                                    [24 / 600, 48/ 600, 96 / 600,  48 / 600, 24 / 600],
                                    [12 / 600, 24/ 600, 48 / 600,  24 / 600, 12 / 600],
                                    [6  / 600, 12/ 600, 24 / 600,  12 / 600, 6  / 600]]
                                 
            #============================================(High-pass kernel)=======================================================
                
                high_pass_kernel = [[ 0 , -1 ,  0 ],
                                    [-1 ,  5 , -1 ],
                                    [ 0 , -1 ,  0 ]]
                
            #___________________________________________________________________________________________________________
            
                if selectedKernel== "Gaussian filter":
                     kernel = gaussian_kernel
                elif selectedKernel== "Box filter":
                     kernel = box_kernel
                elif selectedKernel== "Sharpening filter":
                     kernel = high_pass_kernel
                     
                
                
                
                if selectedKernel in ["Box filter","Sharpening filter"]:
                    
                    output_image=self.convolution (input_image,kernel)#convolution function
                    output_image.save("smoothing_Sharpening_output.png")
                    pixmap2 = QtGui.QPixmap("smoothing_Sharpening_output.png") # Setup pixmap with the provided image
                    pixmap2 = pixmap2.scaled(self.label_filters_output.width(), self.label_filters_output.height(), QtCore.Qt.KeepAspectRatio) # Scale pixmap
                    self.label_filters_output.setPixmap(pixmap2) # Set the pixmap onto the label
                    self.label_filters_output.setAlignment(QtCore.Qt.AlignCenter) # Align the label to center""
                
                elif selectedKernel == "Gaussian filter":
                    
                    img = mpimg.imread(fileName)
                    img_grey= rgb2gray(img)
                    img_grey=filters.gaussian(img_grey, sigma=int(self.lineEdit.text()))
                    scipy.misc.imsave("smoothing_Sharpening_output.png",img_grey)
                    pixmap2 = QtGui.QPixmap("smoothing_Sharpening_output.png") # Setup pixmap with the provided image
                    pixmap2 = pixmap2.scaled(self.label_filters_output.width(), self.label_filters_output.height(), QtCore.Qt.KeepAspectRatio) # Scale pixmap
                    self.label_filters_output.setPixmap(pixmap2) # Set the pixmap onto the label
                    self.label_filters_output.setAlignment(QtCore.Qt.AlignCenter) # Align the label to center""
                 
#_______________________________________________Edge Detection Part_________________________________________________________________                         
                
          def Edge_detection(self):
              
              #===========================================(Sobel operator)=======================================
                if selectedKernel== "Sobel":
              
                    intensity = [[sum(input_pixels[x, y]) / 3 
                                  for y in range(input_image.height)] 
                                  for x in range(input_image.width)]#grey
                   

                    # Sobel kernels
                    kernelx = [[1, 0, -1],
                               [2, 0, -2],
                               [1, 0, -1]]
                    
                    kernely = [[1 ,  2, 1 ],
                               [0 ,  0, 0 ],
                               [-1, -2, -1]]
                    
                    # Create output image
                    output_image = Image.new("RGB", input_image.size)
                    draw = ImageDraw.Draw(output_image)
                    
                    # Compute convolution between intensity and kernels
                    for x in range(1, input_image.width - 1):#for image indices
                        for y in range(1, input_image.height - 1):
                            Sx, Sy = 0, 0
                            for a in range(3):#for kernel indices
                                for b in range(3):
                                    xn = x + a - 1
                                    yn = y + b - 1
                                    Sx += intensity[xn][yn] * kernelx[a][b]
                                    Sy += intensity[xn][yn] * kernely[a][b]
                    
                            # Draw in black and white the magnitude
                            Sobel = int(sqrt(Sx**2 + Sy**2))
                            draw.point((x, y), (Sobel, Sobel, Sobel))
                    output_image.save("Sobel_outputt.png")   
                    pixmap2 = QtGui.QPixmap("Sobel_outputt.png")
                    
                    
         #==========================================(Prewitt operator)==================================================================
                elif  selectedKernel== "Prewitt":                                 
                  
                    img = np.array(Image.open(fileName)).astype(np.uint8)
                    gray_img = np.round(0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]).astype(np.uint8)          
                    h, w = gray_img.shape
                # define filters
                    horizontal = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])  # s2
                    vertical   = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])  # s1
         
               # define images with 0s
                    newgradientImage  = np.zeros((h, w))
                    # offset by 1
                    for i in range(1, h - 1):
                        for j in range(1, w - 1):
                            horizontalGrad = (horizontal[0, 0] * gray_img[i - 1, j - 1]) + \
                                             (horizontal[0, 1] * gray_img[i - 1, j]) + \
                                             (horizontal[0, 2] * gray_img[i - 1, j + 1]) + \
                                             (horizontal[1, 0] * gray_img[i, j - 1]) + \
                                             (horizontal[1, 1] * gray_img[i, j]) + \
                                             (horizontal[1, 2] * gray_img[i, j + 1]) + \
                                             (horizontal[2, 0] * gray_img[i + 1, j - 1]) + \
                                             (horizontal[2, 1] * gray_img[i + 1, j]) + \
                                             (horizontal[2, 2] * gray_img[i + 1, j + 1])
                    
                            verticalGrad = (vertical[0, 0] * gray_img[i - 1, j - 1]) + \
                                           (vertical[0, 1] * gray_img[i - 1, j]) + \
                                           (vertical[0, 2] * gray_img[i - 1, j + 1]) + \
                                           (vertical[1, 0] * gray_img[i, j - 1]) + \
                                           (vertical[1, 1] * gray_img[i, j]) + \
                                           (vertical[1, 2] * gray_img[i, j + 1]) + \
                                           (vertical[2, 0] * gray_img[i + 1, j - 1]) + \
                                           (vertical[2, 1] * gray_img[i + 1, j]) + \
                                           (vertical[2, 2] * gray_img[i + 1, j + 1])


                      # Edge Magnitude
                            mag = np.sqrt(pow(horizontalGrad, 2.0) + pow(verticalGrad, 2.0))
                            newgradientImage [i - 1, j - 1] = mag
                    plt.imsave('Prewitt_output.png', newgradientImage , cmap='gray', format='png')
                    pixmap2 = QtGui.QPixmap("Prewitt_output.png") # Setup pixmap with the provided image
         
#==========================================================(Laplcian operator)==========================================================                
                elif   selectedKernel== "Laplacian":                  
                    
                            laplcian_kernel= [[-1, -1, -1],
                                              [-1,  8, -1],
                                              [-1, -1, -1]]
                            kernel = laplcian_kernel
                            laplacianImage=self.convolution(input_image,kernel)
                            laplacianImage.save("Laplacian_output.png")
                            pixmap2 = QtGui.QPixmap("Laplacian_output.png")
                
#============================================(Laplacian of Gaussian (LoG) operator)===========================================                
                elif   selectedKernel== "Laplacian of Gaussian (LoG)":
                
                    img = mpimg.imread(fileName)
                    img_grey= rgb2gray(img)
                    scipy.misc.imsave("Laplacian_of_Gaussian_output1.png",img_grey)
                    
                   
                    input_image1=Image.open("Laplacian_of_Gaussian_output1.png")
                    
                    
                    LoG_kernel=  [[0,0,3,2,2,2,3,0,0],
                                  [0,2,3,5,5,5,3,2,0],
                                  [3,3,5,3,0,3,5,3,3],
                                  [2,5,3,-12,-23,-12,3,5,2],
                                  [2,5,0,-23,-40,-23,0,5,2],
                                  [2,5,3,-12,-23,-12,3,5,2],
                                  [3,3,5,3,0,3,5,3,3],
                                  [0,2,3,5,5,5,3,2,0],
                                  [0,0,3,2,2,2,3,0,0]]#LoG kernerl with sigma = 1.6
                             
                    kernel = LoG_kernel        
                    LoGImage=self.convolution(input_image1,kernel)           
                    LoGImage.save("Laplacian_of_Gaussian_outputtt.png")
                    pixmap2 = QtGui.QPixmap("Laplacian_of_Gaussian_outputtt.png")
#====================================================(Difference of Gaussian (DoG) operator)==================================                   
                elif selectedKernel=="Difference of Gaussian (DoG)":
                   
                   
                 img = mpimg.imread(fileName)
                 img_grey= rgb2gray(img)
                 a=1.2
                 threshold=0.3
                 phi=0.6
                 sigma1 = int(self.lineEdit_2.text())
                 sigma2 = int(self.lineEdit_2.text())
                 #==================================== Segmentation by Winnemuller ==================================
                 t=0.6
                 s1 = filters.gaussian(img_grey,a*sigma1)
                 s2 = filters.gaussian(img_grey,sigma2)
                 
                 dog = s1 - s2
                 #xdog= s2-t*s1
                 scipy.misc.imsave("DoG_outputt1.png",dog)
                 scipy.misc.imsave("DoG_outputt0.png",s1)
                 s11  = Image.open("DoG_outputt0.png")
                 dog1 = Image.open("DoG_outputt1.png")
                 dog2=dog1.load()
                 s111=s11.load()
                 height1,width1=dog1.size
                 xdog = Image.new('I',(height1,width1))
                 
                 for ii in range (0,height1):
                     for jj in range (0,width1):
                         
                       xdog.putpixel((ii,jj),math.ceil(((1-t)*s111[ii,jj])+(t*dog2[ii,jj])))
                       
                       
                 scipy.misc.imsave("DoG_outputt.png",xdog)
                 img2 = Image.open("DoG_outputt.png")
                 pix_val = img2.load()
                 height,width=img2.size
                 output_image = Image.new('I',(height,width))
                 
                 print(255*math.ceil(tanh(phi*(pix_val[0,0]-threshold))))
                 
                 for i in range(0,height):
                    for j in range(0,width):
                        
                       if (pix_val[i,j])/255>threshold:
                           output_image.putpixel((i,j),0)
                           
                       elif    (pix_val[i,j])/255<=threshold: 
                           output_image.putpixel((i,j),255*math.ceil(tanh(phi*(pix_val[i,j]-threshold))))
                 plt.imshow(output_image)          
                 scipy.misc.imsave("DoG_outputtt.png",output_image)
                 pixmap2 = QtGui.QPixmap("DoG_outputtt.png")
#________________________________________________________________________________________________________________________                   
                    
                    
                pixmap2 = pixmap2.scaled(self.label_filters_output.width(), self.label_filters_output.height(), QtCore.Qt.KeepAspectRatio) # Scale pixmap
                self.label_filters_output.setPixmap(pixmap2) # Set the pixmap onto the label
                self.label_filters_output.setAlignment(QtCore.Qt.AlignCenter) # Align the label to center""


#===========================================(Median filter)==========================================================  
          def Median_filter(self):
             
                img = input_image
                members = [(0,0)] * 9 #creat array=9 of (0,0)
                val=int(self.lineEdit_3.text())
                output_image = Image.new("RGB",(img.width,img.height),"white")#white image of same size as the input image
            
                for i in range(val,img.width-val):
                    for j in range(val,img.height-val):
                        members[0] = img.getpixel((i-val,j-val))
                        members[1] = img.getpixel((i-val,j))
                        members[2] = img.getpixel((i-val,j+val))
                        members[3] = img.getpixel((i,j-val))
                        members[4] = img.getpixel((i,j))
                        members[5] = img.getpixel((i,j+val))
                        members[6] = img.getpixel((i+val,j-val))
                        members[7] = img.getpixel((i+val,j))
                        members[8] = img.getpixel((i+val,j+val))
                        members.sort()
                        output_image.putpixel((i,j),(members[4]))
                output_image.save("Median_filter_output.png")
                pixmap2 = QtGui.QPixmap("Median_filter_output.png") # Setup pixmap with the provided image
                pixmap2 = pixmap2.scaled(self.label_filters_output.width(), self.label_filters_output.height(), QtCore.Qt.KeepAspectRatio) # Scale pixmap
                self.label_filters_output.setPixmap(pixmap2) # Set the pixmap onto the label
                self.label_filters_output.setAlignment(QtCore.Qt.AlignCenter) # Align the label to center""
         
            
#===============================================(Histogram equalization and matching)==========================================
         
            
          def histImage(self):#browse for histogram equalization image
                    global pixmap3 ,pixmapCopy ,fileName2 ,loaded_image1
                    fileName2, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image", "", "Image Files (*.png *.jpg *jpeg *.bmp)") # Ask for file
                    loaded_image1 = scipy.misc.imread(fileName2, flatten=True)
                    loaded_image2 =  mpimg.imread(fileName2)#, flatten=True)
                    
                    def rgb2gray(rgb):
                      return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])  
                  
                    grey=rgb2gray(loaded_image2)
                    scipy.misc.imsave("histImage.png",grey)
                    pixmap3 = QtGui.QPixmap("histImage.png") # Setup pixmap with the provided image 
                    pixmap3 = pixmap3.scaled(self.label_histograms_input.width(), self.label_histograms_input.height(), QtCore.Qt.KeepAspectRatio) # Scale pixmap
                    self.label_histograms_input.setPixmap(pixmap3) # Set the pixmap onto the label  
                    self.label_histograms_input.setAlignment(QtCore.Qt.AlignCenter) # Align the label to center""
          
          def histImageMatch(self):#browse for histogram matching image
                    global pixmap4 ,pixmapCopy ,fileName3 ,loaded_image2
                    fileName3, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image", "", "Image Files (*.png *.jpg *jpeg *.bmp)") # Ask for file
                    loaded_image2 = scipy.misc.imread(fileName3, flatten=True)
                    loaded_image5 =  mpimg.imread(fileName3)#, flatten=True)
                    
                    def rgb2gray(rgb):
                      return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])  
                  
                    grey=rgb2gray(loaded_image5)
                    scipy.misc.imsave("histEImage.png",grey)
                    pixmap4 = QtGui.QPixmap("histEImage.png") # Setup pixmap with the provided image 
                    pixmap4 = pixmap4.scaled(self.label_histograms_hinput.width(), self.label_histograms_hinput.height(), QtCore.Qt.KeepAspectRatio) # Scale pixmap
                    self.label_histograms_hinput.setPixmap(pixmap4) # Set the pixmap onto the label  
                    self.label_histograms_hinput.setAlignment(QtCore.Qt.AlignCenter) # Align the label to center""

            
          def histogramEqualization(self):
                global cum_hist,hist,hb,pixels,Hheight,Hwidth
                Hheight = loaded_image1.shape[0]
                Hwidth = loaded_image1.shape[1]
                pixels = Hwidth * Hheight
                hist = np.zeros((256))
                
                for i in np.arange(Hheight):
                    for j in np.arange(Hwidth):
                        a = int(loaded_image1.item(i,j))
                        hist[a] += 1
                cum_hist = hist.copy()
                
                for i in np.arange(1, 256):
                    cum_hist[i] = cum_hist[i-1] + cum_hist[i]
                    
                for i in np.arange(Hheight):
                    for j in np.arange(Hwidth):
                        a = int (loaded_image1.item(i,j))
                        hb = math.floor( (cum_hist[a] * 255.0 / pixels))
                        loaded_image1.itemset((i,j), hb)
                scipy.misc.imsave("Histogram_equalization_output.png",loaded_image1)
                Hpixmap = QtGui.QPixmap("Histogram_equalization_output.png") # Setup pixmap with the provided image
                Hpixmap = Hpixmap.scaled(self.label_histograms_output.width(), self.label_histograms_output.height(), QtCore.Qt.KeepAspectRatio) # Scale pixmap
                self.label_histograms_output.setPixmap(Hpixmap) # Set the pixmap onto the label
                self.label_histograms_output.setAlignment(QtCore.Qt.AlignCenter) # Align the label to center""
                return cum_hist/pixels
                                
          def histogramMatching(self): 
                
                
                global cum_hist1,hist1
                height = loaded_image2.shape[0]
                width = loaded_image2.shape[1]
                pixels1 = width * height
                hist1 = np.zeros((256))
                
                for i in np.arange(height):
                    for j in np.arange(width):
                        a = int(loaded_image2.item(i,j))
                        hist1[a] += 1
                cum_hist1 = hist1.copy()
                
                for i in np.arange(1, 256):
                    cum_hist1[i] = cum_hist1[i-1] + cum_hist1[i]
                    
                for i in np.arange(height):
                    for j in np.arange(width):
                        a = int (loaded_image2.item(i,j))
                        b = math.floor( (cum_hist1[a] * 255.0 / pixels))
                        loaded_image2.itemset((i,j), b)
                        
                prob_cum_hist = self.histogramEqualization() #get relative cumulative of my image
                prob_cum_hist_ref = cum_hist1 / pixels1      #get relative cumulative of the desired image
                        
                K = 256 #to intialize matrix to save intentisies of the new image
                new_values = np.zeros((K))
                
                for a in np.arange(K):
                    j = K - 1
                    while True:
                        new_values[a] = j
                        j = j - 1
                        if j < 0 or prob_cum_hist[a] > prob_cum_hist_ref[j]:
                            break
                
                for i in np.arange(Hheight):
                    for j in np.arange(Hwidth):
                        a = int(loaded_image1.item(i,j))
                        b = new_values[a]
                        loaded_image1.itemset((i,j), b)   
                scipy.misc.imsave("Histogram_matching_output.png",loaded_image1)
                HMpixmap = QtGui.QPixmap("Histogram_matching_output.png") # Setup pixmap with the provided image
                HMpixmap = HMpixmap.scaled(self.label_histograms_houtput.width(), self.label_histograms_houtput.height(), QtCore.Qt.KeepAspectRatio) # Scale pixmap
                self.label_histograms_houtput.setPixmap(HMpixmap) # Set the pixmap onto the label
                self.label_histograms_houtput.setAlignment(QtCore.Qt.AlignCenter) # Align the label to center""

#===================================================(HoughTransform)===========================================

          def openImage_Hough_Lines(self):
                   
                    global pixmap ,pixmapCopy ,fileName5 ,input_image,input_pixels
                    fileName5, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image", "", "Image Files (*.png *.jpg *jpeg *.bmp)") # Ask for file
                    image_process = scipy.misc.imread(fileName5, flatten=True, mode=None)
                    input_image = Image.open(fileName5)
                    input_pixels = input_image.load()
                    pixmap = QtGui.QPixmap(fileName5) # Setup pixmap with the provided image
                    pixmap = pixmap.scaled(self.label_lines_input.width(), self.label_lines_input.height(), QtCore.Qt.KeepAspectRatio) # Scale pixmap
                    self.label_lines_input.setPixmap(pixmap) # Set the pixmap onto the label  
                    self.label_lines_input.setAlignment(QtCore.Qt.AlignCenter) # Align the label to center""
                    Main.openImage_Hough_Lines_process(image_process)
                    
                    #OUTPUT
                    pixmap1 = QtGui.QPixmap("Hough_output.png") # Setup pixmap with the provided image
                    pixmap1 = pixmap1.scaled(self.label_lines_input_2.width(), self.label_lines_input_2.height(), QtCore.Qt.KeepAspectRatio) # Scale pixmap
                    self.label_lines_input_2.setPixmap(pixmap1) # Set the pixmap onto the label  
                    self.label_lines_input_2.setAlignment(QtCore.Qt.AlignCenter) # Align the label to center""
                    #Hough Space
                    pixmap1 = QtGui.QPixmap("Hough_Space.png") # Setup pixmap with the provided image
                    pixmap1 = pixmap1.scaled(self.label_lines_hough.width(), self.label_lines_hough.height(), QtCore.Qt.KeepAspectRatio) # Scale pixmap
                    self.label_lines_hough.setPixmap(pixmap1) # Set the pixmap onto the label  
                    self.label_lines_hough.setAlignment(QtCore.Qt.AlignCenter) # Align the label to center""
                    
                    #CANNY EDGES
                    pixmap1 = QtGui.QPixmap("edges.png") # Setup pixmap with the provided image
                    pixmap1 = pixmap1.scaled(self.label_lines_hough_2.width(), self.label_lines_hough_2.height(), QtCore.Qt.KeepAspectRatio) # Scale pixmap
                    self.label_lines_hough_2.setPixmap(pixmap1) # Set the pixmap onto the label  
                    self.label_lines_hough_2.setAlignment(QtCore.Qt.AlignCenter) # Align the label to center""
         

          def openImage_Hough_Lines_process(input_image):
                    
            h, theta, d = hough_line(input_image)
            edges = canny(input_image, 2, 1, 25)
            lines = probabilistic_hough_line(edges, threshold=10, line_length=5,line_gap=3)
            plt.clf();
            plt.imshow(np.log(1 + h),extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), d[-1], d[0]],
                         cmap=cm.gray, aspect=1/10)
            plt.title('Hough Space')
            plt.savefig("Hough_Space.png")

            #Canny Edge Detection
            plt.clf();
            plt.imshow(edges, cmap=cm.gray)
            plt.title('CannyEdges')
            plt.savefig("CannyEdges.png")
            
            
            #OUTPUT
            plt.clf();
            for line in lines:
                p0, p1 = line
                plt.plot((p0[0], p1[0]), (p0[1], p1[1]))
                plt.title('output')
                plt.imsave('edges.png', edges , cmap='gray', format='png')
                plt.savefig("Hough_output.png")

#==================================================(Hough circles)==============================================
          def openImage_Hough_Circles(self):
                    global pixmap6 ,pixmapCopy ,fileName6 ,input_image,input_pixels
                    fileName6, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image", "", "Image Files (*.png *.jpg *jpeg *.bmp)") # Ask for file
                    input_image = Image.open(fileName6)
                    input_pixels = input_image.load()
                    pixmap6 = QtGui.QPixmap(fileName6) # Setup pixmap with the provided image
                    pixmap6 = pixmap6.scaled(self.label_circles_input.width(), self.label_circles_input.height(), QtCore.Qt.KeepAspectRatio) # Scale pixmap
                    pixmapCopy=pixmap6.copy();
                    self.label_circles_input.setPixmap(pixmap6) # Set the pixmap onto the label  
                    self.label_circles_input.setAlignment(QtCore.Qt.AlignCenter) # Align the label to center""
                    Main.openImage_Hough_Circles_process(input_image)
                    
                    #OUTPUT
                    pixmap1 = QtGui.QPixmap("Hough_Circles_output.png") # Setup pixmap with the provided image
                    pixmap1 = pixmap1.scaled(self.label_circles_hough.width(), self.label_circles_hough.height(), QtCore.Qt.KeepAspectRatio) # Scale pixmap
                    self.label_circles_hough.setPixmap(pixmap1) # Set the pixmap onto the label  
                    self.label_circles_hough.setAlignment(QtCore.Qt.AlignCenter) # Align the label to center""                
                
          def openImage_Hough_Circles_process(input_image):                                   
                    # Output image:
                    output_image = Image.new("RGB", input_image.size)
                    output_image.paste(input_image)
                    draw_result = ImageDraw.Draw(output_image)                
                    # Find circles
                    rmin = 12
                    rmax = 14
                    steps = 100
                    threshold = 0.36
                    points = []
                    for r in range(rmin, rmax + 1):
                        for t in range(steps):
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
                        draw_result.ellipse((x-r, y-r, x+r, y+r), outline=(0,0,255,255))                
                    # Save output image
                    output_image.save("Hough_Circles_output.png")                
                
                
#____________________________________________________________________________________________________________________________                           
                
  
if __name__ == "__main__":
    app = 0  # This is the solution As the Kernel died every time I restarted the consol
    argument=0
    app = QApplication(sys.argv)
    widget = Main()
    widget.show()
    sys.exit(app.exec_())