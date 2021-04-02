#file 2 -- use_filename1.py --
#first method
from libraries  import *

Window = True
event = True
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


class Global(object):
    arrX=[]
    arrY=[]
    filname_snake_image=0
        
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
      super(Main, self).__init__()
      loadUi('mainwindow.ui.autosave.ui', self)      
      self.load_snake_image.clicked.connect(self.openImage_snake_algorithm)
      self.Contour_snake_image.clicked.connect(self.buttonClicked_snake_algorithm)
      self.input_snake_image.mousePressEvent = self.getPixel_snake_label_image

        
    def initUI(self):      
        
        grid = QGridLayout()
        
        btn1 = QPushButton("Button 1", self)
        btn1.move(30, 50)
        btn1.clicked.connect(self.buttonClicked)            

        x = 0
        y = 0
        
        self.text = "x: {0},  y: {1}".format(x, y)
        
        self.label = QLabel(self.text, self)
        grid.addWidget(self.label, 0, 0, Qt.AlignTop)
        
        self.setMouseTracking(True)
        
        self.setLayout(grid)
        
        self.setGeometry(300, 300, 350, 200)
        self.setWindowTitle('Event object')
        self.show()
        
        
    def alpha_slider(self, e):
        size = self.alphaslider.value()
        
        print("alphaslider  : ".size)
        self.ALPHA_vaLUE.setText(size/10)
    
    def getPixel_snake_label_image(self, e):
        x = e.x()  - 40
        if(e.y() > 10):
             y = e.y()  - 10
        else:
             y = e.y()   
        print("x: {0},  y: {1}".format(x, y))

        Global.arrX.append(x)
        Global.arrY.append(y)
        
    def buttonClicked_snake_algorithm(self):
        
        min_X= np.min(Global.arrX) 
        max_X= np.max(Global.arrX) 
        min_Y= np.min(Global.arrY) 
        max_Y= np.max(Global.arrY)         
        if(max_X > max_Y):
            radiuse  =  ((max_X - min_X)/2.6)
        else:
            radiuse  =  ((max_Y - min_Y)/2.6)
     
        centerOfX=  ((max_X + min_X)/2)
        centerOfY=  ((max_Y + min_Y)/2)
        
        img = Global.filname_snake_image
        img = rgb2gray(img)
        s = np.linspace(0, 2*np.pi, 400)
        x = centerOfX + radiuse*np.cos(s)
        y = centerOfY + radiuse*np.sin(s)
        init = np.array([x, y]).T

        a = float(self.alpha.text())
        b = float(self.beta.text())
        g = float(self.gamma.text())

        snake = active_contour(gaussian(img, 3),
                       init, alpha=a, beta=b, gamma=g)

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(img, cmap=plt.cm.gray)
        ax.plot(init[:, 0], init[:, 1], '--r', lw=3)
        ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3)
        ax.set_xticks([]), ax.set_yticks([])
        ax.axis([0, img.shape[1], img.shape[0], 0])
        plt.savefig('Snake Output.png')
        pixmap = QtGui.QPixmap('Snake Output.png') # Setup pixmap with the provided image
        pixmap = pixmap.scaled(self.output_snake_image.width(), self.output_snake_image.height(), QtCore.Qt.KeepAspectRatio) # Scale pixmap
        pixmapCopy=pixmap.copy();
        self.output_snake_image.setPixmap(pixmap) # Set the pixmap onto the label  
        self.output_snake_image.setAlignment(QtCore.Qt.AlignCenter) # Align the label to center""
        Global.arrX.clear()
        Global.arrY.clear()
    def openImage_snake_algorithm(self):
        global pixmap ,pixmapCopy ,fileName1 ,input_image,input_pixels
        fileName1, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image", "", "Image Files (*.png *.jpg *jpeg *.bmp)") # Ask for file
        image_process = scipy.misc.imread(fileName1, mode=None)
        Global.filname_snake_image = image_process
        input_pixels = input_image.load()
        pixmap = QtGui.QPixmap(fileName1) # Setup pixmap with the provided image
        pixmap = pixmap.scaled(self.input_snake_image.width(), self.input_snake_image.height(), QtCore.Qt.KeepAspectRatio) # Scale pixmap
        pixmapCopy=pixmap.copy();
        self.input_snake_image.setPixmap(pixmap) # Set the pixmap onto the label  
        self.input_snake_image.setAlignment(QtCore.Qt.AlignCenter) # Align the label to center""
        
    

if __name__ == "__main__":
    app = 0  # This is the solution As the Kernel died every time I restarted the consol
    argument=0
    app = QApplication(sys.argv)
    widget = Main()
    widget.show()
    

   
    sys.exit(app.exec_())