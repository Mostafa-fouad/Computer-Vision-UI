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
    arrX_kmean_points=[]
    arrY_kmean_points=[]
    f1=0
    f2=0
    image=[]
    fileName1 = 0

class Main(QMainWindow):
    
    
    def __init__(self):

      super(Main, self).__init__()
      loadUi('mainwindow.ui.autosave.ui', self)      
      self.load_Kmean_image.clicked.connect(self.openImage_Kmean_algorithm)
      self.Kmean_initial_points.clicked.connect(self.buttonClicked_Kmean_output)
      self.input_Kmean_image.mousePressEvent = self.getPixel_kmean_label_image

        
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
        
        
    def mouseMoveEvent(self, e):
        x = e.x() 
        y = e.y() 
        
        text = "x: {0},  y: {1}".format(x, y)
        self.label.setText(text)
    
    def getPixel_kmean_label_image(self, e):
        x = e.x()  
        y = e.y()  

        print("x: {0},  y: {1}".format(x, y))

        Global.arrX_kmean_points.append(x)
        Global.arrY_kmean_points.append(y)
        
        

    def buttonClicked_Kmean_output(self):
       
        image = ndimage.imread(Global.fileName1)
        
        plt.figure(figsize = (15,8))
        plt.imshow(image)
        image.shape
        
        x, y, z = image.shape
        image_2d = image.reshape(x*y, z)
        image_2d.shape
        
        
        kmeans_cluster = cluster.KMeans(n_clusters=7)
        kmeans_cluster.fit(image_2d)
        cluster_centers = kmeans_cluster.cluster_centers_
        cluster_labels = kmeans_cluster.labels_
        
        plt.figure(figsize = (15,8))
        plt.imshow(cluster_centers[cluster_labels].reshape(x, y, z))
        plt.savefig("output")
        pixmap = QtGui.QPixmap('output.PNG') # Setup pixmap with the provided image
        pixmap = pixmap.scaled(self.output_Kmean_image.width(), self.output_Kmean_image.height(), QtCore.Qt.KeepAspectRatio) # Scale pixmap
        pixmapCopy=pixmap.copy();
        self.output_Kmean_image.setPixmap(pixmap) # Set the pixmap onto the label  
        self.output_Kmean_image.setAlignment(QtCore.Qt.AlignCenter) # Align the label to center""
        
    def openImage_snake_algorithm(self):
        global pixmap ,pixmapCopy ,fileName1 ,input_image,input_pixels
        fileName1, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image", "", "Image Files (*.png *.jpg *jpeg *.bmp)") # Ask for file
        Global.fileName1 =fileName1

        input_image = Image.open(fileName1)
        input_pixels = input_image.load()
        pixmap = QtGui.QPixmap(fileName1) # Setup pixmap with the provided image
        pixmap = pixmap.scaled(self.input_snake_image.width(), self.input_snake_image.height(), QtCore.Qt.KeepAspectRatio) # Scale pixmap
        pixmapCopy=pixmap.copy();
        self.input_snake_image.setPixmap(pixmap) # Set the pixmap onto the label  
        self.input_snake_image.setAlignment(QtCore.Qt.AlignCenter) # Align the label to center""
        
    
    def openImage_Kmean_algorithm(self):
        global pixmap ,pixmapCopy ,fileName1 ,input_image,input_pixels
        fileName1, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image", "", "Image Files (*.png *.jpg *jpeg *.bmp)") # Ask for file
        Global.fileName1 =fileName1

        
        pixmap = QtGui.QPixmap(fileName1) # Setup pixmap with the provided image
        pixmap = pixmap.scaled(self.input_Kmean_image.width(), self.input_Kmean_image.height(), QtCore.Qt.KeepAspectRatio) # Scale pixmap
        pixmapCopy=pixmap.copy();
        self.input_Kmean_image.setPixmap(pixmap) # Set the pixmap onto the label  
        self.input_Kmean_image.setAlignment(QtCore.Qt.AlignCenter) # Align the label to center""
        

if __name__ == "__main__":
    app = 0  # This is the solution As the Kernel died every time I restarted the consol
    argument=0
    app = QApplication(sys.argv)
    widget = Main()
    widget.show()
    

   
    sys.exit(app.exec_())