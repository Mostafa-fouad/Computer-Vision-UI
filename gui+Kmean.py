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
    
class Main(QMainWindow):
    
    
    def __init__(self):

      super(Main, self).__init__()
      loadUi('mainwindow.ui.autosave.ui', self)      
      self.load_snake_image.clicked.connect(self.openImage_snake_algorithm)
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
        x =  e.x()  /4
        y =  e.y()  /4

        print("x: {0},  y: {1}".format(x, y))

        Global.arrX_kmean_points.append(x)
        Global.arrY_kmean_points.append(y)
        
        

    def buttonClicked_Kmean_output(self):
        C = np.array(list(zip(Global.arrX_kmean_points, Global.arrY_kmean_points)), dtype=np.float32)
        print("Initial Centroids")
        print(C)
        print("type of :" , type(C))

        X = np.array(list(zip(Global.f1, Global.f2)))
        print(X)
        k= len(Global.arrX_kmean_points)
        print("size of array :",len(Global.arrX_kmean_points))
        # Euclidean Distance Caculator

        # Plotting along with the Centroids
        plt.scatter(Global.f1, Global.f2, c='#050505', s=7)
        plt.scatter(Global.arrX_kmean_points, Global.arrY_kmean_points, marker='*', s=200, c='g')
        def dist(a, b, ax=1):

            return np.linalg.norm(a - b, axis=ax)        
        # To store the value of centroids when it updates
        C_old = np.zeros(C.shape)
        # Cluster Lables(0, 1, 2)
        clusters = np.zeros(len(X))
        # Error func. - Distance between new centroids and old centroids
        error = dist(C, C_old, None)
        # Loop will run till the error becomes zero

        while error != 0:
            # Assigning each value to its closest cluster
            for i in range(len(X)):
                distances = dist(X[i], C)
                cluster = np.argmin(distances)
                clusters[i] = cluster
            # Storing the old centroid values
            C_old = deepcopy(C)
            # Finding the new centroids by taking the average value
            for i in range(k):
                points = [X[j] for j in range(len(X)) if clusters[j] == i]
                C[i] = np.mean(points, axis=0)
            error = dist(C, C_old, None)
        print("helloooooooo")

        colors = ['r', 'g', 'b', 'y', 'c', 'm']
        fig, ax = plt.subplots()
        
        for i in range(k):
                points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
                plt.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
                print("hellloooooo")
                plt.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='#050505')
    
        plt.savefig("K-means\output")
        pixmap = QtGui.QPixmap('K-means\output.PNG') # Setup pixmap with the provided image
        pixmap = pixmap.scaled(self.output_Kmean_image.width(), self.output_Kmean_image.height(), QtCore.Qt.KeepAspectRatio) # Scale pixmap
        pixmapCopy=pixmap.copy();
        self.output_Kmean_image.setPixmap(pixmap) # Set the pixmap onto the label  
        self.output_Kmean_image.setAlignment(QtCore.Qt.AlignCenter) # Align the label to center""
    
        

    def openImage_snake_algorithm(self):
        global pixmap ,pixmapCopy ,fileName1 ,input_image,input_pixels
        fileName1, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image", "", "Image Files (*.png *.jpg *jpeg *.bmp)") # Ask for file
        image_process = scipy.misc.imread(fileName1, flatten=True, mode=None)
        input_image = Image.open(fileName1)
        input_pixels = input_image.load()
        pixmap = QtGui.QPixmap(fileName1) # Setup pixmap with the provided image
        pixmap = pixmap.scaled(self.input_snake_image.width(), self.input_snake_image.height(), QtCore.Qt.KeepAspectRatio) # Scale pixmap
        pixmapCopy=pixmap.copy();
        self.input_snake_image.setPixmap(pixmap) # Set the pixmap onto the label  
        self.input_snake_image.setAlignment(QtCore.Qt.AlignCenter) # Align the label to center""
        
    
    def openImage_Kmean_algorithm(self):
        global pixmap ,pixmapCopy ,fileName1 ,input_image,input_pixels
        fileName1, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image", "", "Image Files (*.csv)") # Ask for file
        plt.rcParams['figure.figsize'] = (16, 9.8)
        plt.style.use('ggplot')
        # Importing the dataset
        data = pd.read_csv(fileName1)
        print("Input Data and Shape")
        print(data.shape)
        data.head()       
        # Getting the values and plotting it
        Global.f1 = data['V1'].values
        Global.f2 = data['V2'].values
        X = np.array(list(zip(Global.f1, Global.f2)))
        plt.scatter(Global.f1, Global.f2, c='black', s=7)
                
        #plt.style.use('ggplot')
        plt.savefig("K-means\initial_kmean_image")
        loadedImage = Image.open('K-means\initial_kmean_image.PNG')
        
        savedImage = loadedImage.resize((391,241),Image.ANTIALIAS)
        scipy.misc.imsave("K-means\initial_kmean_image_resize.PNG",savedImage)   
            
        
        pixmap = QtGui.QPixmap('K-means\initial_kmean_image_resize.PNG') # Setup pixmap with the provided image
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