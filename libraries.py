import sys
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QApplication, QGridLayout, QLabel,QPushButton
import matplotlib.pyplot as plt
from PyQt5.uic import loadUi
from PyQt5 import QtCore , QtGui ,QtWidgets
from PIL import Image,ImageDraw,ImageOps
from math import sqrt
from PyQt5.QtWidgets import QDialog, QMainWindow
from PIL.ImageQt import ImageQt
from PyQt5.QtWidgets import QMessageBox, QFileDialog, QApplication, QLCDNumber, QTextEdit, QLabel, QProgressBar
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.misc
import scipy.ndimage.filters
import scipy as sp
import scipy.ndimage as nd
from PIL import Image as im
from scipy.ndimage import gaussian_filter
import matplotlib.image as mpimg
from skimage import data, feature, color, filters
from skimage.io import imread
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from scipy import signal as sig
from skimage.transform import (hough_line, hough_line_peaks,
                               probabilistic_hough_line)
from skimage.feature import canny
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image, ImageDraw
from math import sqrt, pi, cos, sin
#from canny import canny_edge_detector
from collections import defaultdict
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt