from cv2 import cv2
import numpy as np
from matplotlib import pyplot as plt

# loading image
#img0 = cv2.imread('SanFrancisco.jpg',)
def laplace(img0):

    # converting to gray scale
    gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)

    # remove noise
    img = cv2.GaussianBlur(gray,(3,3),0)

    # convolute with proper kernels
    laplacian = cv2.Laplacian(img,cv2.CV_64F)
    return laplacian

def sobelx(img0):

    # converting to gray scale
    gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)

    # remove noise
    img = cv2.GaussianBlur(gray,(3,3),0)

    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)  # x
    return sobelx

def sobely(img0):

    # converting to gray scale
    gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)

    # remove noise
    img = cv2.GaussianBlur(gray,(3,3),0)

    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)  # y
    return sobely

def plot(img, filtered):
    plt.subplot(1,2,1),plt.imshow(img,cmap = 'gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(1,2,2),plt.imshow(filtered,cmap = 'gray')
    plt.title('Filtered'), plt.xticks([]), plt.yticks([])
    plt.show()
    