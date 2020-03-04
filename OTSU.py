# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 10:11:02 2018

@author: sihua
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt
def otsu(image):
    height=image.shape[0]
    width=image.shape[1]
    n=height*width
    threshold=0
    picCount=[0]*256
    picPro=[0]*256
    for i in range(height):
        for j in range(width):
            temp=int(image[i][j])
            picCount[temp]=picCount[temp]+1
    for i in range(256):
        picPro[i]=picCount[i]/float(n)
    deltaMax=0
    for i in range(256):
        w0=w1=u0tmp=u1tmp=u0=u1=u=deltatmp=0
        for j in range(256):
            if j<=i:
                w0=w0+picCount[j]#picpre
                u0tmp=u0tmp+j*picPro[j]
            else:
                w1=w1+picCount[j]##picpre
                u1tmp=u1tmp+j*picPro[j]
             
        if w0==0:
            u0=0
        else:
            u0=u0tmp/w0
        if w1==0:
            u1=0
        else:
            u1=u1tmp/w1
        u=u0tmp+u1tmp
        deltatmp=w0*(u0-u)*(u0-u)+w1*(u1-u)*(u1-u)
        if deltatmp>deltaMax:
            deltaMax=deltatmp
            threshold=i
    return threshold
image=cv2.imread('D:\\boy.png')
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
k=otsu(gray)
print(k)
ret,t1=cv2.threshold(gray,k,255,cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,3,5)
th3 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,3,5)
plt.subplot(2,2,1),plt.imshow(t1,'gray')
plt.title('global otsu'),plt.xticks([]),plt.yticks([])
plt.subplot(2,2,2), plt.hist(image.ravel(), 256,[0,256])
plt.title("Histogram")
plt.xlim(0,256)
plt.subplot(2,2,3),plt.imshow(th2,'gray')
plt.title('Adaptive Mean Thresholding'),plt.xticks([]),plt.yticks([])
plt.subplot(2,2,4),plt.imshow(th3,'gray')
plt.title('Adaptive Gaussian Thresholding'),plt.xticks([]),plt.yticks([])

'''
plt.subplot(2,2,1),plt.imshow(image,'gray')
plt.title("source image"), plt.xticks([]), plt.yticks([])
#th1 = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,3,5)
th3 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,3,5)
#plt.subplot(3,2,6),plt.imshow(t1,'gray')
plt.subplot(2,2,2), plt.hist(image.ravel(), 256)
plt.title("Histogram"), plt.xticks([]), plt.yticks([])
#plt.title('Global Thresholding'),plt.xticks([]),plt.yticks([])
plt.subplot(2,2,3),plt.imshow(th2,'gray')
plt.title('Adaptive Mean Thresholding'),plt.xticks([]),plt.yticks([])
plt.subplot(2,2,4),plt.imshow(th3,'gray')
plt.title('Adaptive Gaussian Thresholding'),plt.xticks([]),plt.yticks([])

print("threshold",k)
'''
plt.show()


