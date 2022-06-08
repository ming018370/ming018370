import cv2
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

img = cv2.imread('lines.png',-1)
cv2.imshow('line',img)
print(img.shape)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(5,5),0)
canny = cv2.Canny(img,50,150)
cv2.imshow('draw_line',canny)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(5,5),0)
canny = cv2.Canny(img,50,150)

lines = cv2.HoughLinesP(canny,1,np.pi/180,50,maxLineGap=180,minLineLength=20)
for line in lines:
    x1,y1,x2,y2 = line[0]
    cv2.line(img,(x1,y1),(x2,y2),(255,0,0),3)
    
cv2.imshow('draw_line',img)

lane_img = cv2.imread('lane.png',-1)
hsv = cv2.cvtColor(lane_img,cv2.COLOR_BGR2HSV)
low_yellow = np.array([15,90,140])
high_yellow = np.array([50,160,255])
mask = cv2.inRange(hsv,low_yellow,high_yellow)

lane_img = cv2.imread('lane.png',-1)
blur = cv2.GaussianBlur(lane_img,(5,5),0)
hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)

low_yellow = np.array([15,90,140])
high_yellow = np.array([50,160,255])
mask = cv2.inRange(hsv,low_yellow,high_yellow)

canny = cv2.Canny(mask,50,150)

lines = cv2.HoughLinesP(canny,1,np.pi/180,50,maxLineGap=50,minLineLength=20)
for line in lines:
    x1,y1,x2,y2 = line[0]
    cv2.line(lane_img,(x1,y1),(x2,y2),(255,0,0),3)

cv2.imshow('lane_img',lane_img)
