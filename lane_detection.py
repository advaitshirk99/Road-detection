import os
import re
import cv2
import numpy as np
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt

img1 = cv2.imread('D:/Road detecion/frames/0.png')
img = cv2.resize(img1, (480,270))

grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray image',grayImage)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(grayImage.shape)

def create_mask(grayimg):
	img_mask = np.zeros_like(grayimg)
	lane_shape = np.array([[50,270], [220,160], [360,160], [480,270]]) 
	cv2.fillConvexPoly(img_mask, lane_shape, 255)
	print(img_mask)

	cv2.imshow('Image mask',img_mask)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
	masked_img = cv2.bitwise_and(grayimg, grayimg, mask = img_mask)	
	cv2.imshow('Masked Image',masked_img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	ret, thresh_img = cv2.threshold(masked_img, 130, 255, cv2.THRESH_BINARY)
	cv2.imshow('Threshold image',thresh_img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()	

	img_lines = cv2.HoughLinesP(thresh_img, 1, np.pi/180, 30, maxLineGap=200)
	dummy_img = grayimg.copy()
	for i in img_lines:
		x1, y1, x2, y2 = i[0]
		cv2.line(dummy_img, (x1, y1), (x2, y2), (255, 0, 0), 3)

	cv2.imshow('Hough Lines image',dummy_img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


create_mask(grayImage)