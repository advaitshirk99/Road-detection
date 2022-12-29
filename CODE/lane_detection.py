#Importing all necessary libraries
import os								#Library for os functions like time
import re								#Library for regular expressions
import cv2								#OpenCV
import numpy as np
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt

img = cv2.imread('D:/Academic stuff/MSIS Github/Mini-project-data/Camera data for unstructured roads/IMG20221011121313.jpg')
resized_img = cv2.resize(img, (480,270))

def create_mask(resized_img):
	grayImage = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)					#opencv images are in bgr format by default
	cv2.imwrite('Unstructured_first.jpg', grayImage)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	#print(grayImage.shape)

	## CREATING MASK FOR THE ROAD ##
	img_mask = np.zeros_like(grayImage)
	lane_shape = np.array([[0,270], [220,160], [360,160], [480,270]]) 			#Creating a mask for the road
	cv2.fillConvexPoly(img_mask, lane_shape, 255)								#The generated mask will be white
 
	cv2.imshow('Image mask',img_mask)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
	## THRESHOLDING THE MASK ##
	masked_img = cv2.bitwise_and(grayImage, grayImage, mask = img_mask)	
	cv2.imshow('Masked Image',masked_img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	ret, thresh_img = cv2.threshold(masked_img, 130, 255, cv2.THRESH_BINARY)
	cv2.imshow('Threshold image',thresh_img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()	

	## HOUGH LINES ##
	img_lines = cv2.HoughLinesP(thresh_img, 1, np.pi/180, 30, maxLineGap=200)
	dummy_img = grayImage.copy()
	for i in img_lines:
		x1, y1, x2, y2 = i[0]
		cv2.line(dummy_img, (x1, y1), (x2, y2), (255, 0, 0), 3)

	cv2.imwrite('Unstructured_final.jpg',dummy_img)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()


create_mask(resized_img)