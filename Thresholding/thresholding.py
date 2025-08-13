import numpy as np
import matplotlib.pyplot as plt
import cv2
def func1(img):
	m = 128
	height ,width = img.shape
	img1 = np.zeros_like(img)
	for i in range(height):
		for j in range (width):
			r = img[i][j]
			if(r>=m):
				img1[i][j] = 1;
			else:
				img1[i][j] = 0;
	return img1
def func2(img):
	m = 1.02
	c = 0
	thres1 = 128
	thres2 = 196
	height ,width = img.shape
	img1 = np.zeros_like(img)
	for i in range (height):
		for j in range(width):
			r = img[i,j]
			if (r<thres1):
				img1[i,j] =m*r+c
			elif r<thres2:
				img1[i,j] = (0.5*255)
			else:
			 img1[i,j] = 255
	return img1

def func3(img):
	m = 1.05
	c = 5
	thres1 = 50
	thres2 = 196
	height,width = img.shape
	img1 = np.zeros_like(img)
	for i in range (height):
		for j in range(width):
			r = img[i,j]
			if(r<=thres1):
				img1[i,j] = 0
			elif (r<=thres2):
				img1[i,j] = m*r+c
			else:
				img1[i,j] = (0.75*255)
	return img1
	
def main():
	img1 = cv2.imread('/home/alamin/1.PART_IV/DIP/DIP_basic(lec 1)/Assignments/Thresholding/child1.jpg',0)
	img11 = func1(img1)
	img12  = func2(img1)
	img13 = func3(img1)

	img_set = [img1,img11,img12,img13]
	title_set = ["original image","applying step func","applying func2","applying func3"]
	
	for i in range (1,len(img_set)+1):
		plt.subplot(2,4,i)
		plt.title(title_set[i-1])
		plt.imshow(img_set[i-1],cmap='gray')
		plt.subplot(2,4,i+4)
		plt.hist(img_set[i-1].ravel(), bins = 256, color='gray') # dot ravel krar karon 1D te niye asa
	
	plt.show()

	
if __name__ =='__main__':
	main()
