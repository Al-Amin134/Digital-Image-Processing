"""
Write a report based on your observation after performing spatial filtering by OpenCV's built-in function cv2.filter2D() using: 
a smoothing or average kernel [e.g., a kernel with 1 only]
a Sobel kernel in x-direction and a Sobel kernel in y-direction
a Prewitt kernel in x-direction and a Prewitt kernel in y-direction
a Laplace kernel
Note: Report should be prepared by Latex, code link should be included in the report instead of code
"""


import numpy as np
import matplotlib.pyplot as plt
import cv2

def main():

	img = cv2.imread("lotus.jpg",cv2.IMREAD_GRAYSCALE)
	gray_img =cv2.resize(img,(400,400))
	
	# a smoothing or average kernel 
	avg_kernel = (1/9)*np.array([[1,1,1],
				[1,1,1],
				[1,1,1]],dtype=np.float32)
	
	# a Sobel kernel in x-direction 
	sobel_x = np.array([[-1,0,1],
				[-2,0,2],
				[-1,0,1]],dtype=np.float32)
				
	
	# a Sobel kernel in y-direction 
	sobel_y = np.array([[1,2,1],
				[0,0,0],
				[-1,-2,-1]],dtype=np.float32)
				
	#a Prewitt kernel in x-direction
	prewitt_x = np.array([[-1,0,1],
				[-1,0,1],
				[-1,0,1]],dtype=np.float32)
	
	#a Prewitt kernel in y-direction			
	prewitt_y = np.array([[1,1,1],
				[0,0,0],
				[-1,-1,-1]],dtype=np.float32)
	#a Laplace kernel
	laplace = np.array([[ 0, -1,  0],
                		    [-1,  4, -1],
                 		   [ 0, -1,  0]], dtype=np.float32)
                    		
	sobel_x_img = cv2.filter2D(gray_img,-1,sobel_x)
	sobel_y_img = cv2.filter2D(gray_img,-1,sobel_y)
	prewitt_x_img = cv2.filter2D(gray_img,-1,prewitt_x)
	prewitt_y_img = cv2.filter2D(gray_img,-1,prewitt_y)
	laplace_img = cv2.filter2D(gray_img,-1,laplace)
	smooth_img = cv2.filter2D(gray_img, -1, avg_kernel)
	
	img_set = [gray_img,sobel_x_img, sobel_y_img, prewitt_x_img, prewitt_y_img, laplace_img,smooth_img]
	title_set = ['main_image', 'sobel_x', 'sobel_y',' prewitt_x',' prewitt_y', 'laplace', "After Smoothing"]
	
	n = len(img_set)
	for i in range(n):
		plt.subplot(3,3,i+1)
		plt.title(title_set[i])
		plt.imshow(img_set[i],  cmap= 'gray')
	
	plt.tight_layout()
	plt.show()
	
	
if __name__ == '__main__':
	main()
