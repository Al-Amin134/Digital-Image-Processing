import matplotlib.pyplot as plt
import numpy as np
import cv2

def sobel_x_func(img):
	return np.array([[-1,0,1],
			 [-2,0,2],
			 [-1,0,1]
			 ])
def sobel_y_func(img):
	return np.array([[-1,-2,-1],
			[0,0,0],
			[1,2,1]
			])
			
def my_filter2D(img,kernel):
	img = img.astype(np.float32)
	k_h,k_w = kernel.shape
	pad_h, pad_w = k_h//2,k_w//2
	
	padded = np.pad(img,((pad_h,pad_h),(pad_w,pad_w)),mode="reflect")
	output = np.zeros_like(img)
	
	for i in range(img.shape[0]):
		for j in range (img.shape[1]):
			region = padded[i:i+k_h,j:j+k_w]
			output[i,j] = np.sum(region*kernel)
	return output.astype(np.uint64)
	
def main():
	img = cv2.imread("child1.jpg",0)
	sobel_x = sobel_x_func(img)
	sobel_y  = sobel_y_func(img)
	sobelx_img = my_filter2D(img,sobel_x)
	sobely_img = my_filter2D(img,sobel_y)
	
	plt.subplot(2,1,1)
	plt.title("Sobel x")
	plt.imshow(sobelx_img, cmap="gray")
	
	plt.subplot(2,1,2)
	plt.title("Sobel y")
	plt.imshow(sobely_img,cmap="gray")
	plt.show()
	
if __name__ == "__main__":
	main()
