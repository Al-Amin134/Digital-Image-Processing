import matplotlib.pyplot as plt
import numpy as np
import cv2

def average_func(img):
	return (1/9)*np.array([[1,1,1],
			 [1,1,1],
			 [1,1,1]
			 ])

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

def prewitt_x_func(img):
	return np.array([[-1,0,1],
			 [-1,0,1],
			 [-1,0,1]
			 ])
			 
def prewitt_y_func(img):
	return np.array([[1,1,1],
			 [0,0,0],
			 [-1,-1,-1]
			 ])
			 
def scharr_x_func(img):
	return np.array([[-1,0,3],
			 [-10,0,10],
			 [-3,0,3]
			 ])
			 
def scharr_y_func(img):
	return np.array([[-3,-10,-3],
			 [0,0,0],
			 [3,10,3]
			 ])
			 
def laplace_func(img):
	return np.array([[0,-1,0],
			 [-1,4,-1],
			 [0,-1,0]
			 ])
def own_kernel1(img):
	return np.array([
			[-1,2,0],
			[5,-4,2],
			[0,1,-2]
			]) 

def own_kernel2(img):
	return np.array([
			[1,-2,3],
			[0,0,1],
			[0,-1,-2]
			]) 
			
def own_kernel3(img):
	return np.array([
			[1,-2,0],
			[0,-1,1],
			[2,0,2]
			]) 
			
def own_kernel4(img):
	return np.array([
			[1,2,3],
			[0,0,3],
			[-2,-1,-2]
			]) 
			
			 
def valid_filter_2D(img, kernel):
	img = img.astype(np.float32)
	

                			 			
def my_filter2D(img,kernel,mode):
	img = img.astype(np.float32)
	k_h,k_w = kernel.shape
	pad_h, pad_w = k_h//2,k_w//2
	
	if(mode=="same"):
		padded = np.pad(img,((pad_h,pad_h),(pad_w,pad_w)),mode="constant") #clahe
		output = np.zeros_like(img)
	
		for i in range(img.shape[0]):
			for j in range (img.shape[1]):
				region = padded[i:i+k_h,j:j+k_w]
				output[i,j] = np.sum(region*kernel)
		output = np.clip(output, 0, 255)
		
	elif (mode =="valid"):
		out_h = img.shape[0]-k_h+1
		out_w = img.shape[1]-k_w + 1
		output = np.zeros((out_h,out_w),dtype=np.float32)
		
		for i in range(out_h):
			for j in range(out_w):
				region = img[i:i+k_h, j:j+k_w]
				output[i,j] = np.sum(region * kernel)

		output = np.clip(output,0,255)
	return output.astype(np.uint8) 
	
def main():
	img = cv2.imread("/home/alamin/1.PART_IV/DIP/images/shape2.jpg",0)
	
	average = average_func(img)
	sobel_x = sobel_x_func(img)
	sobel_y  = sobel_y_func(img)
	prewitt_x = prewitt_x_func(img)
	prewitt_y = prewitt_y_func(img)
	scharr_x = scharr_x_func(img)
	scharr_y = scharr_y_func(img)
	laplace = laplace_func(img)
	kernel1 = own_kernel1(img)
	kernel2 = own_kernel2(img)
	kernel3 = own_kernel3(img)
	kernel4 = own_kernel4(img)
	
	
	mode = input("Enter the mode (must same or valid) : ")
	average_img = my_filter2D(img, average,mode)
	sobelx_img = my_filter2D(img,sobel_x,mode)
	sobely_img = my_filter2D(img,sobel_y,mode)
	prewittx_img = my_filter2D(img, prewitt_x,mode)
	prewitty_img = my_filter2D(img, prewitt_y,mode)
	scharrx_img = my_filter2D(img, scharr_x,mode)
	scharry_img = my_filter2D(img, scharr_y,mode)
	laplace_img = my_filter2D(img,laplace,mode)
	kernel1_image = my_filter2D(img,kernel1,mode)
	kernel2_image = my_filter2D(img,kernel2,mode)
	kernel3_image = my_filter2D(img,kernel3,mode)
	kernel4_image = my_filter2D(img,kernel4,mode)
	
	
	
	img_set = [img,average_img,sobelx_img,sobely_img,prewittx_img,prewitty_img,scharrx_img,scharry_img,laplace_img,kernel1_image, kernel2_image,kernel3_image,kernel4_image]
	title_set = ['original image','average_kernel','sobel_x','sobel_y','prewitt_x','prewitt_y','scharr_x','scharr_y','laplace','own_kernel1','own_kernel2','own_kernel3','own_kernel4']
	
	plt.suptitle(f"convolution mode : {mode}")
	for i in range(len(img_set)):
		plt.subplot(3,5,i+1)
		plt.title(title_set[i])
		plt.imshow(img_set[i],cmap = 'gray')
	h,w = img.shape
	h1,w1 = laplace_img.shape
	print(f"main Image height: {h} width: {w}")
	print(f"laplace Image height: {h1} width: {w1}")
	
	plt.show()
	
if __name__ == "__main__":
	main()
