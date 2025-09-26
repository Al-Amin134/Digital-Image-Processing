import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

#display function
def display(img_set, title_set):
	for i in range (len(img_set)):
		plt.subplot(4,5,i+1)
		plt.imshow(img_set[i], cmap='gray')
		plt.title(title_set[i])
		plt.axis('off')
		
	plt.tight_layout()
	plt.show()
	

#kernel functions	
def get_kernel(name):
	if name=='kernel_rectangle':
		return np.ones((5,5), np.uint8)
	elif name=='kernel_ellips':
		return  np.array([
				[0, 0, 1, 0, 0],
				[0, 1, 1, 1, 0],
				[1, 1, 1, 1, 1],
				[0, 1, 1, 1, 0],
				[0, 0, 1, 0, 0]
				], dtype=np.uint8)
	elif name=='kernel_cross':			
		return np.array([
				[0, 0, 1, 0, 0],
				[0, 0, 1, 0, 0],
				[1, 1, 1, 1, 1],
				[0, 0, 1, 0, 0],
				[0, 0, 1, 0, 0]
				], dtype=np.uint8)
				
	elif name=='kernel_diamond':
		return np.array([
				[0,0,1,0,0],
				[0,1,1,1,0],
				[1,1,1,1,1],
				[0,1,1,1,0],
				[0,0,1,0,0],
				], dtype = np.uint8)
	else :
		raise ValueError("Error kernel")

#Manual Functions
def manual_erosion(img,kernel):
	k_h, k_w = kernel.shape
	pad_h, pad_w = k_h//2, k_w//2
	padded_img = np.pad(img,((pad_h, pad_h),(pad_w, pad_w)), mode = 'constant' , constant_values = 0)
	eroded_img = np.zeros_like(img)
	
	for i in range(eroded_img.shape[0]):
		for j in range(eroded_img.shape[1]):
			region = padded_img[i:i + k_h, j:j + k_w]
			if np.all(region[kernel==1]==1):
				eroded_img[i,j] = 1
			else :
				eroded_img[i,j] = 0
	return eroded_img

def manual_dilation(img, kernel):
	k_h , k_w = kernel.shape
	pad_h, pad_w = k_h//2, k_w//2
	padded_img = np.pad(img,((pad_h,pad_h),(pad_w,pad_w)), mode = 'constant', constant_values = 0)
	dilated_img = np.zeros_like(img)
	
	for i in range(dilated_img.shape[0]):
		for j in range(dilated_img.shape[1]):
			region = padded_img[i:i+k_h, j:j+k_w]
			if np.any(region[kernel==1]==1):
				dilated_img[i,j] = 1
			else:
				dilated_img[i,j] = 0
	return dilated_img
	
def manual_opening(img, kernel):
	eroded = manual_erosion(img, kernel)
	opened = manual_dilation(eroded, kernel)
	return opened

def manual_closing(img, kernel):
	dilated = manual_dilation(img, kernel)
	closed = manual_erosion(dilated, kernel)
	return closed

def manual_top_hat(img, kernel):
	opened = manual_opening(img, kernel)
	top_hat = img - opened
	return top_hat

def manual_black_hat(img, kernel):
	closed = manual_closing(img, kernel)
	black_hat = closed - img
	return black_hat


#Main Function					
def main():				
	img = cv2.imread("/home/alamin/1.PART_IV/DIP/images/morphological_image_neg.png",0)
	img = np.where(img>127,1,0).astype(np.uint8)
	
	operations = ['erosion','dilation','opening','closing','top_hat','black_hat']
	kernel = {
	'kernel_rectangle': get_kernel('kernel_rectangle'),
	'kernel_ellips':get_kernel('kernel_ellips'),
	'kernel_cross':get_kernel('kernel_cross'),
	'kernel_diamond':get_kernel('kernel_diamond')
	}
	
	for op in operations:
		img_set = [img]
		title_set = ['Original Image']
		for k_name, k in kernel.items():
			if op=='erosion':
				img1 =  (cv2.erode(img, k,iterations=3))
			elif op=='dilation':
				img1 =  (cv2.dilate(img, k,iterations=3))
			elif op=='opening':
				img1 = (cv2.morphologyEx(img, cv2.MORPH_OPEN, k,iterations=3))*255.0
			elif op=='closing':
				img1 = (cv2.morphologyEx(img, cv2.MORPH_CLOSE, k,iterations=3))*255.0
			elif op=='top_hat':
				img1 = (cv2.morphologyEx(img, cv2.MORPH_TOPHAT, k,iterations=3))*255.0
			elif op=='black_hat':
				img1 = (cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, k,iterations=3))*255.0
			img_set.append(img1)
			title_set.append(f'{op} {k_name}(built in )')
		for k_name, k in kernel.items():
			if op=='erosion':
				img1 =  manual_erosion(img, k)
			if op=='dilation':
				img1 = manual_dilation(img, k)
			if op=='opening':
				img1 = manual_opening(img, k)
			if op=='closing':
				img1 = manual_closing(img, k)
			if op=='top_hat':
				img1 = manual_top_hat(img, k)
			if op=='black_hat':
				img1 = manual_black_hat(img, k)
			img_set.append(img1)
			title_set.append(f' {op} {k_name}(user define)')
			
			
		display(img_set, title_set)
	
	
if __name__ == "__main__":
	main()

