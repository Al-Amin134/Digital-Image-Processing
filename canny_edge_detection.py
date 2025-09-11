import cv2
import matplotlib.pyplot as plt
import numpy as np
def histogram(img):
	return img.ravel()	
	
def main():
	img = cv2.imread("/home/alamin/1.PART_IV/DIP/images/child1.jpg",cv2.IMREAD_GRAYSCALE)
	blurred_image = cv2.GaussianBlur(img,(3,3),0) #ekhane 3*3 kernel jar 0 standard deviation
	low_threshold = 20
	high_threshold = 150
	edges = cv2.Canny(blurred_image, low_threshold,high_threshold)
	hist_main = histogram(img)
	hist_blurred = histogram(blurred_image)
	hist_edges = histogram(edges)
	
	image_set = [img,blurred_image,edges]
	title_set = ["main_image", "After Noise Reduction","Edges","histogram_main_image","histogram_after_reducing_noise","histogram_images"]
	histogram_set = [hist_main,hist_blurred,hist_edges]
	histogram_title_set = ["histogram_main_image","histogram_after_noise_reduction","histogram_for_edges"]
	
	
	for i in range(len(image_set)):
		plt.subplot(2,3,i+1)
		plt.title(title_set[i])
		plt.imshow(image_set[i],cmap="gray")
	for i in range (len(histogram_set)):
		plt.subplot(2,3,4+i)
		plt.title(histogram_title_set[i])
		plt.hist(histogram_set[i],256,[0,256], color='black')
		
	plt.tight_layout()
	plt.show()

if __name__=="__main__":
	main()
