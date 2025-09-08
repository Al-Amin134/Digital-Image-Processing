import cv2
import numpy as np
import matplotlib.pyplot as plt

def hist_equalization_from_scratch(img):
	flat = img.ravel() # img k 2D theke 1D te convert krlam
	hist = np.bincount(flat,minlength = 256)
	cdf = np.cumsum(hist)
	
	cdf_min = cdf[cdf>0][0] # jei value gula 0 er theke boro segular first value ta
	N = flat.size
	table = ((cdf-cdf_min)/(N-cdf_min)*255).clip(0,255).astype(np.uint8)
	return table[flat].reshape(img.shape)
	
def main():
	
	img = cv2.imread("/home/alamin/1.PART_IV/DIP/images/child.png",cv2.IMREAD_GRAYSCALE)
	custom = hist_equalization_from_scratch(img)
	built_in = cv2.equalizeHist(img)
	
	image_set = [img,custom,built_in]
	title_set = ["Original","Equalized(Custom)","Equalized(Built in Function)"]
	
	for i in range(3):
		plt.subplot(3,2,2*i+1)
		plt.title(title_set[i])
		plt.imshow(image_set[i],cmap="gray")
		plt.axis("off")
		
		plt.subplot(3,2,2*i+2)
		plt.hist(image_set[i].ravel(),bins=256,range = (0,255),color="black")
		plt.title(title_set[i] + "Histogram")
		plt.xlabel("Intensity")
		plt.ylabel("Count")
	plt.suptitle("Histogram Equalization")
	plt.tight_layout()
	plt.show()

if __name__ == "__main__":
	main()
