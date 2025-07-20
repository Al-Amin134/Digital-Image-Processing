import matplotlib.pyplot as plt
import cv2 
def main():
	img_path = '/home/alamin/DIP/rgb.png' #provide the actual image path here
	#img_4D = plt.imread(img_path)
	#img_3D = img_4D[:, :, :3]
	# uporer 2 ta line use krbo jdi open cv na thake#
	img_3D = cv2.imread(img_path) # open cv use krle direct 3D
	img_3D = cv2.cvtColor(img_3D,cv2.COLOR_BGR2RGB)
	print(img_3D.shape)
	# the negative image
	img_neg = img_3D.max()-img_3D
	
	plt.figure(figsize=(20,20))
	
	#plot main picture
	plt.subplot(2,3,1)
	plt.title("Main Picture")
	plt.imshow(img_3D)
	
	#plot Negative picture
	plt.subplot(2,3,2)
	plt.title("Negative Picture")
	plt.imshow(img_neg)
	
	#plot Red Channel
	plt.subplot(2,3,3)
	plt.title("Red Channel")
	plt.imshow(img_3D[:,:,0],cmap='Reds')
	
	#plot Green Channel
	plt.subplot(2,3,4)
	plt.title("Green Channel")
	plt.imshow(img_3D[:,:,1],cmap='Greens')
	
	#plot Bliue Channel
	plt.subplot(2,3,5)
	plt.title("Blue Channel")
	plt.imshow(img_3D[:,:,2],cmap='Blues')
	
	plt.show()
if __name__=='__main__' :
	main()

		
