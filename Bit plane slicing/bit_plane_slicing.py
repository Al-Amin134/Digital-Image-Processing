import numpy as np
import matplotlib.pyplot as plt
import cv2
def bit_plane_slicing(img):
	bit_planes = []
	for i in range (8):
		shifted_img = img>>i
		lsb_bit = shifted_img&1
		sliced_image = lsb_bit* np.power(2,i)
		bit_planes.append(sliced_image)
	return bit_planes
def display(img_set,title_set,color_set ):
	row = 2
	k = 1
	col = len(img_set)+1//row
	for i in range(1, row+1):
		for j in range (1,col+1):
			if(k>len(img_set)):
				break
			img  = img_set[k-1]
			plt.subplot(row,col,k)
			plt.title(title_set[k-1])
			if(len(img.shape)==3):
					plt.imshow(img)
			else:
					plt.imshow(img,cmap=color_set[i])
			k+=1
	plt.show()
def main():
	img_path = '/home/alamin/1.PART_IV/DIP/DIP_basic(lec 1)/day5/child1.jpg'
	img = cv2.imread(img_path,0)
	bit_planes = bit_plane_slicing(img)
	bit_planes12 = bit_planes[ 0] + bit_planes[1]
	bit_planes34 = bit_planes[ 2]+bit_planes[ 3]
	bit_planes56 = bit_planes[4]+bit_planes[5]
	bit_planes78 = bit_planes[6]+bit_planes[7]

	reconstructed_image = bit_planes[0]+bit_planes[0]+bit_planes[2]+bit_planes[3]+bit_planes[4]+bit_planes[5]+bit_planes[6]+bit_planes[7]
	
	img_set =[img,bit_planes12, bit_planes34,bit_planes56,bit_planes78, reconstructed_image]
	title_set = ['original_image','plane1','plane2','plane3','palne4','recontructed_image']
	color = ['gray','gray','gray','gray','gray','gray']
	
	display(img_set,title_set,color)
	
if __name__ =='__main__':
	main()
