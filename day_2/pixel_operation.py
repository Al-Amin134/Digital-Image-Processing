import matplotlib.pyplot as plt
import cv2
import numpy as np

def main():
    img = cv2.imread('/home/alamin/DIP_basic(lec 1)/day_2/rose.jpg')
    img1 = cv2.imread('/home/alamin/DIP_basic(lec 1)/day_2/child.jpg')
    img = cv2.resize(img, (img.shape[1], img.shape[0]))
    img1 = cv2.resize(img1, (img.shape[1], img.shape[0]))
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    
    c = int(input("Enter the Number of Pixel"))
    img2 = img+c # if you want to add c
    img3 = img+img1 # if you want to add two image
    
    plt.subplot(2,2,1)
    plt.title("Original 1")
    plt.imshow(img)
    
    plt.subplot(2,2,2)
    plt.title("Modified 1")
    plt.imshow(img2)
    
    plt.subplot(2, 2, 3)
    plt.title("Original 2")
    plt.imshow(img1)
	
    plt.subplot(2, 2, 4)
    plt.title("Mix 1 & 2")
    plt.imshow(img3)
    plt.show()

if __name__ == '__main__':
    main()

