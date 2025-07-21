#Modify the copied image so that the top-left 1000×1000 pixel area becomes completely black.
import matplotlib.pyplot as plt
import cv2
import numpy as np

def main():
    img = cv2.imread('/home/alamin/DIP_basic(lec 1)/day_2/child.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img1 = img.copy()
    img1 [0:1000, 0:1000]=0 
    
    plt.subplot(1,2,1)
    plt.title("Original Picture")
    plt.imshow(img)
    
    plt.subplot(1,2,2)
    plt.title("Modified Picture")
    plt.imshow(img1)
    plt.show()

if __name__ == '__main__':
    main()

