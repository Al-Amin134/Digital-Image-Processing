import matplotlib.pyplot as plt
import cv2
import numpy as np

def main():
    img1 = cv2.imread('/home/alamin/DIP_basic(lec 1)/day4/child1.jpg')
    img2 = cv2.imread('/home/alamin/DIP_basic(lec 1)/day4/child.png')

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    
    h1, w1 = gray1.shape
    h2, w2 = gray2.shape

    
    height =int(abs (h1 - h2) / 2)
    width = int(abs((w1 - w2) /2))
    
    pded_img = np.zeros((h2,w2))
    h = 0
    for i in range(height, height+h1):
        w = 0
        for j in range(width,width+w1):
                 pded_img[ i][ j] = gray1[h][w]
                 w+=1
        h+=1

             
    plt.subplot(1, 3, 1)
    plt.title("Original (Gray1)")
    plt.imshow(gray1, cmap='gray')

    plt.subplot(1, 3, 2)
    plt.title("Original (Gray2)")
    plt.imshow(gray2, cmap='gray')
    
    plt.subplot(1, 3, 3)
    plt.title("Padded (Image: Gray1, Size: Gray2)")
    plt.imshow(pded_img, cmap='gray')
    

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()

