#"Apply the functions listed below to the provided image and display the resulting effects for each. Ensure that the original image and the output from each function are clearly shown for comparison."
# s = c*r^(Γ)
# s = c log2(1+r)

import matplotlib.pyplot as plt
import cv2
import numpy as np

gamma = [0.3, 0.5, 0.7,1,2,3]
def applying_gamma(img, gamma):
    c = 1
    r = img / 255.0 
    s = c * np.power(r, gamma)  
    return s
def applying_log(img):
    c = 1
    r = img/255.0
    s = c*np.log2(1+r)
    return s;
    
def main():
    img1 = cv2.imread('/home/alamin/DIP_basic(lec 1)/day4/child1.jpg')
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    j = 1;
    for i in range(0,len(gamma)):
    	gamma1 = applying_gamma(img1, gamma[i])
    	plt.subplot(3, 3, j)
    	plt.imshow(gamma1, cmap='gray')
    	plt.title(f"Gamma: {gamma[i]}");
    	j+=1;
    
    plt.subplot(3,3,7)
    logged_image = applying_log(img1)
    plt.imshow(logged_image, cmap= 'grey')
    plt.title("s = c log 2(1+r)")
    plt.show()

if __name__ == '__main__':
    main()

