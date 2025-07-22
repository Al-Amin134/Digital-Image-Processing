import matplotlib.pyplot as plt
import cv2
import numpy as np

def main():
    img = cv2.imread('/home/alamin/DIP_basic(lec 1)/child.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img= img[:, :, 1]
    prepare_histogram(img)
    plt.show()

def prepare_histogram(img):
    h, w = img.shape
    pixel_array = np.zeros(256, dtype = int)  

    for i in range(h):
        for j in range(w):
            pixel_value = img[i, j]
            pixel_array[pixel_value] += 1

    print(pixel_array)
    n = np.arange(256)
    plt.scatter(n,pixel_array, color='r')
    plt.show()
    
if __name__ == '__main__':
    main()

