import matplotlib.pyplot as plt

def main():
    # Load an image
    img_path = '/home/alamin/CSE4161_DIP/Images/MyScreen.png'
    img_4D = plt.imread(img_path)
    print(img_4D.shape)
    
    # Convert 4D image into an 3D image
    img_3D = img_4D[:, :, :3]
    print(img_3D.shape)
    
    # Print some values of images
    print(img_3D[:5,:5,:2])
    print(img_3D.max(), img_3D.min())
    
    # Prepare negative of loaded image
    negative_img = img_3D.max() - img_3D
    
    # Display loaded 3D image
    plt.figure(figsize = (20,20))
    plt.subplot(2, 3, 1)
    plt.imshow(img_3D)
    plt.subplot(2, 3, 2)
    plt.imshow(img_3D[:, :, 0], cmap = 'Reds')
    plt.subplot(2, 3, 3)
    plt.imshow(img_3D[:, :, 1],cmap = 'Greens')
    plt.subplot(2, 3, 4)
    plt.imshow(img_3D[:, :, 2],cmap = 'Blues')
    plt.subplot(2, 3, 5)
    plt.imshow(negative_img)
    plt.show()
    plt.close()
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
