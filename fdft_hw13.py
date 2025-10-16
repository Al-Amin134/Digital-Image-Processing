import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def apply_dft(img):
	dft = np.fft.fft2(img)
	shifted_dft = np.fft.fftshift(dft)
	magnitude_dft = np.log(np.abs(shifted_dft) + 1)
	return magnitude_dft, shifted_dft

def apply_idft(shifted_dft):
	f_ishift = np.fft.ifftshift(shifted_dft)
	img_recons = np.fft.ifft2(f_ishift)
	img_reconst = np.abs(img_recons)
	return img_reconst

def dft_mag(arr):
	return np.log(np.abs(arr) + 1)

#====================================main function=======================================================
def main():
	base_dir = "/home/alamin/1.PART_IV/DIP/images/three_contrast_images"
	images = sorted(glob.glob(os.path.join(base_dir, "*.png")))
	
	for img_path in images:
		img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

		magnitude_dft, shifted_dft = apply_dft(img)
		height, width = img.shape

		# =================== Mask creation ===================
		mask = np.zeros((height, width), np.uint8)
		mask_high = 1 - mask

		center = (width // 2, height // 2)
		radius = 100
		color = 255
		thickness = -1

		# Low-pass mask
		cv2.circle(mask, center, radius, color, thickness)

		# High-pass mask
		mask_high = 1 - (mask.astype(np.float32) / 255.0)
		mask = mask.astype(np.float32) / 255.0

		# Band-pass mask (outer circle minus inner circle)
		mask_band = np.zeros((height, width), np.uint8)
		cv2.circle(mask_band, center, radius + 30, 255, -1)
		cv2.circle(mask_band, center, radius - 20, 0, -1)
		mask_band = mask_band.astype(np.float32) / 255.0

		# =================== Apply filters ===================
		low_pass_spec = shifted_dft * mask
		high_pass_spec = shifted_dft * mask_high
		band_pass_spec = shifted_dft * mask_band

		img_low = apply_idft(low_pass_spec)
		img_high = apply_idft(high_pass_spec)
		img_band = apply_idft(band_pass_spec)

		# DFT magnitudes
		dft_orig_mag = magnitude_dft
		dft_low_mag = dft_mag(low_pass_spec)
		dft_high_mag = dft_mag(high_pass_spec)
		dft_band_mag = dft_mag(band_pass_spec)

		# =================== Plot ===================
		plt.figure(figsize=(14, 10))
		plt.suptitle(f"Image: {os.path.basename(img_path)}", fontsize=14)

		plt.subplot(4, 3, 1); plt.title("Original");     plt.imshow(img, cmap='gray'); plt.axis('off')
		plt.subplot(4, 3, 2); plt.title("DFT");          plt.imshow(dft_orig_mag, cmap='gray'); plt.axis('off')

		plt.subplot(4, 3, 4); plt.title("Low-pass Mask"); plt.imshow(mask, cmap='gray'); plt.axis('off')
		plt.subplot(4, 3, 5); plt.title("Low-pass DFT"); plt.imshow(dft_low_mag, cmap='gray'); plt.axis('off')
		plt.subplot(4, 3, 6); plt.title("Low-pass Img"); plt.imshow(img_low, cmap='gray'); plt.axis('off')

		plt.subplot(4, 3, 7); plt.title("High-pass Mask"); plt.imshow(mask_high, cmap='gray'); plt.axis('off')
		plt.subplot(4, 3, 8); plt.title("High-pass DFT"); plt.imshow(dft_high_mag, cmap='gray'); plt.axis('off')
		plt.subplot(4, 3, 9); plt.title("High-pass Img"); plt.imshow(img_high, cmap='gray'); plt.axis('off')

		plt.subplot(4, 3, 10); plt.title("Band-pass Mask"); plt.imshow(mask_band, cmap='gray'); plt.axis('off')
		plt.subplot(4, 3, 11); plt.title("Band-pass DFT"); plt.imshow(dft_band_mag, cmap='gray'); plt.axis('off')
		plt.subplot(4, 3, 12); plt.title("Band-pass Img"); plt.imshow(img_band, cmap='gray'); plt.axis('off')

		plt.tight_layout(rect=[0, 0.95, 1, 0.95])
		plt.show()

if __name__ == "__main__":
	main()

