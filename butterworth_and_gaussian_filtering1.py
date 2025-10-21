import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ------------- Frequency helpers -------------
def dft(img):
	return np.fft.fftshift(np.fft.fft2(img))

def idft(fshift):
	return np.abs(np.fft.ifft2(np.fft.ifftshift(fshift)))

def dist_grid(shape):
	rows, cols = shape
	crow, ccol = rows//2, cols//2
	v = np.arange(rows) - crow
	u = np.arange(cols) - ccol
	U, V = np.meshgrid(u, v)
	return np.sqrt(U**2 + V**2)

def norm8(arr):
	a = arr - np.min(arr)
	if np.max(a) > 0:
		a = a / np.max(a)
	return (a * 255).astype('uint8')

# ------------- Ideal Filters -------------
def ideal_low(shape, cutoff):
	D = dist_grid(shape)
	return np.where(D <= cutoff, 1, 0)

def ideal_high(shape, cutoff):
	return 1 - ideal_low(shape, cutoff)

def ideal_band(shape, low_cutoff, high_cutoff):
	low_mask = ideal_low(shape, high_cutoff)
	high_mask = ideal_high(shape, low_cutoff)
	return low_mask * high_mask

# ------------- Butterworth Filters -------------
def butter_low(shape, cutoff, order=2):
	D = dist_grid(shape)
	return 1.0 / (1.0 + (D / (cutoff + 1e-9))**(2 * order))

def butter_high(shape, cutoff, order=2):
	return 1.0 - butter_low(shape, cutoff, order)

def butter_band(shape, low_cutoff, high_cutoff, order=2):
	return butter_low(shape, high_cutoff, order) * butter_high(shape, low_cutoff, order)

# ------------- Gaussian Filters -------------
def gauss_low(shape, sigma):
	D = dist_grid(shape)
	return np.exp(-(D**2) / (2 * (sigma**2 + 1e-9)))

def gauss_high(shape, sigma):
	return 1.0 - gauss_low(shape, sigma)

def gauss_band(shape, low_sigma, high_sigma):
	return gauss_low(shape, high_sigma) * gauss_high(shape, low_sigma)

# ------------- Parameters (edit if needed) -------------
bw_cutoff_simple = 60 
bw_orders = [1, 2, 3, 4 ,5, 6]  
bw_band = (40, 90)  

g_sigma_simple = 30 
g_band = (20, 70) 
ideal_cutoff = 60  # Ideal cutoff for low-pass, high-pass

# ------------- Image Preparation -------------


# ------------- Processing -------------
def main():
	images = [
	"/home/alamin/1.PART_IV/DIP/images/three_contrast_images/image_low_contrast.png",
	"/home/alamin/1.PART_IV/DIP/images/three_contrast_images/image_high_contrast.png",
	"/home/alamin/1.PART_IV/DIP/images/three_contrast_images/image_normal_contrast.png"
		]

	for img_path in images:
		if not os.path.isfile(img_path):
			print("File not found:", img_path)
			continue

		img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
		if img is None:
			print("Failed to read:", img_path)
			continue

		h, w = img.shape
		shape = (h, w)
		fname = os.path.splitext(os.path.basename(img_path))[0]
		fshift = dft(img)
		dft_mag = np.log(np.abs(fshift) + 1)

		# Build masks
		masks = {}
		masks['butter_low'] = butter_low(shape, bw_cutoff_simple, bw_orders[1])
		masks['butter_high'] = butter_high(shape, bw_cutoff_simple, bw_orders[1])
		masks['gauss_low'] = gauss_low(shape, g_sigma_simple)
		masks['gauss_high'] = gauss_high(shape, g_sigma_simple)

		# Butterworth Band and Gaussian Band
		bw_low_for_band = butter_low(shape, bw_band[1], bw_orders[1])
		bw_high_for_band = butter_high(shape, bw_band[0], bw_orders[1])
		masks['butter_band'] = butter_band(shape, bw_band[0], bw_band[1], bw_orders[1])

		g_low_for_band = gauss_low(shape, g_band[1])
		g_high_for_band = gauss_high(shape, g_band[0])
		masks['gauss_band'] = gauss_band(shape, g_band[0], g_band[1])

		# For Ideal Filtering
		masks['ideal_low'] = ideal_low(shape, ideal_cutoff)
		masks['ideal_high'] = ideal_high(shape, ideal_cutoff)
		masks['ideal_band'] = ideal_band(shape, bw_band[0], bw_band[1])

		reconstructions = {}

		# Apply Butterworth with different values of n
		for n in bw_orders:
			masks[f'butter_low_n{n}'] = butter_low(shape, bw_cutoff_simple, n)
			masks[f'butter_high_n{n}'] = butter_high(shape, bw_cutoff_simple, n)
			masks[f'butter_band_n{n}'] = butter_band(shape, bw_cutoff_simple, n)

		for key, mask in masks.items():
			spec = fshift * mask
			recon = idft(spec)
			recon_u8 = norm8(recon)
			reconstructions[key] = recon_u8
		
		# Compact plot: 5x3 matrix
		plt.figure(figsize=(15, 10))
		plt.suptitle(fname)

		# First row: main image, DFT mag, low-pass, high-pass, band-pass
		plt.subplot(5, 3, 1); plt.title("Original Image"); plt.imshow(img, cmap='gray'); plt.axis('off')
		plt.subplot(5, 3, 2); plt.title("DFT mag"); plt.imshow(dft_mag, cmap='gray'); plt.axis('off')
		plt.subplot(5, 3, 3); plt.title("DFT mag (Low-pass)"); plt.imshow(np.log(np.abs(fshift * masks['butter_low']) + 1), cmap='gray'); plt.axis('off')
		plt.subplot(5, 3, 4); plt.title("DFT mag (High-pass)"); plt.imshow(np.log(np.abs(fshift * masks['butter_high']) + 1), cmap='gray'); plt.axis('off')
		plt.subplot(5, 3, 5); plt.title("DFT mag (Band-pass)"); plt.imshow(np.log(np.abs(fshift * masks['butter_band']) + 1), cmap='gray'); plt.axis('off')

		# Second row: Reconstructed Image
		plt.subplot(5, 3, 6); plt.title("Reconstructed Image"); plt.imshow(reconstructions['butter_low'], cmap='gray'); plt.axis('off')

		# Third row: Butterworth Filters (n=1, n=2, n=3)
		plt.subplot(5, 3, 7); plt.title("Butter Low-pass"); plt.imshow(reconstructions['butter_low'], cmap='gray'); plt.axis('off')
		plt.subplot(5, 3, 8); plt.title("Butter High-pass"); plt.imshow(reconstructions['butter_high'], cmap='gray'); plt.axis('off')
		plt.subplot(5, 3, 9); plt.title("Butter Band-pass"); plt.imshow(reconstructions['butter_band'], cmap='gray'); plt.axis('off')

		# Fourth row: Gaussian Filters (low, high, band)
		plt.subplot(5, 3, 10); plt.title("Gaussian Low-pass"); plt.imshow(reconstructions['gauss_low'], cmap='gray'); plt.axis('off')
		plt.subplot(5, 3, 11); plt.title("Gaussian High-pass"); plt.imshow(reconstructions['gauss_high'], cmap='gray'); plt.axis('off')
		plt.subplot(5, 3, 12); plt.title("Gaussian Band-pass"); plt.imshow(reconstructions['gauss_band'], cmap='gray'); plt.axis('off')

		# Fifth row: Ideal Filters (low, high, band)
		plt.subplot(5, 3, 13); plt.title("Ideal Low-pass"); plt.imshow(reconstructions['ideal_low'], cmap='gray'); plt.axis('off')
		plt.subplot(5, 3, 14); plt.title("Ideal High-pass"); plt.imshow(reconstructions['ideal_high'], cmap='gray'); plt.axis('off')
		plt.subplot(5, 3, 15); plt.title("Ideal Band-pass"); plt.imshow(reconstructions['ideal_band'], cmap='gray'); plt.axis('off')

		plt.show()

		# Plot for Butterworth n=1, n=3, n=5
	plt.suptitle("applying different values of 'n' in Butterworth filtering")
	
	plt.subplot(3, 3, 1); plt.title("Butterworth_low_pass n=1"); plt.imshow(reconstructions['butter_low_n1'], cmap='gray'); plt.axis('off')  # Smoother
	plt.subplot(3, 3, 2); plt.title("Butterworth_low_pass n=3"); plt.imshow(reconstructions['butter_low_n3'], cmap='gray'); plt.axis('off')  # Sharper
	plt.subplot(3, 3, 3); plt.title("Butterworth_low_pass n=5"); plt.imshow(reconstructions['butter_low_n5'], cmap='gray'); plt.axis('off')  # Sharpest
	
	plt.subplot(3, 3, 4); plt.title("Butterworth_band_pass n=1"); plt.imshow(reconstructions['butter_band_n1'], cmap='gray'); plt.axis('off')  # Smoother
	plt.subplot(3, 3, 5); plt.title("Butterworth_band_pass n=3"); plt.imshow(reconstructions['butter_band_n3'], cmap='gray'); plt.axis('off')  # Sharper
	plt.subplot(3, 3, 6); plt.title("Butterworth_band_pass n=5"); plt.imshow(reconstructions['butter_band_n5'], cmap='gray'); plt.axis('off')  # Sharpest
	
	plt.subplot(3, 3, 7); plt.title("Butterworth_high_pass n=1"); plt.imshow(reconstructions['butter_high_n1'], cmap='gray'); plt.axis('off')  # Smoother
	plt.subplot(3, 3, 8); plt.title("Butterworth_high_pass n=3"); plt.imshow(reconstructions['butter_high_n3'], cmap='gray'); plt.axis('off')  # Sharper
	plt.subplot(3, 3, 9); plt.title("Butterworth_high_pass n=5"); plt.imshow(reconstructions['butter_high_n5'], cmap='gray'); plt.axis('off')  # Sharpest
	
	plt.show()

	print("Finished processing images.")

# Run the main function
if __name__ == '__main__':
	main()
