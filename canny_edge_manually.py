import cv2
import numpy as np
import matplotlib.pyplot as plt

def simple_canny(gray, use_gaussian=True, ksize=5, sigma=1.0, low=50, high=150):
	# 1) optional Gaussian blur
	if use_gaussian:
		if ksize % 2 == 0:
			raise ValueError("Gaussian kernel size must be odd.")
		img = cv2.GaussianBlur(gray, (ksize, ksize), sigma)
	else:
		img = gray.copy()

	# 2) Sobel gradient (Gx, Gy)
	Kx = np.array([[-1, 0, 1],
	               [-2, 0, 2],
	               [-1, 0, 1]], dtype=np.float32)
	Ky = np.array([[ 1,  2,  1],
	               [ 0,  0,  0],
	               [-1, -2, -1]], dtype=np.float32)

	Gx = cv2.filter2D(img, cv2.CV_32F, Kx)
	Gy = cv2.filter2D(img, cv2.CV_32F, Ky)

	# 3) gradient magnitude + angle
	mag = np.sqrt(Gx * Gx + Gy * Gy)
	mag = mag / (mag.max() + 1e-8) * 255
	angle = np.arctan2(Gy, Gx) * 180.0 / np.pi
	angle[angle < 0] += 180

	M, N = mag.shape

	# 4) simple non-maximum suppression (4 main directions)
	nms = np.zeros((M, N), dtype=np.float32)

	for i in range(1, M - 1):
		for j in range(1, N - 1):
			q = 0
			r = 0

			if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
				q = mag[i, j + 1]
				r = mag[i, j - 1]
			elif 22.5 <= angle[i, j] < 67.5:
				q = mag[i + 1, j - 1]
				r = mag[i - 1, j + 1]
			elif 67.5 <= angle[i, j] < 112.5:
				q = mag[i + 1, j]
				r = mag[i - 1, j]
			elif 112.5 <= angle[i, j] < 157.5:
				q = mag[i - 1, j - 1]
				r = mag[i + 1, j + 1]

			if mag[i, j] >= q and mag[i, j] >= r:
				nms[i, j] = mag[i, j]
			else:
				nms[i, j] = 0

	# 5) double threshold + very simple hysteresis
	strong = 255
	weak = 100

	result = np.zeros((M, N), dtype=np.uint8)

	strong_i, strong_j = np.where(nms >= high)
	weak_i, weak_j = np.where((nms >= low) & (nms < high))

	result[strong_i, strong_j] = strong
	result[weak_i, weak_j] = weak

	for i in range(1, M - 1):
		for j in range(1, N - 1):
			if result[i, j] == weak:
				if (result[i-1:i+2, j-1:j+2] == strong).any():
					result[i, j] = strong
				else:
					result[i, j] = 0

	return result

def main():
# 1) image load (grayscale)
	img_gray = cv2.imread("/home/alamin/1.PART_IV/DIP/images/canny.jpg", cv2.IMREAD_GRAYSCALE)
	low = 50
	high = 150

# -------------------------------------------------
#  Effect of different Gaussian kernel sizes
# -------------------------------------------------
	kernel_sizes = [1, 3, 5,7]
	edges_with_kernels = []
	edge_counts = []

	for k in kernel_sizes:
		edges = simple_canny(
			img_gray,
			use_gaussian=True,
			ksize=k,
			sigma=1.0,
			low=low,
			high=high
		)
		edges_with_kernels.append(edges)
		edge_counts.append(np.count_nonzero(edges))

	print("=== Edge Pixel Counts for Different Gaussian Kernel Sizes (Manual Canny) ===")
	for k, c in zip(kernel_sizes, edge_counts):
		print(f"Kernel size {k}x{k} : {c} edge pixels")

# show original + different kernel results
	plt.figure(figsize=(12, 8))

	plt.subplot(2, 3, 1)
	plt.imshow(img_gray, cmap="gray")
	plt.title("Original Grayscale")
	plt.axis("off")

	for i, (k, edges) in enumerate(zip(kernel_sizes, edges_with_kernels), start=2):
		plt.subplot(2, 3, i)
		plt.imshow(edges, cmap="gray")
		plt.title(f"Gaussian k={k}")
		plt.axis("off")

	plt.tight_layout()
	plt.show()

# -------------------------------------------------
#  Effect of using vs not using Gaussian (ksize=5)
# -------------------------------------------------
	edges_no_gauss = simple_canny(
		img_gray,
		use_gaussian=False,
		ksize=5,
		sigma=1.0,
		low=low,
		high=high
	)

	edges_gauss_5 = simple_canny(
		img_gray,
		use_gaussian=True,
		ksize=5,
		sigma=1.0,
		low=low,
		high=high
	)

	count_no = np.count_nonzero(edges_no_gauss)
	count_g5 = np.count_nonzero(edges_gauss_5)

	print("\n=== With vs Without Gaussian Blur (Manual Canny, k=5) ===")
	print(f"Without Gaussian : {count_no} edge pixels")
	print(f"With Gaussian k=5: {count_g5} edge pixels")

# show comparison
	plt.figure(figsize=(12, 4))

	plt.subplot(1, 3, 1)
	plt.imshow(img_gray, cmap="gray")
	plt.title("Original")
	plt.axis("off")

	plt.subplot(1, 3, 2)
	plt.imshow(edges_no_gauss, cmap="gray")
	plt.title(f"No Gaussian\nEdges: {count_no}")
	plt.axis("off")

	plt.subplot(1, 3, 3)
	plt.imshow(edges_gauss_5, cmap="gray")
	plt.title(f"Gaussian k=5\nEdges: {count_g5}")
	plt.axis("off")

	plt.tight_layout()
	plt.show()

if __name__ == "__main__":
	main()

