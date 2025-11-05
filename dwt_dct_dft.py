import cv2
import numpy as np
import matplotlib.pyplot as plt

try:
	import pywt
	HAS_PYWT = True
except ImportError:
	HAS_PYWT = False

IMG_PATH = "/home/alamin/1.PART_IV/DIP/images/shape.jpg"

def normalization(x):
	x = x.astype(np.float64)
	mn, mx = x.min(), x.max()
	return (x - mn) / (mx - mn + 1e-12)

def magnitude(x):
	return np.log(np.abs(x) + 1.0)

def main():
	img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
	img32 = img.astype(np.float32)

	dct = cv2.dct(img32)
	dct_vis = normalization(magnitude(dct))

	F = np.fft.fft2(img32)
	Fshift = np.fft.fftshift(F)
	dft_vis = normalization(magnitude(Fshift))

	if HAS_PYWT:
		LL, (LH, HL, HH) = pywt.dwt2(img32 / 255.0, 'haar')
		LL_v = normalization(LL)
		LH_v = normalization(np.abs(LH))
		HL_v = normalization(np.abs(HL))
		HH_v = normalization(np.abs(HH))
	else:
		LL_v = LH_v = HL_v = HH_v = None

	plt.figure(figsize=(10, 8))

	plt.subplot(3, 3, 1)
	plt.title("Original")
	plt.imshow(img, cmap='gray'); plt.axis('off')

	plt.subplot(3, 3, 2)
	plt.title("DCT")
	plt.imshow(dct_vis, cmap='gray'); plt.axis('off')

	plt.subplot(3, 3, 3)
	plt.title("DFT")
	plt.imshow(dft_vis, cmap='gray'); plt.axis('off')

	if HAS_PYWT:
		plt.subplot(3, 3, 4); plt.title("DWT LL")
		plt.imshow(LL_v, cmap='gray'); plt.axis('off')

		plt.subplot(3, 3, 5); plt.title("DWT LH")
		plt.imshow(LH_v, cmap='gray'); plt.axis('off')

		plt.subplot(3, 3, 6); plt.title("DWT HL")
		plt.imshow(HL_v, cmap='gray'); plt.axis('off')

		plt.subplot(3, 3, 7); plt.title("DWT HH")
		plt.imshow(HH_v, cmap='gray'); plt.axis('off')
	else:
		plt.subplot(3, 3, 4)
		plt.title("Install PyWavelets for DWT")
		plt.text(0.5, 0.5, "pip install PyWavelets", ha='center', va='center')
		plt.axis('off')

	plt.tight_layout()
	plt.show()

if __name__ == "__main__":
	main()

'''	
" For Understanding Myself "

F = np.fft.fft2(img32) → 2D DFT (ফ্রিকোয়েন্সি স্পেকট্রাম) তৈরি করে।

Fshift = np.fft.fftshift(F) → ফ্রিকোয়েন্সি সেন্টার করে।

logmag(Fshift) → স্পেকট্রামের লগ-এম্যাগনিটিউড বের করে।

norm01() → 0 থেকে 1 এর মধ্যে স্কেল করে, যাতে imshow সহজে সঠিকভাবে ভিজুয়ালাইজ করতে পারে।
'''

