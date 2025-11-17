import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

# ================== SETTINGS ==================
IMG_PATH   = "/home/alamin/1.PART_IV/DIP/images/v2.png"
COLOR_BITS = 4
DCT_Q      = 20.0
DWT_Q      = 10.0
# ==============================================

try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False
    print("PyWavelets (pywt) not installed → DWT will be skipped.")


def pct_change(orig_bits, comp_bits):
    return 100.0 * (1.0 - comp_bits / max(1, orig_bits))


# ================== 1) RLE (LOSSLESS) ======================

def rle_encode(arr: np.ndarray):
    flat = arr.ravel()
    if flat.size == 0:
        return np.array([], dtype=np.uint8), np.array([], dtype=np.int32)

    values = []
    counts = []
    prev = int(flat[0])
    cnt = 1

    for v in flat[1:]:
        v = int(v)
        if v == prev:
            cnt += 1
        else:
            values.append(prev)
            counts.append(cnt)
            prev = v
            cnt = 1

    values.append(prev)
    counts.append(cnt)

    return np.array(values, dtype=np.uint8), np.array(counts, dtype=np.int32)


def rle_decode(values: np.ndarray, counts: np.ndarray, shape):
    flat = np.repeat(values, counts)
    return flat.reshape(shape).astype(np.uint8)


def rle_bits(values: np.ndarray, counts: np.ndarray) -> int:
    if len(values) == 0:
        return 0
    max_count = int(counts.max())
    count_bits = 1 if max_count <= 1 else int(np.ceil(np.log2(max_count + 1)))
    bits_per_run = 8 + count_bits
    return len(values) * bits_per_run


def compress_rle_lossless(img: np.ndarray):
    values, counts = rle_encode(img)
    bits = rle_bits(values, counts)
    recon = rle_decode(values, counts, img.shape)
    if not np.array_equal(img, recon):
        print("Warning: RLE decode != original.")
    return bits, recon


# ========== 2) Color Depth Reduction (LOSSY) ============

def compress_color_depth(img: np.ndarray, bits: int = COLOR_BITS):
    H, W = img.shape
    levels = 2 ** bits

    reduced = np.floor(img.astype(np.float32) * (levels / 256.0)).astype(np.uint8)
    reduced = np.clip(reduced, 0, levels - 1)

    comp_bits = H * W * bits

    step = 256.0 / levels
    recon = (reduced.astype(np.float32) * step + step / 2.0)
    recon = np.clip(recon, 0, 255).astype(np.uint8)

    return comp_bits, recon


# ========== 3) DCT-based Compression (LOSSY) ============

def compress_dct(img: np.ndarray, q: float = DCT_Q):
    x = img.astype(np.float32) - 128.0
    C = cv2.dct(x)
    Cq = np.round(C / q).astype(np.int32)

    flat = Cq.ravel()
    presence_bits = flat.size
    nz = flat[flat != 0]
    value_bits = len(nz) * 8
    comp_bits = presence_bits + value_bits

    Crecon = Cq.astype(np.float32) * q
    y = cv2.idct(Crecon) + 128.0
    y = np.clip(y, 0, 255).astype(np.uint8)

    return comp_bits, y


# ========== 4) DWT-based Compression (LOSSY) ============

def compress_dwt(img: np.ndarray, q: float = DWT_Q):
    if not HAS_PYWT:
        raise RuntimeError("PyWavelets (pywt) not installed")

    x = img.astype(np.float32)
    cA, (cH, cV, cD) = pywt.dwt2(x, 'haar')

    cAq = np.round(cA / q).astype(np.int32)
    cHq = np.round(cH / q).astype(np.int32)
    cVq = np.round(cV / q).astype(np.int32)
    cDq = np.round(cD / q).astype(np.int32)

    all_coeffs = np.concatenate([cAq.ravel(), cHq.ravel(), cVq.ravel(), cDq.ravel()])
    presence_bits = all_coeffs.size
    nz = all_coeffs[all_coeffs != 0]
    value_bits = len(nz) * 8
    comp_bits = presence_bits + value_bits

    cArec = cAq.astype(np.float32) * q
    cHrec = cHq.astype(np.float32) * q
    cVrec = cVq.astype(np.float32) * q
    cDrec = cDq.astype(np.float32) * q
    y = pywt.waverec2((cArec, (cHrec, cVrec, cDrec)), 'haar')
    y = y[:img.shape[0], :img.shape[1]]
    y = np.clip(y, 0, 255).astype(np.uint8)

    return comp_bits, y


# ========== Quality Metrics: MSE, PSNR, SSIM ============

def ssim_gray(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(np.float32)
    y = y.astype(np.float32)

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    kernel_size = (11, 11)
    sigma = 1.5

    mu_x = cv2.GaussianBlur(x, kernel_size, sigma)
    mu_y = cv2.GaussianBlur(y, kernel_size, sigma)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = cv2.GaussianBlur(x * x, kernel_size, sigma) - mu_x2
    sigma_y2 = cv2.GaussianBlur(y * y, kernel_size, sigma) - mu_y2
    sigma_xy = cv2.GaussianBlur(x * y, kernel_size, sigma) - mu_xy

    num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)

    ssim_map = num / (den + 1e-12)
    return float(ssim_map.mean())


def mse_psnr_ssim(ref: np.ndarray, test: np.ndarray):
    ref = ref.astype(np.float32)
    test = test.astype(np.float32)
    mse = np.mean((ref - test) ** 2)
    if mse == 0:
        psnr = float("inf")
    else:
        psnr = 10.0 * math.log10((255.0 ** 2) / mse)
    s = ssim_gray(ref, test)
    return mse, psnr, s


# ==================== MAIN ======================

def main():
    img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not read image: " + IMG_PATH)

    H, W = img.shape
    orig_bits = H * W * 8

    rle_bits, rle_img = compress_rle_lossless(img)
    cd_bits, cd_img = compress_color_depth(img, COLOR_BITS)
    dct_bits, dct_img = compress_dct(img, DCT_Q)

    if HAS_PYWT:
        dwt_bits, dwt_img = compress_dwt(img, DWT_Q)
    else:
        dwt_bits, dwt_img = None, None

    rle_pct = pct_change(orig_bits, rle_bits)
    cd_pct = pct_change(orig_bits, cd_bits)
    dct_pct = pct_change(orig_bits, dct_bits)
    dwt_pct = pct_change(orig_bits, dwt_bits) if dwt_bits is not None else 0.0

    print("\n=========== Bit Compression Summary ===========")
    print(f"Original bits                  : {orig_bits}")
    print("------------------------------------------------")
    print(f"RLE (lossless) bits            : {rle_bits}   | Change: {rle_pct:.2f}%")
    print(f"Color depth ({COLOR_BITS} bpp) : {cd_bits}   | Change: {cd_pct:.2f}%")
    print(f"DCT bits                       : {dct_bits}   | Change: {dct_pct:.2f}%")
    if dwt_bits is not None:
        print(f"DWT bits                       : {dwt_bits}   | Change: {dwt_pct:.2f}%")
    else:
        print("DWT                            : skipped (pywt not installed)")
    print("===============================================\n")

    # ---------- MSE, PSNR, SSIM ----------
    print("=========== Quality Metrics (vs Original) ===========")
    print("Method        |    MSE      |  PSNR (dB) |   SSIM")
    print("------------- | ----------  | ---------- | ------")

    mse, psnr, s = mse_psnr_ssim(img, rle_img)
    print(f"RLE           | {mse:10.2f} | {psnr:10.2f} | {s:6.4f}")

    mse, psnr, s = mse_psnr_ssim(img, cd_img)
    print(f"Color depth   | {mse:10.2f} | {psnr:10.2f} | {s:6.4f}")

    mse, psnr, s = mse_psnr_ssim(img, dct_img)
    print(f"DCT           | {mse:10.2f} | {psnr:10.2f} | {s:6.4f}")

    if dwt_bits is not None:
        mse, psnr, s = mse_psnr_ssim(img, dwt_img)
        print(f"DWT           | {mse:10.2f} | {psnr:10.2f} | {s:6.4f}")
    else:
        print("DWT           |   (skipped, pywt not installed)")
    print("====================================================\n")

    cols = 5 if dwt_bits is not None else 4
    plt.figure(figsize=(4 * cols, 4))

    plt.subplot(1, cols, 1)
    plt.title("Original\n0.00%")
    plt.imshow(img, cmap="gray")
    plt.axis("off")

    plt.subplot(1, cols, 2)
    plt.title(f"RLE (recon)\n{rle_pct:.2f}%")
    plt.imshow(rle_img, cmap="gray")
    plt.axis("off")

    plt.subplot(1, cols, 3)
    plt.title(f"Color depth {COLOR_BITS} bpp\n{cd_pct:.2f}%")
    plt.imshow(cd_img, cmap="gray")
    plt.axis("off")

    plt.subplot(1, cols, 4)
    plt.title(f"DCT (recon)\n{dct_pct:.2f}%")
    plt.imshow(dct_img, cmap="gray")
    plt.axis("off")

    if dwt_bits is not None:
        plt.subplot(1, cols, 5)
        plt.title(f"DWT (recon)\n{dwt_pct:.2f}%")
        plt.imshow(dwt_img, cmap="gray")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

