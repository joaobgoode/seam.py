import numpy as np
from PIL import Image
from grayscale import pixels_to_luminance
from sobel import calculate_energy, get_sobels


def open_image(path):
    img = Image.open(path)
    pixels = np.array(img)
    return pixels


def compute_vertical_seam_dp(energy):
    H, W = energy.shape
    dp = energy.copy()
    backtrack = np.zeros((H, W), dtype=np.int32)

    for y in range(1, H):
        left = np.roll(dp[y - 1], 1)
        center = dp[y - 1]
        right = np.roll(dp[y - 1], -1)

        left[0] = right[-1] = np.inf

        opts = np.vstack([left, center, right])
        best = np.argmin(opts, axis=0)
        dp[y] += opts[best, np.arange(W)]
        backtrack[y] = best - 1

    return dp, backtrack


def find_vertical_seam(dp, backtrack):
    H, W = dp.shape
    seam = np.zeros(H, dtype=np.int32)

    seam[-1] = np.argmin(dp[-1])
    for y in range(H - 2, -1, -1):
        seam[y] = seam[y + 1] + backtrack[y + 1, seam[y + 1]]

    return seam


def remove_vertical_seam(img, seam):
    H, W = img.shape[:2]
    out = np.zeros((H, W - 1, 3), dtype=img.dtype)

    for y in range(H):
        x = seam[y]
        out[y, :, :] = np.delete(img[y, :, :], x, axis=0)

    return out


def seam_carving_vertical(input_path, amount, output_path):
    pixels = open_image(input_path)
    n = 0

    for i in range(amount):
        luminance = pixels_to_luminance(pixels)

        Gx, Gy = get_sobels(luminance)
        energy = calculate_energy(Gx, Gy)

        dp, backtrack = compute_vertical_seam_dp(energy)
        seam = find_vertical_seam(dp, backtrack)

        pixels = remove_vertical_seam(pixels, seam)

        s = f"Removed seam {i + 1}/{amount}"
        print(" " * n, end="\r")
        print(s, end="\r")
        n = len(s)

    Image.fromarray(pixels).save(output_path)


if __name__ == "__main__":
    import sys

    args = sys.argv
    if len(args) != 4:
        print(f"Usage: python {args[0]} <input_path> <amount> <output_path>")
        exit(1)

    seam_carving_vertical(args[1], int(args[2]), args[3])
