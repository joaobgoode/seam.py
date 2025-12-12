import numpy as np
from PIL import Image
from grayscale import pixels_to_luminance
from scipy.signal import convolve2d


def open_image(path):
    img = Image.open(path)
    pixels = np.array(img)
    return pixels


def convolve2d_pair(img, kx, ky):
    Gx = convolve2d(img, kx, mode="same", boundary="symm")
    Gy = convolve2d(img, ky, mode="same", boundary="symm")
    return Gx, Gy


def calculate_energy(Gx, Gy):
    return np.sqrt(Gx**2 + Gy**2)


def get_sobels(luminance):
    Kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=float)

    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=float)

    Gx, Gy = convolve2d_pair(luminance, Kx, Ky)
    return Gx, Gy


def luminance_to_sobel(luminance):
    Gx, Gy = get_sobels(luminance)
    energy = calculate_energy(Gx, Gy)
    sobel = np.clip(energy, 0, 255).astype(np.uint8)
    return sobel


def sobel_to_image(sobel, path):
    sobel_img = Image.fromarray(sobel)
    sobel_img.save(path)


def image_to_sobel_image(input_path, output_path):
    pixels = open_image(input_path)
    luminance = pixels_to_luminance(pixels)
    sobel = luminance_to_sobel(luminance)
    sobel_to_image(sobel, output_path)


if __name__ == "__main__":
    import sys

    args = sys.argv
    if len(args) != 3:
        print(f"Usage: python {args[0]} <input_image> <output_image>")
        exit(1)

    image_to_sobel_image(args[1], args[2])
