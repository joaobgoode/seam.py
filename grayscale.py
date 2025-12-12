from PIL import Image
import numpy as np


def open_image(path):
    img = Image.open(path)
    pixels = np.array(img)
    return pixels


def pixels_to_luminance(pixels):
    R, G, B = pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2]
    L = 0.2126 * R + 0.7152 * G + 0.0722 * B
    L = np.clip(L, 0, 255).astype(np.uint8)
    return L


def luminance_to_image(L, path):
    img = np.stack([L, L, L], axis=2)
    Image.fromarray(img).save(path)


def image_to_luminance_image(input_path, output_path):
    img = Image.open(input_path)
    pixels = np.array(img)
    L = pixels_to_luminance(pixels)
    luminance_to_image(L, output_path)


if __name__ == "__main__":
    import sys

    args = sys.argv
    if len(args) != 3:
        print(f"Usage: python {args[0]} <input_path> <output_path>")
        exit(1)

    image_to_luminance_image(args[1], args[2])
