"""Pure Python Standard Library Computer Vision and Image Corruption Engine.

Implements ImageNet-C style perceptual corruptions (Pixel Noise, Contrast Shift, Occlusion)
with zero external dependencies (no PIL, OpenCV, or NumPy required).
"""

import hashlib
import os
import random
import struct
import zlib
from typing import Optional, Tuple

from src.schemas import PerturbationType


def _paeth_predictor(a: int, b: int, c: int) -> int:
    """Standard PNG Paeth filter predictor algorithm."""
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    elif pb <= pc:
        return b
    else:
        return c


def _unfilter_scanline(ftype: int, row_bytes: bytes, prev_row: Optional[bytes], bpp: int) -> bytearray:
    """Reconstructs unfiltered scanline pixels across all 5 PNG filter types."""
    recon = bytearray(len(row_bytes))
    for i in range(len(row_bytes)):
        left = recon[i - bpp] if i >= bpp else 0
        up = prev_row[i] if prev_row else 0
        up_left = prev_row[i - bpp] if prev_row and i >= bpp else 0

        if ftype == 0:  # None
            val = row_bytes[i]
        elif ftype == 1:  # Sub
            val = row_bytes[i] + left
        elif ftype == 2:  # Up
            val = row_bytes[i] + up
        elif ftype == 3:  # Average
            val = row_bytes[i] + ((left + up) >> 1)
        elif ftype == 4:  # Paeth
            val = row_bytes[i] + _paeth_predictor(left, up, up_left)
        else:
            val = row_bytes[i]

        recon[i] = val & 0xFF
    return recon


def read_png(file_path: str) -> Tuple[int, int, int, bytearray]:
    """Decodes an 8-bit PNG image into (width, height, channels, raw_pixel_bytearray)."""
    with open(file_path, "rb") as f:
        data = f.read()

    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"File {file_path} is not a valid PNG image.")

    idx = 8
    idat_chunks = []
    width, height, bit_depth, color_type = 0, 0, 0, 0

    while idx < len(data):
        length = struct.unpack(">I", data[idx:idx + 4])[0]
        chunk_type = data[idx + 4:idx + 8]
        chunk_data = data[idx + 8:idx + 8 + length]
        idx += 8 + length + 4

        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type = struct.unpack(">IIBB", chunk_data[:10])
            if bit_depth != 8:
                raise NotImplementedError(f"Only 8-bit depth PNGs are supported, got {bit_depth}")
        elif chunk_type == b"IDAT":
            idat_chunks.append(chunk_data)
        elif chunk_type == b"IEND":
            break

    # Color type mapping: 0=Grayscale (1ch), 2=RGB (3ch), 6=RGBA (4ch)
    if color_type == 2:
        channels = 3
    elif color_type == 6:
        channels = 4
    elif color_type == 0:
        channels = 1
    else:
        raise NotImplementedError(f"Unsupported PNG color type: {color_type}")

    decompressed = zlib.decompress(b"".join(idat_chunks))
    stride = 1 + width * channels
    pixels = bytearray()
    prev_row = None

    for y in range(height):
        filter_type = decompressed[y * stride]
        row_raw = decompressed[y * stride + 1:(y + 1) * stride]
        recon_row = _unfilter_scanline(filter_type, row_raw, prev_row, channels)
        pixels.extend(recon_row)
        prev_row = recon_row

    return width, height, channels, pixels


def write_png(file_path: str, width: int, height: int, channels: int, pixels: bytes) -> str:
    """Encodes raw pixel bytes into a valid compressed 8-bit PNG file."""
    raw_scanlines = bytearray()
    row_len = width * channels

    for y in range(height):
        raw_scanlines.append(0)  # Filter 0: None
        raw_scanlines.extend(pixels[y * row_len:(y + 1) * row_len])

    compressed_idat = zlib.compress(bytes(raw_scanlines), 6)

    def pack_chunk(chunk_type: bytes, payload: bytes) -> bytes:
        length = struct.pack(">I", len(payload))
        crc = struct.pack(">I", zlib.crc32(chunk_type + payload) & 0xFFFFFFFF)
        return length + chunk_type + payload + crc

    png = b"\x89PNG\r\n\x1a\n"
    colortype = 2 if channels == 3 else (6 if channels == 4 else 0)
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, colortype, 0, 0, 0)

    png += pack_chunk(b"IHDR", ihdr_data)
    png += pack_chunk(b"IDAT", compressed_idat)
    png += pack_chunk(b"IEND", b"")

    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(png)

    return file_path


def apply_pixel_noise(
    pixels: bytearray,
    width: int,
    height: int,
    channels: int,
    sigma: int = 35,
    seed: int = 42,
) -> bytearray:
    """Injects Gaussian-like pixel noise to test model visual perception under camera grain/sensor noise."""
    rng = random.Random(seed)
    corrupted = bytearray(pixels)
    rgb_channels = min(3, channels)

    for i in range(0, len(corrupted), channels):
        for ch in range(rgb_channels):
            noise = rng.gauss(0, sigma)
            val = corrupted[i + ch] + noise
            corrupted[i + ch] = max(0, min(255, int(val)))

    return corrupted


def apply_contrast_shift(
    pixels: bytearray,
    width: int,
    height: int,
    channels: int,
    factor: float = 0.35,
) -> bytearray:
    """Degrades dynamic contrast towards midpoint 128 to simulate severe low-light / washed-out environments."""
    corrupted = bytearray(pixels)
    rgb_channels = min(3, channels)

    for i in range(0, len(corrupted), channels):
        for ch in range(rgb_channels):
            original = corrupted[i + ch]
            # Compress around gray midpoint 128.0
            adjusted = 128.0 + factor * (original - 128.0)
            corrupted[i + ch] = max(0, min(255, int(adjusted)))

    return corrupted


def apply_occlusion(
    pixels: bytearray,
    width: int,
    height: int,
    channels: int,
    box_rel: Tuple[float, float, float, float] = (0.30, 0.30, 0.70, 0.70),
    fill_color: Tuple[int, int, int] = (20, 20, 20),
) -> bytearray:
    """Overlays an opaque rectangular mask patch to test VLM robustness against partial occlusions."""
    corrupted = bytearray(pixels)
    ymin_f, xmin_f, ymax_f, xmax_f = box_rel

    ymin = max(0, min(height - 1, int(height * ymin_f)))
    ymax = max(ymin + 1, min(height, int(height * ymax_f)))
    xmin = max(0, min(width - 1, int(width * xmin_f)))
    xmax = max(xmin + 1, min(width, int(width * xmax_f)))

    rgb_channels = min(3, channels)

    for y in range(ymin, ymax):
        for x in range(xmin, xmax):
            idx = (y * width + x) * channels
            for ch in range(rgb_channels):
                corrupted[idx + ch] = fill_color[ch]

    return corrupted


def generate_corrupted_image(
    image_path: str,
    perturbation_type: PerturbationType,
    output_dir: Optional[str] = None,
    seed: int = 42,
) -> str:
    """High-level generator that reads an image, applies visual corruption, and saves the new artifact.
    
    Returns:
        Absolute filesystem path to the generated corrupted image file.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Source image not found at {image_path}")

    width, height, channels, pixels = read_png(image_path)

    if perturbation_type == PerturbationType.IMAGE_PIXEL_NOISE:
        corrupted_pixels = apply_pixel_noise(pixels, width, height, channels, sigma=40, seed=seed)
    elif perturbation_type == PerturbationType.IMAGE_CONTRAST_SHIFT:
        corrupted_pixels = apply_contrast_shift(pixels, width, height, channels, factor=0.35)
    elif perturbation_type == PerturbationType.IMAGE_OCCLUSION:
        corrupted_pixels = apply_occlusion(pixels, width, height, channels, box_rel=(0.30, 0.30, 0.70, 0.70))
    else:
        raise ValueError(f"Unsupported visual perturbation type: {perturbation_type}")

    # Compute deterministic filename
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    out_name = f"{base_name}__pert_{perturbation_type.value}.png"

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(image_path), "perturbed")

    out_path = os.path.join(output_dir, out_name)
    write_png(out_path, width, height, channels, corrupted_pixels)

    return out_path
