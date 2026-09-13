"""Generates minimal valid PNG images for the sample benchmark using only standard library."""

import os
import struct
import zlib


def create_png(width: int, height: int, rgb_fill=(240, 240, 245), rects=None) -> bytes:
    """Creates a basic PNG with background fill and colored rectangular regions."""
    # Build raw image pixel buffer (RGB)
    pixels = bytearray()
    for y in range(height):
        pixels.append(0)  # Filter byte: None
        for x in range(width):
            color = rgb_fill
            if rects:
                for rx, ry, rw, rh, r_color in rects:
                    if rx <= x < rx + rw and ry <= y < ry + rh:
                        color = r_color
                        break
            pixels.extend(color)

    compressed_data = zlib.compress(bytes(pixels), 9)

    def make_chunk(chunk_type: bytes, data: bytes) -> bytes:
        length = struct.pack(">I", len(data))
        crc = struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
        return length + chunk_type + data + crc

    # PNG Signature
    png = b"\x89PNG\r\n\x1a\n"

    # IHDR: width (4), height (4), depth (1), color_type (1: 2=RGB), comp (1), filter (1), interlace (1)
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png += make_chunk(b"IHDR", ihdr_data)

    # IDAT: Image data
    png += make_chunk(b"IDAT", compressed_data)

    # IEND: End of file
    png += make_chunk(b"IEND", b"")

    return png


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "images"))
    os.makedirs(base_dir, exist_ok=True)

    # 1. Bar Chart: Background white with bars
    chart_png = create_png(
        width=200, height=120, rgb_fill=(255, 255, 255),
        rects=[
            (30, 70, 25, 40, (50, 120, 220)),   # Q1 bar
            (70, 50, 25, 60, (50, 120, 220)),   # Q2 bar
            (110, 30, 25, 80, (230, 80, 50)),   # Q3 bar (tallest: 320M)
            (150, 45, 25, 65, (50, 120, 220)),  # Q4 bar
        ]
    )
    with open(os.path.join(base_dir, "q3_revenue_chart.png"), "wb") as f:
        f.write(chart_png)

    # 2. Product growth chart
    growth_png = create_png(
        width=180, height=100, rgb_fill=(250, 250, 250),
        rects=[
            (20, 40, 40, 50, (60, 160, 60)),
            (80, 20, 40, 70, (50, 100, 200)),  # High growth
            (140, 60, 30, 30, (200, 150, 40)),
        ]
    )
    with open(os.path.join(base_dir, "product_growth_chart.png"), "wb") as f:
        f.write(growth_png)

    # 3. Invoice document simulation
    invoice_png = create_png(
        width=150, height=180, rgb_fill=(255, 255, 255),
        rects=[
            (10, 10, 130, 15, (220, 220, 220)),
            (10, 40, 130, 4, (180, 180, 180)),
            (10, 55, 130, 4, (180, 180, 180)),
            (10, 70, 130, 4, (180, 180, 180)),
            (80, 150, 60, 15, (230, 240, 255)), # Total box
        ]
    )
    with open(os.path.join(base_dir, "invoice_sample_88.png"), "wb") as f:
        f.write(invoice_png)

    # 4. Street scene (No truck - for hallucination test)
    street_png = create_png(
        width=160, height=120, rgb_fill=(180, 200, 220), # Sky
        rects=[
            (0, 70, 160, 50, (90, 90, 95)),    # Road
            (50, 85, 60, 8, (255, 255, 255)),  # Crosswalk lines
            (20, 30, 40, 40, (140, 140, 150)), # Building
        ]
    )
    with open(os.path.join(base_dir, "street_scene_01.png"), "wb") as f:
        f.write(street_png)

    # 5. Unsigned contract
    contract_png = create_png(
        width=150, height=180, rgb_fill=(255, 255, 255),
        rects=[
            (15, 20, 120, 6, (150, 150, 150)),
            (15, 35, 120, 4, (200, 200, 200)),
            (15, 50, 120, 4, (200, 200, 200)),
            (15, 65, 120, 4, (200, 200, 200)),
            (80, 160, 50, 2, (100, 100, 100)), # Signature line (empty)
        ]
    )
    with open(os.path.join(base_dir, "unsigned_contract.png"), "wb") as f:
        f.write(contract_png)

    # 6. Spatial shapes grid
    shapes_png = create_png(
        width=160, height=100, rgb_fill=(245, 245, 245),
        rects=[
            (30, 35, 30, 30, (30, 90, 210)),   # Blue container (left)
            (80, 30, 25, 40, (230, 200, 40)),  # Yellow cylinder (center)
            (120, 35, 25, 30, (210, 40, 40)),  # Red cube (right)
        ]
    )
    with open(os.path.join(base_dir, "spatial_shapes_grid.png"), "wb") as f:
        f.write(shapes_png)

    # 7. Aerial roof array (18 panels: 3 rows of 6)
    panels = []
    for r in range(3):
        for c in range(6):
            panels.append((25 + c * 18, 25 + r * 18, 14, 14, (30, 45, 90)))
    roof_png = create_png(width=160, height=110, rgb_fill=(180, 140, 120), rects=panels)
    with open(os.path.join(base_dir, "aerial_roof_array.png"), "wb") as f:
        f.write(roof_png)

    print(f"✅ Generated 7 sample benchmark PNG images in {base_dir}")


if __name__ == "__main__":
    main()
