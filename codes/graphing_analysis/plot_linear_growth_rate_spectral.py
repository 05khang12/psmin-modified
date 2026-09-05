#!/usr/bin/env python3
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def growth_rate(kx, ky, c, kap, d, nu):
    k2 = kx**2 + ky**2
    gamma = np.full_like(k2, np.nan, dtype=float)
    nonzero = k2 > 0.0
    q = k2[nonzero]
    kyq = ky[nonzero]
    g = c**2 / q + (((q - 1.0) * c + (d - nu) * q) ** 2) / (4.0 * q**2)
    h = np.sqrt(g**2 + c**2 * kap**2 * kyq**2 / q**2)
    gamma[nonzero] = np.sqrt((h + g) / 2.0) - (
        (1.0 + q) * c + q**2 * (d + nu)
    ) / (2.0 * q)
    return gamma


C, KAPPA, D, NU = 10.0, 1.0, 1.5e-4, 1.5e-4
KMAX, NX, NY = 7.0, 1200, 600
YELLOW = np.array([255, 217, 47], dtype=np.uint8)
BLUE = np.array([55, 126, 184], dtype=np.uint8)

kx_values = np.linspace(-KMAX, KMAX, NX)
ky_values = np.linspace(KMAX, 1.0e-5, NY)
kx, ky = np.meshgrid(kx_values, ky_values, indexing="xy")
gamma = growth_rate(kx, ky, C, KAPPA, D, NU)

positive = gamma > 0.0
rgb = np.where(positive[..., None], YELLOW, BLUE)
boundary = np.zeros(gamma.shape, dtype=bool)
boundary[:, 1:] |= positive[:, 1:] != positive[:, :-1]
boundary[1:, :] |= positive[1:, :] != positive[:-1, :]
for _ in range(2):
    boundary |= np.roll(boundary, 1, 0) | np.roll(boundary, -1, 0)
    boundary |= np.roll(boundary, 1, 1) | np.roll(boundary, -1, 1)
rgb[boundary] = (0, 0, 0)

left, top, plot_w, plot_h = 125, 45, 1200, 600
canvas = Image.new("RGB", (1480, 760), "white")
region = Image.fromarray(rgb, "RGB").resize((plot_w, plot_h), Image.Resampling.NEAREST)
canvas.paste(region, (left, top))
draw = ImageDraw.Draw(canvas)


def font(size, bold=False):
    try:
        return ImageFont.truetype("arialbd.ttf" if bold else "arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


axis_font, tick_font = font(25), font(19)
draw.rectangle((left, top, left + plot_w, top + plot_h), outline="black", width=3)

for value in range(-6, 7, 2):
    x = left + (value + KMAX) / (2 * KMAX) * plot_w
    draw.line((x, top + plot_h, x, top + plot_h + 9), fill="black", width=2)
    label = str(value)
    b = draw.textbbox((0, 0), label, font=tick_font)
    draw.text((x - (b[2] - b[0]) / 2, top + plot_h + 13), label, fill="black", font=tick_font)
for value in range(0, 8):
    y = top + plot_h - value / KMAX * plot_h
    draw.line((left - 9, y, left, y), fill="black", width=2)
    label = str(value)
    b = draw.textbbox((0, 0), label, font=tick_font)
    draw.text((left - 17 - (b[2] - b[0]), y - (b[3] - b[1]) / 2), label, fill="black", font=tick_font)

draw.text((left + plot_w / 2 - 18, 700), "kx", fill="black", font=axis_font)
ky_label = Image.new("RGBA", (90, 45), (255, 255, 255, 0))
ImageDraw.Draw(ky_label).text((0, 0), "ky", fill="black", font=axis_font)
ky_label = ky_label.rotate(90, expand=True)
canvas.paste(ky_label, (31, top + plot_h // 2 - ky_label.height // 2), ky_label)

index = np.nanargmax(gamma)
iy, ix = np.unravel_index(index, gamma.shape)
kx_peak = float(kx[iy, ix])
ky_peak = float(ky[iy, ix])
mx = left + (kx_peak + KMAX) / (2.0 * KMAX) * plot_w
my = top + plot_h - ky_peak / KMAX * plot_h
draw.ellipse((mx - 8, my - 8, mx + 8, my + 8), fill="#e41a1c", outline="black", width=2)

legend_x, legend_y = 1065, 125
draw.rectangle((legend_x - 15, legend_y - 12, 1305, legend_y + 75), fill="white", outline="#777777")
items = [("#ffd92f", "γ > 0"), ("#377eb8", "γ < 0")]
for i, (color, label) in enumerate(items):
    y = legend_y + 35 * i
    draw.rectangle((legend_x, y, legend_x + 25, y + 18), fill=color, outline="black")
    draw.text((legend_x + 36, y - 3), label, fill="black", font=tick_font)

root = Path(__file__).resolve().parents[2]
output = root / "figures" / "results" / "linear_growth_rate_spectral_ky_positive.png"
output.parent.mkdir(parents=True, exist_ok=True)
canvas.save(output, dpi=(220, 220))

print(output)
print(f"grid maximum gamma={gamma[iy, ix]:.9g} at kx={kx[iy, ix]:.6g}, ky={ky[iy, ix]:.6g}")
