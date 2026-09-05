from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


root = Path(__file__).resolve().parents[2]
plots = root / "outputs" / "C10_k1_D1p5e-4_gamma30_cross_continue_2026-06-10" / "plots"
output = root / "figures" / "results" / "gamma30_Eky_comparison.png"

paths = [
    plots / "gamma30_hwak_spectrum_ky_t2066.png",
    plots / "gamma30_hm_spectrum_ky_t2066.png",
]
images = [Image.open(path).convert("RGB") for path in paths]

gap = 24
header = 66
width = sum(image.width for image in images) + gap
height = max(image.height for image in images) + header
canvas = Image.new("RGB", (width, height), "white")

draw = ImageDraw.Draw(canvas)
try:
    font = ImageFont.truetype("arial.ttf", 32)
except OSError:
    font = ImageFont.load_default()

title = r"E(ky) at t = 30"
box = draw.textbbox((0, 0), title, font=font)
draw.text(((width - (box[2] - box[0])) / 2, 16), title, fill="black", font=font)

x = 0
for image in images:
    canvas.paste(image, (x, header))
    x += image.width + gap

output.parent.mkdir(parents=True, exist_ok=True)
canvas.save(output, dpi=(200, 200))
print(output)
