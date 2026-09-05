from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    ROOT
    / "outputs"
    / "C10_k1_D1p5e-4_gamma30_cross_continue_2026-06-10"
    / "plots"
)
OUTPUT = ROOT / "figures" / "results" / "gamma30_energy_comparison.png"

PANELS = [
    ("(a)", SOURCE / "gamma30_hwak_total_vs_zonal.png"),
    ("(b)", SOURCE / "gamma30_hm_total_vs_zonal.png"),
    ("(c)", SOURCE / "gamma30_zonal_energy_compare.png"),
]

panel_width = 1280
panel_height = 810
gap = 28
label_height = 54
margin = 24

canvas_width = 2 * panel_width + gap + 2 * margin
canvas_height = 2 * (panel_height + label_height) + gap + 2 * margin
canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
draw = ImageDraw.Draw(canvas)

try:
    font = ImageFont.truetype("DejaVuSans.ttf", 38)
except OSError:
    font = ImageFont.load_default()

positions = [
    (margin, margin),
    (margin + panel_width + gap, margin),
    ((canvas_width - panel_width) // 2, margin + panel_height + label_height + gap),
]

for (label, path), (x, y) in zip(PANELS, positions):
    with Image.open(path) as image:
        panel = image.convert("RGB").resize(
            (panel_width, panel_height), Image.Resampling.LANCZOS
        )
    draw.text((x + 8, y + 4), label, fill="black", font=font)
    canvas.paste(panel, (x, y + label_height))

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
canvas.save(OUTPUT, dpi=(300, 300), optimize=True)
print(OUTPUT)
