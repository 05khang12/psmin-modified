from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    ROOT
    / "outputs"
    / "C10_k1_D1p5e-4_gamma30_cross_continue_2026-06-10"
    / "plots"
)
OUTPUT = ROOT / "figures" / "results" / "gamma30_spectral_snapshots.png"

PANELS = [
    ("(a) HW", SOURCE / "hwak_gamma30_spectral_heatmap_end.png"),
    ("(b) i-delta HM", SOURCE / "hm_gamma30_spectral_heatmap_end.png"),
]

panel_width = 1234
panel_height = 1010
gap = 28
label_height = 58
margin = 24

canvas_width = 2 * panel_width + gap + 2 * margin
canvas_height = panel_height + label_height + 2 * margin
canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
draw = ImageDraw.Draw(canvas)

try:
    font = ImageFont.truetype("DejaVuSans.ttf", 38)
except OSError:
    font = ImageFont.load_default()

for index, (label, path) in enumerate(PANELS):
    x = margin + index * (panel_width + gap)
    y = margin
    draw.text((x + 8, y + 4), label, fill="black", font=font)
    with Image.open(path) as image:
        panel = image.convert("RGB").resize(
            (panel_width, panel_height), Image.Resampling.LANCZOS
        )
    canvas.paste(panel, (x, y + label_height))

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
canvas.save(OUTPUT, dpi=(300, 300), optimize=True)
print(OUTPUT)
