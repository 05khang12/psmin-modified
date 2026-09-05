from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


root = Path(__file__).resolve().parents[2]
source = root / "outputs" / "C10_D1p5e-4_gamma50_2026-07-17" / "plots"
destination = root / "figures" / "results"


def font(size, bold=False):
    try:
        return ImageFont.truetype("arialbd.ttf" if bold else "arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def compose(names, labels, output_name, title):
    images = [Image.open(source / name).convert("RGB") for name in names]
    images = [image.crop((0, 58, image.width, image.height)) for image in images]
    gap, header, label_height = 22, 72, 34
    panel_w = max(image.width for image in images)
    panel_h = max(image.height for image in images) + label_height
    width = 2 * panel_w + gap
    height = header + 2 * panel_h + gap
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    title_font = font(31)
    panel_font = font(22, bold=True)
    box = draw.textbbox((0, 0), title, font=title_font)
    draw.text(((width - (box[2] - box[0])) / 2, 12), title, fill="black", font=title_font)

    positions = (
        (0, header),
        (panel_w + gap, header),
        ((width - panel_w) // 2, header + panel_h + gap),
    )
    for image, label, (x, y) in zip(images, labels, positions):
        draw.text((x + 20, y), label, fill="black", font=panel_font)
        canvas.paste(image, (x, y + label_height))

    destination.mkdir(parents=True, exist_ok=True)
    output = destination / output_name
    canvas.save(output, dpi=(200, 200))
    print(output)


labels = ("(a) HW", r"(b) i-delta HM", r"(c) Rk HM")
compose(
    ("hw_E_Ez_overlay.png", "hm_E_Ez_overlay.png", "hmr_E_Ez_overlay.png"),
    labels,
    "hm_hypoviscosity_energy_comparison.png",
    "Energy evolution",
)
compose(
    ("hw_Eky_t3443.png", "hm_Eky_t3443.png", "hmr_Eky_t3443.png"),
    labels,
    "hm_hypoviscosity_Eky_comparison.png",
    "E(ky) at t = 50",
)
