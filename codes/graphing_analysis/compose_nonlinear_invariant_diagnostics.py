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


def retitle(source_name, output_name, title, crop_top=75):
    image = Image.open(source / source_name).convert("RGB")
    image = image.crop((0, crop_top, image.width, image.height))
    header = 70
    canvas = Image.new("RGB", (image.width, image.height + header), "white")
    canvas.paste(image, (0, header))
    draw = ImageDraw.Draw(canvas)
    title_font = font(31)
    box = draw.textbbox((0, 0), title, font=title_font)
    draw.text(((canvas.width - (box[2] - box[0])) / 2, 14), title, fill="black", font=title_font)
    output = destination / output_name
    canvas.save(output, dpi=(200, 200))
    print(output)


def transfer_comparison():
    names = ("hm_energy_Zq_transfer_t3443.png", "hmr_energy_Zq_transfer_t3443.png")
    labels = (r"(a) i-delta HM", r"(b) Rk HM")
    images = [Image.open(source / name).convert("RGB") for name in names]
    images = [image.crop((0, 105, image.width, image.height)) for image in images]
    header, label_height, gap = 70, 38, 20
    width = max(image.width for image in images)
    height = header + sum(image.height + label_height for image in images) + gap
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    title = "Non-linear transfer of E and W"
    title_font = font(31)
    panel_font = font(23, bold=True)
    box = draw.textbbox((0, 0), title, font=title_font)
    draw.text(((width - (box[2] - box[0])) / 2, 14), title, fill="black", font=title_font)

    y = header
    for image, label in zip(images, labels):
        draw.text((24, y), label, fill="black", font=panel_font)
        y += label_height
        canvas.paste(image, (0, y))
        y += image.height + gap

    output = destination / "nonlinear_invariant_transfer_comparison.png"
    canvas.save(output, dpi=(200, 200))
    print(output)


destination.mkdir(parents=True, exist_ok=True)
transfer_comparison()
retitle(
    "hmr_W_budget_side_by_side.png",
    "generalized_vorticity_budget_diagnostic.png",
    "Generalized-vorticity budget diagnostic",
    crop_top=70,
)
retitle(
    "hmr_W_damping_split.png",
    "generalized_vorticity_damping_split.png",
    "Generalized-vorticity damping contributions",
    crop_top=70,
)
