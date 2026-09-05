#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_archive import archive_plot


def load_budget(csv_path):
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    if data.shape == ():
        data = np.array([data])
    return data


def plot_summary(csv_path, output, title=None):
    data = load_budget(csv_path)
    t = data["t"]

    fig, axes = plt.subplots(3, 1, figsize=(8.2, 10.0), sharex=True)

    axes[0].plot(t, data["W"], linewidth=1.8, color="tab:purple")
    axes[0].set(ylabel=r"$W(t)$", title=r"Global $W$")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, data["G_minus_D_plus_T"], linewidth=1.8, label=r"$G-D+T$")
    axes[1].plot(
        t,
        data["finite_difference_dWdt"],
        "--",
        linewidth=1.3,
        label=r"finite diff. $\partial_t W$",
    )
    axes[1].set(ylabel=r"$\partial_t W$", title=r"$W$ budget check")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(t, data["G"], linewidth=1.7, label=r"$G(t)$")
    axes[2].plot(t, data["minus_D"], linewidth=1.7, label=r"$-D(t)$")
    axes[2].plot(t, data["T"], linewidth=1.7, label=r"$T(t)$")
    axes[2].set(xlabel="t", ylabel=r"terms in $\partial_t W$", title="Budget terms")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    archive_plot(output)
    plt.close(fig)


def save_single_panel(output, title, xlabel, ylabel, series):
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for item in series:
        style = item.get("style", "-")
        ax.plot(
            item["x"],
            item["y"],
            style,
            linewidth=item.get("linewidth", 1.8),
            label=item.get("label"),
            color=item.get("color"),
        )
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.grid(True, alpha=0.3)
    if any(item.get("label") for item in series):
        ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    archive_plot(output)
    plt.close(fig)


def plot_separate(csv_path, output_dir, prefix, title=None):
    data = load_budget(csv_path)
    t = data["t"]
    name = title or "W budget"
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    outputs.append(output_dir / f"{prefix}_W.png")
    save_single_panel(
        outputs[-1],
        f"{name}: global W",
        "t",
        r"$W(t)$",
        [{"x": t, "y": data["W"], "color": "tab:purple"}],
    )

    outputs.append(output_dir / f"{prefix}_budget_check.png")
    save_single_panel(
        outputs[-1],
        f"{name}: W budget check",
        "t",
        r"$\partial_t W$",
        [
            {"x": t, "y": data["G_minus_D_plus_T"], "label": r"$G-D+T$", "color": "tab:blue"},
            {
                "x": t,
                "y": data["finite_difference_dWdt"],
                "label": r"finite diff. $\partial_t W$",
                "style": "--",
                "linewidth": 1.3,
                "color": "tab:orange",
            },
        ],
    )

    outputs.append(output_dir / f"{prefix}_terms.png")
    save_single_panel(
        outputs[-1],
        f"{name}: budget terms",
        "t",
        r"terms in $\partial_t W$",
        [
            {"x": t, "y": data["G"], "label": r"$G(t)$", "color": "tab:blue"},
            {"x": t, "y": data["minus_D"], "label": r"$-D(t)$", "color": "tab:orange"},
            {"x": t, "y": data["T"], "label": r"$T(t)$", "color": "tab:green"},
        ],
    )
    return outputs


def main():
    parser = argparse.ArgumentParser(description="Plot W(t), W-budget check, and W-budget terms in one figure.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output")
    parser.add_argument("--separate-dir")
    parser.add_argument("--prefix", default="w_budget")
    parser.add_argument("--title")
    args = parser.parse_args()

    if args.output:
        plot_summary(Path(args.csv), Path(args.output), args.title)
        print(args.output)
    if args.separate_dir:
        for output in plot_separate(Path(args.csv), Path(args.separate_dir), args.prefix, args.title):
            print(output)
    if not args.output and not args.separate_dir:
        raise SystemExit("provide --output, --separate-dir, or both")


if __name__ == "__main__":
    main()
