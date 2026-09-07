from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("New Folder")
CASES = (
    {
        "ratio": 0.1,
        "folder": ROOT / "C1_D0.1_gamma60",
        "label": r"$D=\nu=0.1\,\gamma_{\max}$",
    },
    {
        "ratio": 0.5,
        "folder": ROOT / "C1_D0.5_gamma60",
        "label": r"$D=\nu=0.5\,\gamma_{\max}$",
    },
)
OUTPUT_DIR = ROOT / "reviewer_spectral_diagnostics"
AVERAGE_FRACTION = 0.2


def grid(nx_padded, ny_padded, lx, ly, nx_dealiased, ny_dealiased):
    kx_1d = 2.0 * np.pi * np.fft.fftfreq(nx_padded, d=lx / nx_padded)
    ky_1d = 2.0 * np.pi * np.fft.rfftfreq(ny_padded, d=ly / ny_padded)
    kx, ky = np.meshgrid(kx_1d, ky_1d, indexing="ij")
    k2 = kx**2 + ky**2
    dk = min(2.0 * np.pi / lx, 2.0 * np.pi / ly)
    kx_max = (nx_dealiased // 2 - 1) * 2.0 * np.pi / lx
    ky_max = (ny_dealiased // 2 - 1) * 2.0 * np.pi / ly
    retained = (np.abs(kx) <= kx_max + 1e-12) & (ky <= ky_max + 1e-12)
    multiplicity = np.full_like(k2, 2.0)
    multiplicity[:, 0] = 1.0
    if ny_padded % 2 == 0:
        multiplicity[:, -1] = 1.0
    return kx, ky, k2, retained, multiplicity, dk


def shell_sum(values, k2, retained, multiplicity, dk):
    shell = np.rint(np.sqrt(k2) / dk).astype(int)
    valid = retained & (shell > 0)
    maximum = int(np.max(shell[valid]))
    result = np.bincount(
        shell[valid],
        weights=(multiplicity * values)[valid],
        minlength=maximum + 1,
    )
    k = np.arange(maximum + 1, dtype=float) * dk
    return k[1:], result[1:] / dk


def directional_sum(values, coordinate, retained, multiplicity, dk):
    mode = np.rint(np.abs(coordinate) / dk).astype(int)
    valid = retained & (mode > 0)
    maximum = int(np.max(mode[valid]))
    result = np.bincount(
        mode[valid],
        weights=(multiplicity * values)[valid],
        minlength=maximum + 1,
    )
    axis = np.arange(maximum + 1, dtype=float) * dk
    return axis[1:], result[1:] / dk


def all_spectra(values, kx, ky, k2, retained, multiplicity, dk):
    return {
        "radial": shell_sum(values, k2, retained, multiplicity, dk),
        "kx": directional_sum(values, kx, retained, multiplicity, dk),
        "ky": directional_sum(values, ky, retained, multiplicity, dk),
    }


def hw_response(k2, ky, c, kappa, diffusion, nu):
    response = np.zeros_like(k2, dtype=complex)
    active = (k2 > 0.0) & (ky > 0.0)
    q = k2[active]
    ky_active = ky[active]
    a = 0.5 * (diffusion * q + c + c / q + nu * q)
    b = 0.5 * (diffusion * q + c - c / q - nu * q)
    g = b**2 + c**2 / q
    h = np.sqrt(g**2 + c**2 * kappa**2 * ky_active**2 / q**2)
    w = np.sqrt((h - g) / 2.0) + 1j * np.sqrt((h + g) / 2.0)
    omega_plus = w - 1j * a
    response[active] = (c - 1j * kappa * ky_active) / (c - 1j * omega_plus)
    return response


def average_hw(path, average_gamma_window=None, gamma_ref=None):
    with h5py.File(path, "r") as handle:
        c = float(handle["data/C"][()])
        kappa = float(handle["data/kap"][()])
        diffusion = float(handle["data/D"][()])
        nu = float(handle["data/nu"][()])
        lx = float(handle["data/Lx"][()])
        ly = float(handle["data/Ly"][()])
        nx = int(handle["data/Nx"][()])
        ny = int(handle["data/Ny"][()])
        shape = handle["fields/om"].shape
        kx, ky, k2, retained, multiplicity, dk = grid(
            shape[1], shape[2], lx, ly, nx, ny
        )
        if average_gamma_window is None:
            start = int((1.0 - AVERAGE_FRACTION) * shape[0])
        else:
            times = np.asarray(handle["fields/t"])
            start_time = times[-1] - average_gamma_window / gamma_ref
            start = int(np.searchsorted(times, start_time, side="left"))
        sums = {name: np.zeros_like(k2) for name in ("energy", "drive", "coupling", "normal")}
        count = 0
        for index in range(start, shape[0]):
            om = np.asarray(handle["fields/om"][index])
            density = np.asarray(handle["fields/n"][index])
            om_k = np.fft.rfft2(om, norm="forward")
            n_k = np.fft.rfft2(density, norm="forward")
            phi_k = np.zeros_like(om_k)
            nonzero = k2 > 0.0
            phi_k[nonzero] = -om_k[nonzero] / k2[nonzero]
            phi2 = np.abs(phi_k) ** 2
            n2 = np.abs(n_k) ** 2
            sums["energy"] += k2 * phi2 + n2
            sums["drive"] += 2.0 * kappa * ky * np.imag(np.conj(n_k) * phi_k)
            sums["coupling"] += -2.0 * c * np.abs(phi_k - n_k) ** 2
            sums["normal"] += -2.0 * (nu * k2**2 * phi2 + diffusion * k2 * n2)
            count += 1
    return {
        name: all_spectra(values / count, kx, ky, k2, retained, multiplicity, dk)
        for name, values in sums.items()
    }


def average_hmr(path, average_gamma_window=None, gamma_ref=None):
    with h5py.File(path, "r") as handle:
        c = float(handle["data/C"][()])
        kappa = float(handle["data/kap"][()])
        diffusion = float(handle["data/D"][()])
        nu = float(handle["data/nu"][()])
        nu_l = float(handle["data/nu_l"][()])
        hypo_power = int(handle["data/hypo_power"][()])
        lx = float(handle["data/Lx"][()])
        ly = float(handle["data/Ly"][()])
        nx = int(handle["data/Nx"][()])
        ny = int(handle["data/Ny"][()])
        shape = handle["fields/phi"].shape
        kx, ky, k2, retained, multiplicity, dk = grid(
            shape[1], shape[2], lx, ly, nx, ny
        )
        response = hw_response(k2, ky, c, kappa, diffusion, nu)
        p = k2 + response
        nonzero = k2 > 0.0
        active = nonzero & (ky > 0.0)
        energy_weight = k2 + np.real(response)
        drive_operator = np.zeros_like(response)
        normal_operator = np.zeros_like(response)
        hypo_operator = np.zeros_like(k2)
        drive_operator[active] = -1j * kappa * ky[active] / p[active]
        normal_operator[active] = (
            -nu * k2[active] ** 2 - diffusion * k2[active] * response[active]
        ) / p[active]
        hypo_operator[nonzero] = -nu_l * k2[nonzero] ** (-0.5 * hypo_power)

        sums = {name: np.zeros_like(k2) for name in ("energy", "drive", "normal", "hypo")}
        if average_gamma_window is None:
            start = int((1.0 - AVERAGE_FRACTION) * shape[0])
        else:
            if gamma_ref is None:
                gamma_ref = float(handle["data/gamma_ref"][()])
            times = np.asarray(handle["fields/t"])
            start_time = times[-1] - average_gamma_window / gamma_ref
            start = int(np.searchsorted(times, start_time, side="left"))
        count = 0
        for index in range(start, shape[0]):
            phi = np.asarray(handle["fields/phi"][index])
            phi_k = np.fft.rfft2(phi, norm="forward")
            phi2 = np.abs(phi_k) ** 2
            sums["energy"] += energy_weight * phi2
            sums["drive"] += 2.0 * energy_weight * np.real(drive_operator) * phi2
            sums["normal"] += 2.0 * energy_weight * np.real(normal_operator) * phi2
            sums["hypo"] += 2.0 * energy_weight * hypo_operator * phi2
            count += 1
    return {
        name: all_spectra(values / count, kx, ky, k2, retained, multiplicity, dk)
        for name, values in sums.items()
    }


def positive_for_log(values):
    values = np.asarray(values)
    return np.where(values > 0.0, values, np.nan)


def plot_energy(all_results):
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=True)
    for ax, case, results in zip(axes, CASES, all_results):
        for model, color in (("HW", "#1f77b4"), ("HMR", "#d62728")):
            k, energy = results[model]["energy"]["radial"]
            ax.loglog(k, positive_for_log(energy), label=model, color=color, linewidth=2.0)
        ax.set_title(case["label"])
        ax.set_xlabel(r"$k$")
        ax.grid(True, which="both", linestyle=":", alpha=0.55)
        ax.legend()
    axes[0].set_ylabel(r"$E(k)$")
    fig.suptitle(r"Late-time energy spectra, $C=1$, averaged over final 20\%")
    fig.tight_layout()
    output = OUTPUT_DIR / "C1_gamma60_energy_spectra_loglog.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def symlog_threshold(series):
    magnitudes = np.concatenate([np.abs(values[np.isfinite(values)]) for values in series])
    positive = magnitudes[magnitudes > 0.0]
    return max(float(np.max(positive)) * 1.0e-5, float(np.min(positive)))


def plot_rates(all_results):
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.0), sharex=True)
    for row, (case, results) in enumerate(zip(CASES, all_results)):
        hw = results["HW"]
        hmr = results["HMR"]

        ax = axes[row, 0]
        hw_series = [hw[name]["radial"][1] for name in ("drive", "coupling", "normal")]
        for name, label, color, style in (
            ("drive", "gradient injection", "#2ca02c", "-"),
            ("coupling", "resistive coupling", "#9467bd", "--"),
            ("normal", "normal dissipation", "#d62728", "-."),
        ):
            ax.plot(hw[name]["radial"][0], hw[name]["radial"][1], label=label, color=color, linestyle=style, linewidth=1.8)
        ax.set_yscale("symlog", linthresh=symlog_threshold(hw_series))
        ax.set_title(f"HW, {case['label']}")

        ax = axes[row, 1]
        hmr_series = [hmr[name]["radial"][1] for name in ("drive", "normal", "hypo")]
        for name, label, color, style in (
            ("drive", "gradient injection", "#2ca02c", "-"),
            ("normal", "normal dissipation", "#d62728", "-."),
            ("hypo", "hypodiffusion", "#ff7f0e", "--"),
        ):
            ax.plot(hmr[name]["radial"][0], hmr[name]["radial"][1], label=label, color=color, linestyle=style, linewidth=1.8)
        ax.set_yscale("symlog", linthresh=symlog_threshold(hmr_series))
        ax.set_title(f"HMR, {case['label']}")

        for ax in axes[row, :]:
            ax.axhline(0.0, color="black", linewidth=0.8)
            ax.grid(True, which="both", linestyle=":", alpha=0.55)
            ax.legend(fontsize=8)
            ax.set_ylabel(r"spectral contribution to $\partial_t E$")

    for ax in axes[-1, :]:
        ax.set_xlabel(r"$k$")
    fig.suptitle(r"Late-time injection and dissipation spectra, $C=1$, final 20\% average")
    fig.tight_layout()
    output = OUTPUT_DIR / "C1_gamma60_injection_dissipation_spectra_symlog.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_directional_energy(all_results):
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.8), sharex="col", sharey="row")
    directions = (("kx", r"$|k_x|$", r"$E(|k_x|)$"), ("ky", r"$k_y$", r"$E(k_y)$"))
    for column, (case, results) in enumerate(zip(CASES, all_results)):
        for row, (direction, xlabel, ylabel) in enumerate(directions):
            ax = axes[row, column]
            for model, color in (("HW", "#1f77b4"), ("HMR", "#d62728")):
                coordinate, energy = results[model]["energy"][direction]
                ax.loglog(
                    coordinate,
                    positive_for_log(energy),
                    label=model,
                    color=color,
                    linewidth=2.0,
                )
            ax.set_title(f"{ylabel}, {case['label']}")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True, which="both", linestyle=":", alpha=0.55)
            ax.legend()
    fig.suptitle(r"Late-time directional energy spectra (log--log), $C=1$, final 20\% average")
    fig.tight_layout()
    output = OUTPUT_DIR / "C1_gamma60_energy_spectra_kx_ky_loglog.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_directional_rates(all_results, model):
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.8), sharex="col")
    directions = (("kx", r"$|k_x|$"), ("ky", r"$k_y$"))
    if model == "HW":
        terms = (
            ("drive", "gradient injection", "#2ca02c", "-"),
            ("coupling", "resistive coupling", "#9467bd", "--"),
            ("normal", "normal dissipation", "#d62728", "-."),
        )
    else:
        terms = (
            ("drive", "gradient injection", "#2ca02c", "-"),
            ("normal", "normal dissipation", "#d62728", "-."),
            ("hypo", "hypodiffusion", "#ff7f0e", "--"),
        )

    for row, (case, results) in enumerate(zip(CASES, all_results)):
        spectra = results[model]
        for column, (direction, xlabel) in enumerate(directions):
            ax = axes[row, column]
            series = [spectra[name][direction][1] for name, _, _, _ in terms]
            for name, label, color, style in terms:
                coordinate, values = spectra[name][direction]
                ax.plot(
                    coordinate,
                    values,
                    label=label,
                    color=color,
                    linestyle=style,
                    linewidth=1.8,
                )
            ax.set_yscale("symlog", linthresh=symlog_threshold(series))
            ax.axhline(0.0, color="black", linewidth=0.8)
            ax.set_title(f"{case['label']}, spectrum in {xlabel}")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(r"contribution to $\partial_t E$")
            ax.grid(True, which="both", linestyle=":", alpha=0.55)
            ax.legend(fontsize=8)

    fig.suptitle(
        rf"{model} directional injection and dissipation spectra, "
        rf"$C=1$, final 20\% average"
    )
    fig.tight_layout()
    output = OUTPUT_DIR / f"C1_gamma60_{model}_injection_dissipation_kx_ky_symlog.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []
    for case in CASES:
        print(f"Processing {case['folder']}")
        all_results.append(
            {
                "HW": average_hw(case["folder"] / "hw" / "out_hwak.h5"),
                "HMR": average_hmr(case["folder"] / "hmr_full" / "out_hmr_full.h5"),
            }
        )
    outputs = (
        plot_energy(all_results),
        plot_rates(all_results),
        plot_directional_energy(all_results),
        plot_directional_rates(all_results, "HW"),
        plot_directional_rates(all_results, "HMR"),
    )
    for output in outputs:
        print(output.resolve())


if __name__ == "__main__":
    main()
