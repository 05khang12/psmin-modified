from dataclasses import dataclass
from functools import cached_property

import numpy as np


@dataclass(frozen=True)
class GammaMax:
    kap: float = 1.0
    C: float = 10.0
    D: float = 0.15e-3
    k2_max: float = 20.0
    npoints: int = 500000

    @cached_property
    def value(self):
        k2 = np.linspace(0.0, self.k2_max, self.npoints)
        gamma = (self.kap / self.C) * k2**2 / (1.0 + k2) ** 3 - self.D * k2
        return float(np.max(gamma))

    def default_spectrum_time(self):
        return 5.0 * int(1.0 / self.value)


@dataclass(frozen=True)
class GammaMaxHWFull:
    kap: float = 1.0
    C: float = 1.0
    k2_max: float = 20.0
    npoints: int = 500000

    @staticmethod
    def gamma_plus(k2, ky, kap, C, nu=0.0, D=0.0):
        k2 = np.asarray(k2)
        ky = np.asarray(ky)
        a_k = 0.5 * ((D * k2 + C) + (C / k2 + nu * k2))
        b_k = 0.5 * ((D * k2 + C) - (C / k2 + nu * k2))
        g_k = b_k**2 + C**2 / k2
        h_k = np.sqrt(g_k**2 + C**2 * kap**2 * ky**2 / k2**2)
        return np.sqrt((h_k + g_k) / 2.0) - a_k

    @cached_property
    def value(self):
        k2 = np.linspace(np.finfo(float).tiny, self.k2_max, self.npoints)
        ky = np.sqrt(k2)
        gamma = self.gamma_plus(k2, ky, self.kap, self.C)
        return float(np.max(gamma))

    def default_spectrum_time(self):
        return 5.0 * int(1.0 / self.value)


@dataclass(frozen=True)
class GammaMaxHWFullGrid:
    k2: object
    ky: object
    kap: float = 1.0
    C: float = 1.0
    nu: float = 0.0
    D: float = 0.0

    @cached_property
    def values(self):
        k2 = np.asarray(self.k2, dtype=float)
        ky = np.asarray(self.ky, dtype=float)
        gamma = np.full_like(k2, -np.inf, dtype=float)
        active = (k2 > 0.0) & (ky > 0.0)
        gamma[active] = GammaMaxHWFull.gamma_plus(
            k2[active],
            ky[active],
            self.kap,
            self.C,
            self.nu,
            self.D,
        )
        return gamma

    @cached_property
    def value(self):
        return float(np.max(self.values))

    @cached_property
    def index(self):
        return int(np.argmax(self.values))

    def default_spectrum_time(self):
        return 5.0 * int(1.0 / self.value)
