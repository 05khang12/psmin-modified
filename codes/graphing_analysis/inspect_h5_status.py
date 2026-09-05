#!/usr/bin/env python3
import os
import sys

import h5py as h5


for name in sys.argv[1:]:
    if not os.path.exists(name):
        print(f"{name}: missing")
        continue
    with h5.File(name, "r") as fl:
        fields_t = fl["fields/t"]
        text = f"{name}: fields {fields_t[0]} -> {fields_t[-1]} count={fields_t.shape[0]}"
        if "energies" in fl:
            energies_t = fl["energies/t"]
            text += f"; energies {energies_t[0]} -> {energies_t[-1]} count={energies_t.shape[0]}"
        print(text)
