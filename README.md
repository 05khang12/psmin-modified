# psmin
The minimal pseudospectral example using the 2D Hasegawa-Wakatani model, and submodules [etdrk4cp](https://github.com/gurcani/etdrk4cp) and [mlsarray](https://github.com/gurcani/mlsarray)

Note: 
Don't forget to execute:
```
git submodule update --init --recursive
```
to get all the submodules.

## Modified simulation and analysis code

- `codes/simulation_codes/`: simulation models and growth-rate calculations.
- `codes/graphing_analysis/`: plotting and diagnostic tools.
- `codes/macro_commands/`: batch runs and continuation helpers.
- `scripts/run_demokan.sh`: simulation launcher.
- `requirements-demokan.txt`: NumPy simulation dependencies.

After cloning this repository, initialize the dependencies and apply the local
NumPy backend modification:

```sh
git submodule update --init --recursive
git -C etdrk4cp apply ../etdrk4cp-numpy.patch
pip install -r requirements-demokan.txt
```

Apply the patch once on a fresh checkout. Individual analysis scripts may need
additional plotting packages. Generated simulation outputs are ignored by Git.

For setup, run examples, troubleshooting, and collaborator/Codex handoff notes, read [PROJECT_GUIDE.txt](PROJECT_GUIDE.txt).
