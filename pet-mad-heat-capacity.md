# PET-MAD heat capacity: LAMMPS/i-PI workflow

`pet-mad-heat-capacity.py` estimates the constant-volume heat capacity of
liquid water with path-integral molecular dynamics (PIMD). It follows the
architecture of `heat-capacity/heat-capacity.py` closely:

```text
i-PI (PIMD, NVT thermostat, scaled-coordinate estimator)
  -> Unix socket -> two LAMMPS force clients -> PET-MAD metatomic model
```

The original recipe uses LAMMPS to evaluate q-TIP4P/f. This recipe retains
i-PI, the PIMD parameters, the socket interface, and the heat-capacity
estimator, but replaces the q-TIP4P/f force field with PET-MAD through
LAMMPS's `pair_style metatomic`.

This is a software and workflow replacement, not a claim that PET-MAD and
q-TIP4P/f produce the same physical heat capacity. They are different
potential-energy surfaces.

## What each component does

| Component | Role in this workflow |
| --- | --- |
| ASE | Reads water inputs, validates O/H ordering, and writes topology-free LAMMPS data. |
| `upet` | Exports PET-MAD to a metatomic-compatible `.pt` model if the requested file is absent. |
| i-PI | Owns the ring-polymer dynamics, PILE-G thermostat, NVT ensemble, outputs, and `scaledcoords` estimator. |
| LAMMPS | Acts only as an i-PI force client through `fix ipi`; it does not integrate the trajectory. |
| metatomic | Lets LAMMPS load the exported PET-MAD model through `pair_style metatomic`. |
| NumPy | Computes the heat-capacity mean, autocorrelation-aware effective sample counts, and sampling error. |

The LAMMPS warning about no time-integration fix is expected: i-PI, not
LAMMPS, advances the dynamics in a socket-coupled simulation.

## Installation

Use the project PET-MAD environment, which contains i-PI and the metatomic
LAMMPS build:

```bash
conda env create --prefix .conda/pet-mad --file pet-mad/environment.yml
conda activate ./.conda/pet-mad

which i-pi
which lmp
lmp -h
```

`lmp -h` must list both `MISC` (for `fix ipi`) and `ML-METATOMIC` (for
`pair_style metatomic`). Use `lammps-metatomic` from the `metatensor` channel;
ordinary PyPI or conda-forge LAMMPS is not a substitute for this workflow.

## Running the script

From the repository root:

```bash
conda run --prefix .conda/pet-mad python \
  pet-mad-heat-capacity/pet-mad-heat-capacity.py
```

The first run exports the default PET-MAD-S v1.0.2 model to
`pet-mad-heat-capacity/models/pet-mad-s-v1.0.2.pt` when it is not already
present. Model export can download weights, so that first export needs network
access.

Useful controls:

```bash
# Small two-bead, two-client smoke test; not a scientific result.
conda run --prefix .conda/pet-mad python \
  pet-mad-heat-capacity/pet-mad-heat-capacity.py \
  --device cpu --beads 2 --steps 4 --stride 1 --skip 1 \
  --output-dir /tmp/pet-mad-lammps-smoke --prefix smoke --rerun

# Analyse an existing output without launching i-PI or LAMMPS.
conda run --prefix .conda/pet-mad python \
  pet-mad-heat-capacity/pet-mad-heat-capacity.py \
  --analyze --skip 1 --output-dir /tmp/pet-mad-lammps-smoke --prefix smoke

# PET-MAD model execution on a CUDA device, when supported.
conda run --prefix .conda/pet-mad python \
  pet-mad-heat-capacity/pet-mad-heat-capacity.py --device cuda --rerun
```

`--clients` defaults to two, matching the original recipe. `--lammps` and
`--ipi` can select explicit executable paths when the environment is not
activated. `--lammps-template` selects the original LAMMPS water data used to
preserve its cell and O/H atom ordering.

All simulation files are placed in `pet-mad-heat-capacity/output/` by default.
Use `--output-dir` to select another directory. `--prefix` is treated as a
filename stem inside that directory; directory components in the prefix are
ignored so generated files cannot escape the selected output folder.

## Generated files

For an output directory `<output-dir>` and prefix `<prefix>`, the script writes:

| File | Contents |
| --- | --- |
| `<output-dir>/<prefix>.xml` | Exact i-PI PIMD/socket input. |
| `<output-dir>/<prefix>.data` | Topology-free LAMMPS water data for PET-MAD. |
| `<output-dir>/<prefix>.lmp` | LAMMPS metatomic force-client input. |
| `<output-dir>/<prefix>.out` | i-PI property output, including `scaledcoords`. |
| `<output-dir>/log.lammps` and restart files | LAMMPS/i-PI runtime files. |

Use a new prefix for each test or production run. The XML and LAMMPS input are
part of the run provenance and should be kept with an interpreted result.

## How it matches the original recipe

The i-PI XML keeps the original recipe's key choices:

- 32 water molecules and the same PDB initial structure;
- eight beads by default;
- 298 K NVT PIMD;
- PILE-G thermostat with 5 fs thermostat time constant;
- 0.5 fs timestep;
- seed 32342;
- `scaledcoords(fd_delta=5e-3)` written every four steps; and
- two LAMMPS clients communicating on a Unix socket.

The generated LAMMPS input changes only the potential side:

```lammps
units metal
atom_style atomic
read_data <prefix>.data

pair_style metatomic <absolute-model-path>.pt device cuda
pair_coeff * * 8 1

fix 1 all ipi <unique-socket-name> 32342 unix
run 100000000
```

LAMMPS `fix ipi` converts between LAMMPS units and i-PI's internal atomic
units during socket communication. `units metal` is used because the PET-MAD
LAMMPS example uses it and the metatomic model provides energies/forces in the
corresponding LAMMPS-compatible convention.

## Why the PET-MAD data file is topology-free

`heat-capacity/data/water_32_data.lmp` contains q-TIP4P/f charges, bonds, and
angles. Those are required by its explicit molecular force field. PET-MAD is a
learned potential for the whole atomic configuration, so adding those terms
would double count interactions and would no longer be a PET-MAD simulation.

The script reads that original data file only to retain its cell and O/H atom
ordering, then writes `<prefix>.data` with `atom_style atomic`: positions,
masses, cell, and two atom types, but no charges, bonds, or angles. The mapping
is checked and is:

```text
LAMMPS type 1 = O -> metatomic atomic type 8
LAMMPS type 2 = H -> metatomic atomic type 1
```

Do not add `lj/cut/tip4p/long`, `kspace_style pppm/tip4p`, TIP4P pair
coefficients, bond styles, or angle styles to the PET-MAD LAMMPS input.

## Heat-capacity estimator and limitations

For `beta = 1/(kB*T)`, the scaled-coordinate estimator is:

```text
E   = <eps_v>
C_V = kB * beta^2 * (<eps_v^2> - <eps_v>^2 - <eps_v_prime>)
```

i-PI outputs `eps_v` and `eps_v_prime` in atomic units. The script discards
the first `--skip` output samples, calculates `C_V`, divides by 32 water
molecules, and reports `kB` per water molecule. Its error estimate accounts
for autocorrelation of the two terms but is only a sampling error.

It does not include PET-MAD model error, finite-size effects, bead-number bias,
finite-difference error, timestep bias, or insufficient equilibration. The
default 2,000-step run is a demonstration, not a converged calculation.

## Test status

The LAMMPS/socket workflow was tested in the project environment using two
beads, two LAMMPS clients, four steps, stride one, skip one, and CPU model
execution. Both clients connected to i-PI, PET-MAD loaded through
`pair_style metatomic`, i-PI wrote scaled-coordinate data, and analysis
completed with `6.46 +/- 1.49 kB` per water molecule. This is only an
integration smoke test; it is far too short and too small in bead count for a
physical heat-capacity result.

A one-bead, four-step CUDA smoke test also completed with the same socket
workflow, confirming that the installed LAMMPS metatomic pair style can execute
this PET-MAD model on `cuda:0`. This confirms wiring only, not GPU-integrated
trajectory propagation or a converged heat capacity.

## Production checklist

- Verify PET-MAD applicability to liquid water at the chosen density and
  temperature before interpreting results.
- Inspect the trajectory for unstable or dissociated water molecules.
- Converge beads, trajectory duration, equilibration skip, stride,
  `fd_delta`, timestep, cell size, and thermostat settings.
- Record model version/size, model path, device, LAMMPS package build, bead
  count, random seed, all XML/LAMMPS inputs, and statistical uncertainty.
- GPU model execution can accelerate PET-MAD evaluation. i-PI remains the
  trajectory driver; LAMMPS GPU integration requires an appropriate KOKKOS
  metatomic build and is not required for this socket workflow.

## References

- [LAMMPS `fix ipi`](https://docs.lammps.org/fix_ipi.html)
- [Metatomic LAMMPS interface](https://docs.metatensor.org/metatomic/latest/engines/lammps.html)
- [i-PI output tags](https://docs.ipi-code.org/output-tags.html)
- [Yamamoto scaled-coordinate estimator](https://arxiv.org/abs/physics/0505109)
