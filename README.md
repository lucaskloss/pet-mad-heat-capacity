# PET-MAD heat capacity with i-PI and LAMMPS

This directory is a compact, modular recipe for estimating the constant-volume
heat capacity of liquid water with path-integral molecular dynamics (PIMD).
It retains the i-PI/LAMMPS socket architecture of the original q-TIP4P/f
recipe, but evaluates energies and forces with the PET-MAD machine-learning
potential through metatomic.

```text
initial water structure
        |
        v
i-PI: PIMD integration, thermostat, observables
        | Unix-domain socket
        v
LAMMPS: fix ipi force client
        |
        v
metatomic: pair_style metatomic -> exported PET-MAD model
```

This is an educational and integration recipe, not a production calculation.
The defaults are deliberately close to the original example, but must be
converged before reporting a physical heat capacity.

## Directory layout

| File | Purpose |
| --- | --- |
| `pet-mad-heat-capacity.py` | Command-line entry point and high-level workflow. |
| `heat_capacity_analysis.py` | Yamamoto estimator, autocorrelation time, and sampling error. |
| `model_utils.py` | Finds or exports the PET-MAD model with `upet`. |
| `workflow_io.py` | Writes i-PI XML and LAMMPS files; reads i-PI output. |
| `lammps_runner.py` | Starts i-PI and LAMMPS clients and cleans up failed launches. |
| `models/` | Exported PET-MAD model files; inputs, not generated trajectory output. |
| `output/` | Default destination for one simulation's generated files. |

Keeping these responsibilities separate is useful for future work: for
example, an automatic-differentiation estimator can replace only the analysis
module without changing the simulation launch code.

## Installation

Use the repository's PET-MAD conda environment. It must contain the
metatensor-provided LAMMPS package, not an ordinary LAMMPS installation,
because this recipe needs both `fix ipi` and `pair_style metatomic`.

From the repository root:

```bash
conda env create --prefix .conda/pet-mad --file pet-mad/environment.yml
conda activate ./.conda/pet-mad

which i-pi
which lmp
lmp -h
```

`lmp -h` should list `MISC` (which provides `fix ipi`) and `ML-METATOMIC`
(which provides `pair_style metatomic`). The environment configuration uses
the `metatensor` channel and `lammps-metatomic`; do not replace it with a plain
PyPI or conda-forge LAMMPS package.

## Running a simulation

The default invocation uses PET-MAD-S v1.0.2, CUDA model evaluation, 32 water
molecules, 8 beads, 2 LAMMPS force clients, and 2,000 i-PI steps:

```bash
conda run --prefix .conda/pet-mad python \
  pet-mad-heat-capacity/pet-mad-heat-capacity.py
```

On a machine without a suitable CUDA device, select CPU explicitly:

```bash
conda run --prefix .conda/pet-mad python \
  pet-mad-heat-capacity/pet-mad-heat-capacity.py --device cpu
```

The first run exports the requested model into `models/` if it is absent. That
export may require network access to retrieve model weights.

For a short integration-only smoke test:

```bash
conda run --prefix .conda/pet-mad python \
  pet-mad-heat-capacity/pet-mad-heat-capacity.py \
  --device cpu --beads 2 --steps 4 --stride 1 --skip 1 \
  --output-dir /tmp/pet-mad-smoke --prefix smoke --rerun
```

This deliberately tiny run cannot provide a meaningful heat capacity. It only
checks that i-PI, LAMMPS, metatomic, PET-MAD, and post-processing are wired
correctly.

To analyze a completed i-PI output without launching dynamics again:

```bash
conda run --prefix .conda/pet-mad python \
  pet-mad-heat-capacity/pet-mad-heat-capacity.py \
  --analyze --output-dir /tmp/pet-mad-smoke --prefix smoke --skip 1
```

Run `--help` to see controls for temperature, beads, timestep, finite-
difference displacement, executable paths, and output locations.

## Inputs and outputs

The water PDB and legacy LAMMPS data file come from `../heat-capacity/data/`.
The legacy data file supplies the simulation cell and the oxygen/hydrogen atom
ordering only. The recipe writes a new topology-free atomic LAMMPS data file:
PET-MAD represents the entire potential-energy surface and must not be mixed
with q-TIP4P/f bonds, angles, charges, long-range TIP4P electrostatics, or
virtual sites.

The LAMMPS atom-type mapping is fixed and checked:

```text
LAMMPS type 1: O -> metatomic atomic type 8
LAMMPS type 2: H -> metatomic atomic type 1
```

All run artifacts are written to `output/` by default. `--output-dir` selects
another directory and `--prefix` selects a filename stem within it. A prefix
cannot escape the chosen output directory.

| Generated file | Meaning |
| --- | --- |
| `<prefix>.xml` | Exact i-PI PIMD and socket configuration. |
| `<prefix>.data` | Atomic LAMMPS cell, coordinates, masses, and O/H types. |
| `<prefix>.lmp` | LAMMPS metatomic `fix ipi` client input. |
| `<prefix>.out` | i-PI properties, including the two `scaledcoords` terms. |
| `log.lammps`, `RESTART` | Runtime log and i-PI restart information. |

Keep the XML and LAMMPS files with any result: they record key simulation
choices needed for reproducibility.

## How the software works together

### ASE

ASE is used here as a structure/file-format layer. It reads the water PDB,
reads the original LAMMPS data for its cell and atom ordering, validates that
the two structures agree, and writes the new `atom_style atomic` data file.
ASE does not run this PIMD trajectory.

### PET-MAD, `upet`, and metatomic

PET-MAD is a machine-learning interatomic potential: it maps an atomic
configuration to an approximate Born-Oppenheimer energy and forces. `upet`
exports the selected PET-MAD release to a portable `.pt` model. Metatomic is
the model interface used by LAMMPS to load that file:

```lammps
pair_style metatomic /absolute/path/pet-mad-s-v1.0.2.pt device cuda
pair_coeff * * 8 1
```

The `8 1` values map the LAMMPS O/H type order to the model's atomic numbers.
`--device cuda` asks metatomic to evaluate the model on a CUDA device;
`--device cpu` is the portable alternative. In this socket workflow, i-PI is
still the integrator, so GPU model evaluation does not by itself make all
trajectory operations GPU-resident.

### i-PI and LAMMPS

i-PI owns the nuclear dynamics. It initializes a ring polymer, applies NVT
PIMD with a PILE-G thermostat, advances the beads with a 0.5 fs default
timestep, and writes observables. It asks external force engines for each
bead's energy, forces, and virial.

LAMMPS is that external force engine. `fix ipi` connects to i-PI through a
unique Unix-domain socket, receives positions, evaluates PET-MAD through
metatomic, and sends the results back. LAMMPS does not integrate the
trajectory in this setup; its warning about no time-integration fix is
therefore expected. Two clients are used by default so i-PI can distribute
force evaluations across them.

## Theory

### Why path integrals are used

At finite temperature, quantum nuclei have a partition function

```math
Z = \mathrm{Tr}[\exp(-\beta \hat{H})], \qquad \beta = 1/(k_B T).
```

The path-integral discretization replaces each quantum nucleus by a classical
ring polymer of `P` beads connected by harmonic springs. As `P` is increased,
the classical configurational sampling approaches the quantum equilibrium
distribution. Each bead is evaluated by the same PET-MAD potential, while the
spring coupling represents quantum delocalization. Here, `P` is `--beads`.

PIMD samples this equilibrium distribution; it is primarily an equilibrium
sampling method. It should not be interpreted automatically as real-time
quantum dynamics.

The number of beads must be converged. Low temperature, high-frequency
vibrations, and light atoms usually require more beads. Eight beads is a
starting point inherited from the original educational recipe, not a generally
validated value.

### Machine-learning potential

An interatomic ML potential replaces an analytical force field with a model
trained on reference energies and forces. PET-MAD therefore includes the
interactions that TIP4P encoded with separate electrostatic and bonded terms.
The result is only reliable in configurations and chemical environments well
represented by the model's training data. Sampling a stable trajectory does
not prove that the model is accurate for the observable of interest.

### Constant-volume heat capacity

The quantity of interest is

```math
C_V = \left(\frac{\partial \langle E \rangle}{\partial T}\right)_V.
```

Directly differentiating noisy finite-temperature simulations is expensive.
Instead, i-PI writes the two terms of Yamamoto's scaled-coordinate estimator,
`eps_v` and `eps_v_prime`. In atomic units the implemented expression is

```math
C_V = k_B \beta^2 \left[
  \left\langle (\varepsilon_V - \langle \varepsilon_V \rangle)^2 \right\rangle
  - \left\langle \varepsilon'_V \right\rangle
\right].
```

The `scaledcoords(fd_delta=...)` output calculates the derivative-like term
with a finite displacement, controlled by `--fd-delta` (default `5e-3`). The
analysis module discards the first `--skip` output records, divides the result
by the number of water molecules, and reports `k_B` per water molecule.

The reported uncertainty estimates serial correlation using an integrated
autocorrelation time and an effective sample count. It is a sampling error
only. It does not include ML-model uncertainty, inadequate equilibration,
finite system-size effects, bead-number error, timestep error, thermostat
bias, or finite-difference-displacement bias.

## Before treating a result as scientific

- Verify that PET-MAD is applicable to liquid water at the chosen state point.
- Inspect structures and trajectories for instability or dissociation.
- Converge cell size, beads, timestep, thermostat parameters, trajectory
  length, equilibration skip, output stride, and `fd_delta`.
- Use multiple independent runs where appropriate.
- Record the exact model/version, model device, LAMMPS build, random seed,
  inputs, and statistical uncertainty.

The existing `pet-mad-heat-capacity.md` offers an implementation-oriented
runbook; `../THEORY_MOF_HEAT_CAPACITY.md` is the broader, evolving theory note
for the planned MOF and automatic-differentiation work.

## References

- [i-PI output tags](https://docs.ipi-code.org/output-tags.html)
- [LAMMPS `fix ipi`](https://docs.lammps.org/fix_ipi.html)
- [Metatomic LAMMPS interface](https://docs.metatensor.org/metatomic/latest/engines/lammps.html)
- [Yamamoto scaled-coordinate estimator](https://arxiv.org/abs/physics/0505109)
