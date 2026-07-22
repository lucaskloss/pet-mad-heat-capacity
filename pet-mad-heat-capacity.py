"""Run PET-MAD/i-PI/LAMMPS PIMD and estimate water heat capacity."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

from ase import io
import numpy as np

from heat_capacity_analysis import heat_capacity_from_scaledcoords
from lammps_runner import run_lammps_socket
from model_utils import ensure_model
from workflow_io import build_input_xml, read_scaledcoords, write_lammps_data, write_lammps_input

SCRIPT_DIR = Path(__file__).resolve().parent
HEAT_CAPACITY_DIR = SCRIPT_DIR.parent / "heat-capacity"
DEFAULT_WATER = HEAT_CAPACITY_DIR / "data" / "water_32.pdb"
DEFAULT_LAMMPS_TEMPLATE = HEAT_CAPACITY_DIR / "data" / "water_32_data.lmp"
DEFAULT_VERSION = os.environ.get("PET_MAD_VERSION", "1.0.2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=SCRIPT_DIR / "models" / f"pet-mad-s-v{DEFAULT_VERSION}.pt")
    parser.add_argument("--model-version", default=DEFAULT_VERSION)
    parser.add_argument("--model-size", default="s", choices=("xs", "s"))
    parser.add_argument("--device", default="cuda", help="metatomic device, e.g. cpu or cuda")
    parser.add_argument("--water", type=Path, default=DEFAULT_WATER)
    parser.add_argument("--lammps-template", type=Path, default=DEFAULT_LAMMPS_TEMPLATE)
    parser.add_argument("--lammps", default="lmp", help="LAMMPS executable or executable name on PATH")
    parser.add_argument("--ipi", default="i-pi", help="i-PI executable or executable name on PATH")
    parser.add_argument("--clients", type=int, default=2, help="number of LAMMPS i-PI force clients")
    parser.add_argument("--temperature", type=float, default=298.0)
    parser.add_argument("--beads", type=int, default=8)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--timestep-fs", type=float, default=0.5)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--fd-delta", type=float, default=5e-3)
    parser.add_argument("--skip", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "output")
    parser.add_argument("--prefix", default="water-pet-mad-cv", help="output filename stem inside --output-dir")
    parser.add_argument("--rerun", action="store_true", help="rerun even if <prefix>.out exists")
    parser.add_argument("--analyze", action="store_true", help="only analyze an existing <prefix>.out")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    water_path = args.water.expanduser().resolve()
    template_path = args.lammps_template.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix_name = Path(args.prefix).name
    if not prefix_name or prefix_name in {".", ".."}:
        raise ValueError("--prefix must contain a valid output filename stem")
    output_prefix = output_dir / prefix_name
    output_file = Path(f"{output_prefix}.out")
    xml_file = Path(f"{output_prefix}.xml")
    data_file = Path(f"{output_prefix}.data")
    lammps_input = Path(f"{output_prefix}.lmp")
    if not water_path.exists():
        raise FileNotFoundError(f"Water structure not found: {water_path}")
    if not template_path.exists():
        raise FileNotFoundError(f"LAMMPS water template not found: {template_path}")
    water = io.read(water_path)
    if water.cell.volume <= 0:
        raise ValueError("The water structure must contain a periodic cell")
    water.pbc = True
    nmolecules = int(np.count_nonzero(water.get_atomic_numbers() == 8))
    if nmolecules == 0 or len(water) != 3 * nmolecules:
        raise ValueError("Expected a 3-atom-per-water O/H structure")
    if not args.analyze and (args.rerun or not output_file.exists()):
        default_model = SCRIPT_DIR / "models" / f"pet-mad-s-v{DEFAULT_VERSION}.pt"
        model_path = args.model
        if model_path.expanduser().resolve() == default_model.resolve():
            model_path = SCRIPT_DIR / "models" / f"pet-mad-{args.model_size}-v{args.model_version}.pt"
        model_path = ensure_model(model_path, version=args.model_version, size=args.model_size)
        socket_name = f"pet-mad-{os.getpid()}-{time.time_ns()}"
        xml_file.write_text(build_input_xml(water_path, temperature=args.temperature, beads=args.beads,
                                             steps=args.steps, timestep_fs=args.timestep_fs,
                                             output_prefix=output_prefix, output_stride=args.stride,
                                             fd_delta=args.fd_delta, socket_name=socket_name) + "\n")
        write_lammps_data(template_path, data_file, water)
        write_lammps_input(lammps_input, data_path=data_file.resolve(), model_path=model_path,
                           device=args.device, socket_name=socket_name)
        print(f"Running {args.beads}-bead PET-MAD PIMD for {args.steps} steps with {args.clients} LAMMPS clients")
        run_lammps_socket(xml_file, lammps_input, socket_name=socket_name, clients=args.clients,
                          ipi_command=args.ipi, lammps_command=args.lammps)
    elif not output_file.exists():
        raise FileNotFoundError(f"{output_file} does not exist; run the simulation first")
    else:
        print(f"Reusing existing output: {output_file}")
    eps_v, eps_v_prime = read_scaledcoords(output_file, args.fd_delta)
    result = heat_capacity_from_scaledcoords(eps_v, eps_v_prime, temperature=args.temperature,
                                             nmolecules=nmolecules, skip=args.skip)
    print(f"Heat capacity (per water molecule): {result['cv_per_molecule_kb']:.2f} +/- "
          f"{result['cv_error_per_molecule_kb']:.2f} kB")
    print("Autocorrelation times (samples): " f"delta_eps_v^2={result['tau_delta_eps_v']:.2f}, "
          f"eps_v_prime={result['tau_eps_v_prime']:.2f}")


if __name__ == "__main__":
    main()
