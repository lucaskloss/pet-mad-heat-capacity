"""Generation and reading of i-PI/LAMMPS recipe files."""

from pathlib import Path
import xml.etree.ElementTree as ET

from ase import io
import ipi
import numpy as np


def build_input_xml(water_path: Path, *, temperature: float, beads: int, steps: int, timestep_fs: float,
                    output_prefix: Path, output_stride: int, fd_delta: float, socket_name: str) -> str:
    """Build the i-PI XML with an LAMMPS Unix-socket force field."""

    if beads < 1 or output_stride < 1 or steps < 1:
        raise ValueError("beads, steps, and output_stride must be positive")
    
    root = ET.Element("simulation", verbosity="medium", safe_stride="100")
    output = ET.SubElement(root, "output", prefix=str(output_prefix))
    properties = ET.SubElement(output, "properties", filename="out", stride=str(output_stride))
    properties.text = ("[ step, time{picosecond}, conserved, potential, kinetic_cv, "
                       f"scaledcoords(fd_delta={fd_delta:g}) ]")
    ET.SubElement(root, "total_steps").text = str(steps)
    prng = ET.SubElement(root, "prng")
    ET.SubElement(prng, "seed").text = "32342"
    forcefield = ET.SubElement(root, "ffsocket", name="lmpserial", mode="unix", pbc="false")
    ET.SubElement(forcefield, "address").text = socket_name
    ET.SubElement(forcefield, "latency").text = "1e-4"
    system = ET.SubElement(root, "system")
    initialize = ET.SubElement(system, "initialize", nbeads=str(beads))
    structure = ET.SubElement(initialize, "file", mode="pdb", units="angstrom")
    structure.text = str(water_path)
    ET.SubElement(initialize, "velocities", mode="thermal", units="kelvin").text = f"{temperature:g}"
    forces = ET.SubElement(system, "forces")
    ET.SubElement(forces, "force", forcefield="lmpserial").text = "lmpserial"
    ensemble = ET.SubElement(system, "ensemble")
    ET.SubElement(ensemble, "temperature", units="kelvin").text = f"{temperature:g}"
    motion = ET.SubElement(system, "motion", mode="dynamics")
    dynamics = ET.SubElement(motion, "dynamics", mode="nvt")
    thermostat = ET.SubElement(dynamics, "thermostat", mode="pile_g")
    ET.SubElement(thermostat, "tau", units="femtosecond").text = "5.0"
    ET.SubElement(dynamics, "timestep", units="femtosecond").text = f"{timestep_fs:g}"
    ET.indent(root, space="  ")

    return ET.tostring(root, encoding="unicode")


def write_lammps_data(template_path: Path, output_path: Path, water) -> None:
    """Write topology-free O/H LAMMPS data while retaining the original cell."""
    template = io.read(template_path, format="lammps-data", atom_style="full")

    if not np.array_equal(template.get_atomic_numbers(), water.get_atomic_numbers()):
        raise ValueError("LAMMPS template and i-PI structure must have identical O/H ordering")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    io.write(output_path, template, format="lammps-data", atom_style="atomic", masses=True, specorder=["O", "H"])


def write_lammps_input(input_path: Path, *, data_path: Path, model_path: Path, device: str, socket_name: str) -> None:
    """Write the PET-MAD LAMMPS force-client input."""
    input_path.write_text("units metal\natom_style atomic\n" f"read_data {data_path}\n"
                          f"pair_style metatomic {model_path} device {device}\n"
                          "pair_coeff * * 8 1\nneighbor 2.0 bin\n"
                          "neigh_modify delay 0 every 1 check yes\n"
                          f"fix 1 all ipi {socket_name} 32342 unix\nrun 100000000\n")


def read_scaledcoords(output_file: Path, fd_delta: float) -> tuple[np.ndarray, np.ndarray]:
    """Read the two scaled-coordinate columns from i-PI output."""
    data, _ = ipi.read_output(str(output_file))
    expected = f"scaledcoords(fd_delta={fd_delta:g})"
    matching_keys = (candidate for candidate in data if candidate.startswith("scaledcoords("))
    key = expected if expected in data else next(matching_keys, None)

    if key is None:
        raise KeyError("No scaledcoords output found. Available columns: " + ", ".join(data))
    
    values = np.asarray(data[key], dtype=float)

    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(f"Expected scaledcoords with shape (n, 2), got {values.shape}")
    
    return values[:, 0], values[:, 1]
