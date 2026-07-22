"""Process management for i-PI and LAMMPS force clients."""

from pathlib import Path
import shutil
import subprocess
import time


def _wait_for_socket(socket_path: Path, server: subprocess.Popen[object]) -> None:
    """Wait until i-PI creates its Unix socket."""
    deadline = time.monotonic() + 30.0
    while not socket_path.exists():
        if server.poll() is not None:
            raise RuntimeError(f"i-PI exited before opening {socket_path}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"i-PI did not open {socket_path} within 30 seconds")
        time.sleep(0.05)


def _stop_processes(processes: list[subprocess.Popen[object]]) -> None:
    """Terminate still-running children after a launch failure."""
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        if process.poll() is None:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()


def run_lammps_socket(xml_file: Path, lammps_input: Path, *, socket_name: str, clients: int,
                      ipi_command: str, lammps_command: str) -> None:
    """Run i-PI and LAMMPS force clients using the socket architecture."""
    if clients < 1:
        raise ValueError("clients must be positive")
    ipi_executable = shutil.which(ipi_command)
    lammps_executable = shutil.which(lammps_command)
    if ipi_executable is None:
        raise FileNotFoundError(f"Could not find i-PI executable: {ipi_command}")
    if lammps_executable is None:
        raise FileNotFoundError(f"Could not find LAMMPS executable: {lammps_command}")
    workdir = xml_file.parent
    socket_path = Path("/tmp") / f"ipi_{socket_name}"
    processes: list[subprocess.Popen[object]] = []
    if socket_path.exists():
        raise FileExistsError(f"Socket path already exists: {socket_path}")
    try:
        server = subprocess.Popen([ipi_executable, xml_file.name], cwd=workdir)
        processes.append(server)
        _wait_for_socket(socket_path, server)
        clients_processes = [subprocess.Popen([lammps_executable, "-in", lammps_input.name], cwd=workdir)
                             for _ in range(clients)]
        processes.extend(clients_processes)
        server_returncode = server.wait()
        client_returncodes = [process.wait() for process in clients_processes]
        if server_returncode != 0 or any(code != 0 for code in client_returncodes):
            raise RuntimeError(f"i-PI/LAMMPS failed: i-PI={server_returncode}, LAMMPS={client_returncodes}")
    except BaseException:
        _stop_processes(processes)
        raise
