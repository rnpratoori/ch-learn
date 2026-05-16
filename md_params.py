"""
Utilities for parsing MD simulation metadata and estimating
Cahn-Hilliard nondimensional timestep from LAMMPS input files
and XYZ trajectory data.
"""

import re
import numpy as np
from pathlib import Path


def parse_lammps_input(filepath):
    """Parse a LAMMPS input file for dt_md and T_lj.

    Handles simple `variable <name> equal <number>` definitions
    and resolves `timestep ${var}` references.

    Returns:
        dict with keys 'dt_md' (float) and 'T_lj' (float or None)
    """
    # LAMMPS input files often define numeric values as variables and then
    # reference them later, so parse variables before looking for `timestep`.
    variables = {}
    dt_md = None
    T_lj = None

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # First pass: collect variable definitions
    for line in lines:
        line = line.strip()
        if line.startswith('#') or not line:
            continue
        m = re.match(r'variable\s+(\w+)\s+equal\s+([\d.eE+-]+)', line)
        if m:
            variables[m.group(1)] = float(m.group(2))

    # Second pass: resolve timestep and temperature
    for line in lines:
        line = line.strip()
        if line.startswith('#') or not line:
            continue

        # Match `timestep <value>` or `timestep ${var}`
        m = re.match(r'timestep\s+(\S+)', line)
        if m:
            val = m.group(1)
            var_ref = re.match(r'\$\{(\w+)\}', val)
            if var_ref and var_ref.group(1) in variables:
                dt_md = variables[var_ref.group(1)]
            else:
                try:
                    dt_md = float(val)
                except ValueError:
                    pass

    # Temperature is optional for the current dt estimate, but recording it in
    # the result makes diagnostics easier when comparing MD input decks.
    if 'T' in variables:
        T_lj = variables['T']

    result = {'dt_md': dt_md, 'T_lj': T_lj}
    print(f"Parsed LAMMPS input ({filepath}):", flush=True)
    print(f"  dt_md = {dt_md}", flush=True)
    print(f"  T_lj = {T_lj}", flush=True)
    return result


def parse_xyz_box_size(filepath):
    """Read the first frame of an XYZ file and compute the box size.

    Assumes the box is cubic (or near-cubic) and returns the average
    side length in the simulation's native units (LJ sigma for `units lj`).

    Returns:
        float: average box side length
    """
    coords = []
    with open(filepath, 'r') as f:
        natoms = int(f.readline().strip())
        f.readline()  # comment line
        for _ in range(natoms):
            parts = f.readline().split()
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])

    coords = np.array(coords)
    # The CH solver runs on the unit domain; L enters the nondimensional time
    # conversion through the L^2 diffusion scaling.
    box_lengths = coords.max(axis=0) - coords.min(axis=0)
    L = box_lengths.mean()

    print(f"Parsed XYZ box size ({filepath}):", flush=True)
    print(f"  Coordinate ranges: x=[{coords[:,0].min():.2f}, {coords[:,0].max():.2f}], "
          f"y=[{coords[:,1].min():.2f}, {coords[:,1].max():.2f}], "
          f"z=[{coords[:,2].min():.2f}, {coords[:,2].max():.2f}]", flush=True)
    print(f"  Box lengths: {box_lengths[0]:.2f} x {box_lengths[1]:.2f} x {box_lengths[2]:.2f}", flush=True)
    print(f"  Average L = {L:.4f}", flush=True)
    return L


def estimate_dt(N1, N2, zeta, dump_interval, dt_md, L):
    """Compute the CH dimensionless timestep from the nondimensionalization.

    Formula: dt_CH = (dump_interval * dt_md) / (N_avg * zeta * L^2)

    This maps one MD frame interval to one CH solver timestep on the unit cube.

    Args:
        N1: degree of polymerization, species 1
        N2: degree of polymerization, species 2
        zeta: monomer friction coefficient (LJ units)
        dump_interval: MD steps between consecutive frames
        dt_md: MD integration timestep (LJ time units)
        L: box side length (LJ length units)

    Returns:
        float: estimated dt for the CH solver
    """
    # Use a symmetric average chain length for the mixture mobility estimate.
    # More detailed composition-dependent mobility can be introduced upstream
    # without changing the frame-to-CH timestep relation here.
    N_avg = 0.5 * (N1 + N2)
    dt_frame_lj = dump_interval * dt_md
    M_phys = 1.0 / (N_avg * zeta)
    dt_ch = M_phys * dt_frame_lj / (L ** 2)

    print(f"Estimated CH timestep:", flush=True)
    print(f"  N_avg = {N_avg}", flush=True)
    print(f"  zeta = {zeta}", flush=True)
    print(f"  M_phys = 1/(N*zeta) = {M_phys:.6f}", flush=True)
    print(f"  dt_frame = {dump_interval} * {dt_md} = {dt_frame_lj:.4f} tau_LJ", flush=True)
    print(f"  L = {L:.4f} sigma", flush=True)
    print(f"  => dt_CH = {dt_ch:.6f}", flush=True)
    return dt_ch


def auto_estimate_dt(data_dir, N1, N2, zeta, dump_interval):
    """Attempt to auto-estimate dt from MD metadata files in data_dir.

    Looks for a *.in (LAMMPS input) and *.xyz file in the data directory.

    Args:
        data_dir: path to the data directory
        N1, N2: degrees of polymerization
        zeta: monomer friction coefficient
        dump_interval: MD dump interval in timesteps

    Returns:
        float or None: estimated dt, or None if files are missing
    """
    data_path = Path(data_dir)

    # The first matching input/trajectory pair is used intentionally: HPC data
    # directories normally contain one LAMMPS input and one exported trajectory.
    in_files = list(data_path.glob('*.in'))
    xyz_files = list(data_path.glob('*.xyz'))

    if not in_files:
        print(f"No *.in file found in {data_dir}, cannot auto-estimate dt.", flush=True)
        return None
    if not xyz_files:
        print(f"No *.xyz file found in {data_dir}, cannot auto-estimate dt.", flush=True)
        return None

    md = parse_lammps_input(str(in_files[0]))
    if md['dt_md'] is None:
        print("Could not parse dt_md from LAMMPS input, cannot auto-estimate dt.", flush=True)
        return None

    L = parse_xyz_box_size(str(xyz_files[0]))
    dt = estimate_dt(N1, N2, zeta, dump_interval, md['dt_md'], L)
    return dt
