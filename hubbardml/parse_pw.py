import io
import collections
import re
from typing import List, Dict, Optional

import ase
import ase.io.espresso
import numpy as np
import pandas as pd

from . import keys

ATOM_NUM = "atom_num"
SPIN_LABEL = "spin_label"
EIGENVALS = "eigenvals"
EIGENVECS = "eigenvecs"
OCCS_MTX = "occs_mtx"
ALAT = "alat"
CELL_PARAMS = "cell_params"
ATOM_POS = "atom_pos"
ATOM_LABEL = "atom_label"
SYSTEM = "system"
RAW = "raw"
ATOM_IDX = "atom_idx"
ATOM_ELEMENT = "atom_element"
ATOM_1_IDX = "atom_1_idx"
ATOM_2_IDX = "atom_2_idx"
ATOM_2_SUPERCELL_IDX = "atom_2_supercell_idx"

IN_ATOMIC_SPECIES = "ATOMIC_SPECIES"
IN_ATOMIC_POSITIONS = "ATOMIC_POSITIONS"
IN_CELL_PARAMETERS = "CELL_PARAMETERS"
IN_K_POINTS = "K_POINTS"

OUT_OCC_BLOCKS = "occ_blocks"

# File hooks
START_NSG = "enter write_nsg"
END_NSG = "exit write_nsg"
START_FINAL_COORDINATES = "Begin final coordinates"
END_END_FINAL_COORDINATES = "End final coordinates"

# From: https://docs.python.org/3/library/re.html#simulating-scanf
FLOAT_NUMBER = r"[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?"
RE_ATOM_LINE = re.compile(r"\s+Atom:\s+(\d+)\s+Spin:\s+(\d+)")
RE_CELL_PARAMETERS = re.compile(r"CELL_PARAMETERS \(alat=\s*([\.\d]+)\)")
RE_ATOMIC_POS_LINE = re.compile(rf"([^\s]+)\s+({FLOAT_NUMBER})\s+({FLOAT_NUMBER})\s+({FLOAT_NUMBER})*")
RE_HUBBARD_ENTRY = re.compile(r"hubbard_v\((\d+),(\d+),(\d+)\)")


SiteInfo = collections.namedtuple("SiteInfo", "idx central_idx lattice_pos")


class PwInputs:
    """A class for storing PW inputs information"""

    @classmethod
    def from_file(cls, filename: str) -> "PwInputs":
        # Parse the input file
        with open(filename, "r") as file:
            return PwInputs(*ase.io.espresso.read_fortran_namelist(file))

    def __init__(self, params: Dict, system_in: List[str]):
        self.params = params
        # Now, from the input parse the system into blocks
        self.system = parse_system(system_in)
        self.system_blocks = self.system[RAW]

    def update_system(self, new_system):
        """Take atomic positions and cell parameters from new_system and overwrite the current ones"""
        for key in (IN_ATOMIC_POSITIONS, IN_CELL_PARAMETERS):
            self.system_blocks[key] = new_system[key]
        self.system = parse_system(self.generate_output())

    def generate_output(self) -> List[str]:
        return create_pw_output_lines(self.params, self.system_blocks)

    def create_ase_atoms(self) -> ase.Atoms:
        """Create an ase.Atoms object for the input structure"""
        with io.StringIO("\n".join(self.generate_output())) as buffer:
            atoms = ase.io.espresso.read_espresso_in(buffer)

        atomic_labels = []
        for line in self.system_blocks[IN_ATOMIC_POSITIONS][1:]:
            atomic_labels.append(line.split()[0])

        atoms.arrays[ATOM_LABEL] = atomic_labels

        return atoms

    def get_hubbard_dataframe(self) -> pd.DataFrame:
        """Create a dataframe of Hubbard parameters from calculation inputs"""
        atoms = self.create_ase_atoms()
        labels = atoms.arrays[ATOM_LABEL]

        # Let's find the Hubbard active sites
        site_infos = {}
        for key, value in self.params["system"].items():
            match = RE_HUBBARD_ENTRY.match(key.strip())
            if match:
                # Get the information in a format we want
                site1, site2 = map(lambda x: int(x) - 1, match.groups()[:2])
                label1, label2 = labels[site1], labels[site2]
                key = frozenset([label1, label2])
                if key in site_infos:
                    # Already done these two
                    continue

                value = float(value)

                site_info = {
                    # Site 1
                    ATOM_1_IDX: site1,
                    # Site 2
                    ATOM_2_SUPERCELL_IDX: site2,
                    # Additional params
                    keys.PARAM_TYPE: keys.PARAM_U if site1 == site2 else keys.PARAM_V,
                    keys.PARAM_IN: value,
                }

                site_infos[key] = site_info

        return pd.DataFrame(site_infos.values())

    def get_atomic_positions_dataframe(self) -> pd.DataFrame:
        rows = []
        for i, line in enumerate(self.system_blocks[IN_ATOMIC_POSITIONS][1:]):
            parts = line.split()
            rows.append(
                {
                    ATOM_IDX: i,
                    ATOM_ELEMENT: ase.io.espresso.label_to_symbol(parts[0]),
                    ATOM_LABEL: parts[0],
                    ATOM_POS: np.fromstring(" ".join(parts[1:]), sep=" "),
                }
            )

        return pd.DataFrame(rows)

    def write_output_to(self, filename):
        lines = self.generate_output()
        with open(filename, "w") as file:
            file.write("\n".join(lines))


def parse_pw_uvout(filename) -> Dict:
    parsed = {RAW: {}}

    occs_blocks = []
    with open(filename, "r") as file:
        line_iter = _line_iter(file)
        for line in line_iter:
            if START_NSG in line:
                nsg_lines = _consume_until(line_iter, END_NSG)
                occs_blocks.append(_parse_occs_blocks(nsg_lines))
            elif START_FINAL_COORDINATES in line:
                system_lines = _consume_until(line_iter, END_END_FINAL_COORDINATES, exclude_end_marker=True)
                parsed.update(parse_system(system_lines))

    parsed[OUT_OCC_BLOCKS] = occs_blocks

    return parsed


def _parse_occs_blocks(lines: List[str]):
    """Parse a group of occupation matrices"""
    line_iter = iter(lines)
    occs_block = []
    try:
        while True:
            # Keep getting occupation blocks
            occs_block.append(_parse_nsg(line_iter))
    except ValueError:
        pass

    return occs_block


def _parse_atomic_positions(lines: List[str]) -> Dict:
    line_iter = iter(lines)
    for line in line_iter:
        if IN_ATOMIC_POSITIONS in line:
            break

    # Now, let's read the coords
    atoms = {}
    kinds = []
    coords = []
    for line in line_iter:
        match = RE_ATOMIC_POS_LINE.match(line)
        if match is None:
            # Done parsing atoms
            break

        parts = line.split()
        kinds.append(parts[0])
        coords.append(np.fromstring(" ".join(parts[1:]), sep=" "))

    atoms[ATOM_POS] = np.stack(coords)

    return atoms


def _parse_cell_params(lines: List[str]) -> Dict:
    cell = {}

    lines_iter = iter(lines)
    for line in lines_iter:
        if IN_CELL_PARAMETERS in line:
            # Try and find the alat
            match = RE_CELL_PARAMETERS.match(line)
            if match:
                cell[ALAT] = float(match.groups()[0])

            break  # Found CELL_PARAMETERS

    # Now we should have the lattice vectors
    lattice_vecs = [
        np.fromstring(next(lines_iter), sep=" "),
        np.fromstring(next(lines_iter), sep=" "),
        np.fromstring(next(lines_iter), sep=" "),
    ]

    cell[CELL_PARAMS] = np.stack(lattice_vecs)

    return cell


def _parse_noop(_lines: List[str]) -> Dict:
    return {}


_in_parsing_functions = {
    IN_ATOMIC_POSITIONS: _parse_atomic_positions,
    IN_ATOMIC_SPECIES: _parse_noop,
    IN_CELL_PARAMETERS: _parse_cell_params,
    IN_K_POINTS: _parse_noop,
}


def _in_line(line: str, possibilities: List[str], where="anywhere") -> Optional[str]:
    if where == "anywhere":
        check_fn = line.__contains__
    elif where == "start":
        check_fn = line.startswith
    elif where == "end":
        check_fn = line.endswith
    else:
        raise ValueError(where)

    for possibility in possibilities:
        if check_fn(possibility):
            return possibility


def parse_system(lines: List[str]) -> Dict:
    """Parse a system in QE input format"""
    parsed = {RAW: {}}

    lines_iter = iter(lines)
    block = []
    # Keep going until we find the first key
    for line in lines_iter:
        key = _in_line(line, _in_parsing_functions.keys())
        if key:
            block.append(line)
            break

    new_key = None
    for line in lines_iter:
        # Keep going until the next key
        while not new_key:
            new_key = _in_line(line, _in_parsing_functions.keys())
            if new_key:
                break
            else:
                block.append(line)

            try:
                line = next(lines_iter)
            except StopIteration:
                break

        # Parse the input block
        parsed.update(_in_parsing_functions[key](block))
        # Also save the raw block
        parsed[RAW][key] = block

        # Set up for the next block
        block = [line]
        key = new_key
        new_key = None

    return parsed


def _parse_nsg(line_iter) -> Dict:
    match = RE_ATOM_LINE.match(next(line_iter))
    if match is None:
        raise ValueError("Could not parse occupations block")

    atom_num, spin_label = match.groups()
    next(line_iter)  # eigenvalues and eigenvectors of the occupation matrix:

    eigenvals = []
    eigenvecs = []
    while True:
        try:
            eigenval = float(next(line_iter))
        except ValueError:
            break  # occupation matrix before diagonalization:
        else:
            eigenvec = np.fromstring(next(line_iter), sep=" ")

        eigenvals.append(eigenval)
        eigenvecs.append(eigenvec)

    # Read in the occupation matrix
    occ_mtx = []
    for _ in range(len(eigenvals)):
        occ_mtx.append(np.fromstring(next(line_iter), sep=" "))

    occ_mtx = np.stack(occ_mtx)

    return {
        ATOM_NUM: atom_num,
        SPIN_LABEL: spin_label,
        EIGENVALS: eigenvals,
        EIGENVECS: eigenvecs,
        OCCS_MTX: occ_mtx,
    }


def create_pw_output_lines(params: Dict, system_blocks: Dict[str, str] = None) -> List[str]:
    """Create the list of lines for a PW input file from a set of parameters and optional system blocks"""
    pw_in = []

    for section in params:
        pw_in.append("&{0}".format(section.upper()))
        for key, value in params[section].items():
            if value is True:
                pw_in.append("   {0:16} = .true.".format(key))
            elif value is False:
                pw_in.append("   {0:16} = .false.".format(key))
            else:
                # repr format to get quotes around strings
                pw_in.append("   {0:16} = {1!r:}".format(key, value))
        pw_in.append("/")  # terminate section
    pw_in.append("")

    if system_blocks is not None:
        for val in system_blocks.values():
            pw_in.extend(val)

    return pw_in


def get_supercell_atom_map(num_atoms: int, sc_size=1) -> Dict[int, SiteInfo]:
    """Get a mapping from base indices to all supercell indices using the convention set in the
    PW Hubbard intersite code"""
    index_map = {}
    # First atoms in the main cell map to themselves
    for idx in range(num_atoms):
        index_map[idx] = SiteInfo(idx, idx, np.zeros(3, dtype=int))

    idx = num_atoms
    for nx in range(-sc_size, sc_size + 1):
        for ny in range(-sc_size, sc_size + 1):
            for nz in range(-sc_size, sc_size + 1):
                if nx == ny == nz == 0:
                    continue

                for atom_idx in range(num_atoms):
                    index_map[idx] = SiteInfo(idx, atom_idx, np.array([nx, ny, nz], dtype=int))
                    idx += 1

    return index_map


def create_occs_mtx_dataframe(outputs: Dict, iter=-1):
    rows = []
    # Create occupation matrix rows
    for occ_block in outputs[OUT_OCC_BLOCKS][iter]:
        row = {
            ATOM_IDX: int(occ_block[ATOM_NUM]) - 1,
            SPIN_LABEL: int(occ_block[SPIN_LABEL]),
            OCCS_MTX: occ_block[OCCS_MTX],
        }

        rows.append(row)

    return pd.DataFrame(rows)


def create_pair_distances_dataframe(system: ase.Atoms, rcut: float = 10.0, max_neighbours=6) -> pd.DataFrame:
    num_atoms = len(system)
    positions = system.positions
    cell_vecs = system.cell.array

    supercell_indices = get_supercell_atom_map(num_atoms)

    dist_data = []
    for i in range(num_atoms):
        pos_i = positions[i]

        for info in supercell_indices.values():
            pos_j = positions[info.central_idx]
            offset = info.lattice_pos @ cell_vecs

            dij = _dist(pos_i, pos_j + offset) / ase.units.Bohr
            if dij < rcut:
                dist_data.append(
                    {
                        ATOM_1_IDX: i,
                        ATOM_2_SUPERCELL_IDX: info.idx,
                        ATOM_2_IDX: info.central_idx,
                        keys.DIST_OUT: dij,
                    }
                )

    dist_data = pd.DataFrame(dist_data)

    if max_neighbours is not None:
        # Keep only the specified number of nearest neighbours
        indices = []
        for idx1 in range(num_atoms):
            sub_frame = dist_data[dist_data[keys.ATOM_1_IDX] == idx1]
            # num_neighbours + 1 because we want to keep the central atom
            idx = sub_frame.sort_values(keys.DIST_OUT).head(max_neighbours + 1).index
            indices.extend(idx)

        dist_data = dist_data.loc[indices]

    return dist_data


def _line_iter(file):
    line = file.readline()
    while line:
        yield line.rstrip("\n")
        line = file.readline()


def _consume_until(iterator, end_marker: str, exclude_end_marker=False) -> List[str]:
    """Consume an iterator until a given string is encountered or until the end of the stream"""
    if isinstance(iterator, io.TextIOBase):
        iterator = _line_iter(iterator)

    lines = []
    for line in iterator:
        line = line
        if end_marker in line:
            if not exclude_end_marker:
                lines.append(line)
            break
        else:
            lines.append(line)

    return lines


def _dist(r1, r2) -> float:
    """Get the Euclidian distance between two points"""
    return np.linalg.norm(r2 - r1)
