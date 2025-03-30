"""
AtomicGraph: construction of the atomic graph network.

Authors: Fabian Thiemann
"""

from typing import Dict

import ase
import torch
from torch_cluster import radius_graph
from torch_geometric.data import Data
from torch_nl import compute_neighborlist

from trajcast.utils.atomic_computes import find_edges_for_periodic_system

from ._keys import (
    ASE_ARRAY_FIELDS,
    ASE_INFO_FIELDS,
    ATOM_TYPES_KEY,
    ATOMIC_MASSES_KEY,
    ATOMIC_NUMBERS_KEY,
    CELL_KEY,
    CELL_SHIFTS_KEY,
    PBC_KEY,
    POSITIONS_KEY,
    TOTAL_MASS_KEY,
)
from ._types import DTYPE_MAPPING, FIELD_TYPES


class AtomicGraph(Data):
    """_summary_

    Args:
        Data (_type_): _description_exit
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def ave_n_neighbors(self):
        return self.num_edges / self.num_nodes

    @classmethod
    def from_atoms_dict(
        cls, atoms_dict: Dict, r_cut: float, atom_type_mapper: dict = None, **kwargs
    ):
        """_summary_

        Args:
            atoms_dict (Dict): _description_
            r_cut (float): _description_
            atom_type_mapper (dict): _description_

        Returns:
            _type_: _description_
        """
        # we assume the atoms_dict is already formatted with the keys corresponding to those given in _keys
        # extract positions and set as nodes
        pos = atoms_dict.get(POSITIONS_KEY)
        cell = atoms_dict.get(CELL_KEY)
        pbc = atoms_dict.get(PBC_KEY)

        # pop positoins from the dict
        atoms_dict.pop(POSITIONS_KEY)

        # in case we don't deal with a periodic system (rarely)
        if not any(pbc):
            # get source and destination of edge
            pos = torch.from_numpy(pos) if not isinstance(pos, torch.Tensor) else pos
            edge_dst, edge_src = radius_graph(pos, r_cut, max_num_neighbors=len(pos))
            # build edge_index for pyg data object
            edge_index = torch.vstack([edge_src, edge_dst])
            shifts = torch.zeros(edge_index.size(-1), 3).to(torch.get_default_dtype())

        else:
            # extract edge indices in accordance with pbc
            edge_index, shifts = find_edges_for_periodic_system(
                node_positions=pos, pbc=pbc, cell=cell, r_cut=r_cut
            )

        # get kwargs
        kwargs[CELL_SHIFTS_KEY] = shifts
        # make sure we know which atom types can be present in the dataset
        if not atom_type_mapper:
            atom_type_mapper = {
                atomic_number.item(): count
                for count, atomic_number in enumerate(
                    atoms_dict[ATOMIC_NUMBERS_KEY].unique()
                )
            }
        kwargs[ATOM_TYPES_KEY] = torch.tensor(
            [
                atom_type_mapper[atomic_number.item()]
                for atomic_number in atoms_dict[ATOMIC_NUMBERS_KEY]
            ],
            dtype=torch.long,
        ).view(-1, 1)

        # compute total masses
        kwargs[TOTAL_MASS_KEY] = atoms_dict[ATOMIC_MASSES_KEY].sum()

        kwargs.update(atoms_dict)

        return cls(pos=pos, edge_index=edge_index, cutoff_radius=r_cut, **kwargs)

    @property
    def ASEAtomsObject(self) -> ase.Atoms:
        atoms_obj = ase.Atoms(
            positions=self.pos.cpu().detach().numpy(),
            numbers=self.__getattr__(ATOMIC_NUMBERS_KEY).squeeze(1).cpu().numpy(),
            pbc=self.get(PBC_KEY, [False, False, False]).cpu(),
            cell=self.get(CELL_KEY, [0, 0, 0]).cpu(),
        )

        # change the type of the positions to align with torch
        atoms_obj.arrays["positions"] = atoms_obj.positions.astype(
            DTYPE_MAPPING[FIELD_TYPES[POSITIONS_KEY]]
        )

        # fill info
        for field in ASE_INFO_FIELDS:
            if hasattr(self, field):
                atoms_obj.info[field] = self.__getattr__(field).cpu().numpy()
        # array fields
        for field in ASE_ARRAY_FIELDS:
            if hasattr(self, field):
                atoms_obj.arrays[field] = self.__getattr__(field).cpu().detach().numpy()

        return atoms_obj

    def update_node_attributes(self, attr_values_dict: Dict):
        for attr, values in attr_values_dict.items():
            self.__setattr__(attr, values)

    def update_edge_index(self):
        self.edge_index, _, shifts_idx = compute_neighborlist(
            self.cutoff_radius,
            self.pos,
            self.cell,
            self.pbc,
            self.batch,
            self_interaction=False,
        )
        self.shifts = torch.einsum("jn,jnm->jm", shifts_idx, self.cell.view(-1, 3, 3))

    def compute_edge_vectors(self):
        edge_vectors = (
            self.pos[self.edge_index[1]] - self.pos[self.edge_index[0]] + self.shifts
        )
        len_edge_vectors = torch.linalg.norm(edge_vectors, dim=-1, keepdim=True)
        self.edge_vectors = edge_vectors
        self.edge_lengths = len_edge_vectors
