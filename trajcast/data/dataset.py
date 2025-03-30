"""
AtomicGraphDataset: Dataset containing all information (positions, species, etc.) of a collection of atomic configurations.

Authors: Fabian Thiemann
"""

import glob
import os
from typing import Dict, List, Optional, Union

import ase
import ase.io
import torch
from numpy import load
from torch_geometric.data import InMemoryDataset

from trajcast.data._keys import POSITIONS_KEY
from trajcast.data.atomic_graph import AtomicGraph
from trajcast.utils.misc import (
    convert_ase_atoms_to_dictionary,
    convert_npz_to_dictionary,
    format_values_in_dictionary,
    guess_filetype,
)


class AtomicGraphDataset(InMemoryDataset):
    """This is the base class for handling data form atomistic simulations and converting each frame into an atomic graph."""

    def __init__(
        self,
        root: str,
        name: str,
        cutoff_radius: float,
        files: Optional[Union[List[str], str]] = "*",
        atom_type_mapper: Optional[Dict[int, int]] = {},
        time_reversibility: bool = False,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        rename=True,
        **ase_kwargs,
    ):
        """_summary_

        Args:
            root (str): _description_
            name (str): _description_
            files (Optional[Union[List[str], str]], optional): _description_. Defaults to "*".
            transform (_type_, optional): _description_. Defaults to None.
            pre_transform (_type_, optional): _description_. Defaults to None.
            pre_filter (_type_, optional): _description_. Defaults to None.
        """

        if rename:
            precision_str = str(torch.get_default_dtype())[-2:]
            time_reversibility_str = ";TR" if time_reversibility else ""
            self.name = f"{name};P:{precision_str}{time_reversibility_str}"
        else:
            self.name = name
        self.root = root
        self.cutoff_radius = cutoff_radius
        self.files = files
        self.atom_type_mapper = atom_type_mapper
        self.time_reversibility = time_reversibility

        super().__init__(
            root=root,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
        )
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.num_nodes = sum([config.num_nodes for config in self])

    @property
    def raw_file_names(self) -> List[str]:
        if isinstance(self.files, str):
            path_list = glob.glob(os.path.join(self.raw_dir, self.files))
            return [os.path.basename(path) for path in path_list]
        else:
            return self.files

    @property
    def raw_dir(self) -> str:
        return self.root

    @property
    def processed_file_names(self) -> List[str]:
        # filename contains information about cutoff, that's why _c40 for a cutoff of 4.0 Angstroms
        return [
            f"{self.name}.pt"
            # file.split(".")[0] + f"_c{int(self.cutoff_radius*10):d}.pt"
            # for file in self.raw_file_names
        ]

    @property
    def processed_dir(self) -> str:
        return self.root

    def process_pre_filter(self):
        # Override the process_pre_filter method to disable file generation
        pass

    def process_pre_transform(self):
        # Override the process_pre_transform method to disable file generation
        pass

    def _augment_for_time_reversibility(self, atoms_dictionary):
        atoms_dictionary["update_velocities"], atoms_dictionary["velocities"] = (
            -atoms_dictionary["velocities"],
            -atoms_dictionary["update_velocities"],
        )
        atoms_dictionary["positions"] = (
            atoms_dictionary["positions"] + atoms_dictionary["displacements"]
        )
        atoms_dictionary["displacements"] = -atoms_dictionary["displacements"]

        rev_atomic_graph_data = AtomicGraph.from_atoms_dict(
            atoms_dict=atoms_dictionary,
            r_cut=self.cutoff_radius,
            atom_type_mapper=self.atom_type_mapper,
        )

        return rev_atomic_graph_data

    def process(self):
        # define tuple of raw paths and processed paths
        # if len(self.raw_paths) == len(self.processed_paths):
        # it = zip(self.raw_paths, self.processed_paths)

        # loop over it
        data_list = []
        for count, raw_path in enumerate(self.raw_paths):
            # we can have either npz format or alternative files from simulation codes read in via ase
            filetype = guess_filetype(self.raw_file_names[count])
            # for npz files
            if filetype == "npz":
                raw_data = load(raw_path)
                npz_dictionary = convert_npz_to_dictionary(raw_data)
                npz_dictionary = format_values_in_dictionary(npz_dictionary)
                for index in range(npz_dictionary[POSITIONS_KEY].size(0)):
                    # create atomic graph
                    # atomic_graph_data = AtomicGraph.from_atoms_dict()
                    pass

            # for ase-readible files:
            else:
                raw_data = ase.io.read(raw_path, format=filetype, index=":")

                for index, config in enumerate(raw_data):
                    # convert each ase.Atoms object to dictionary
                    # we will use rename as well
                    atoms_dictionary = convert_ase_atoms_to_dictionary(config)
                    # next, make sure every field has the correct type
                    atoms_dictionary = format_values_in_dictionary(atoms_dictionary)
                    rev_atoms_dictionary = atoms_dictionary.copy()
                    # now create instance of AtomicGraph

                    atomic_graph_data = AtomicGraph.from_atoms_dict(
                        atoms_dict=atoms_dictionary,
                        r_cut=self.cutoff_radius,
                        atom_type_mapper=self.atom_type_mapper,
                    )

                    # time-reversed configuration
                    if (
                        "update_velocities" in rev_atoms_dictionary
                        and self.time_reversibility
                    ):
                        rev_atomic_graph_data = self._augment_for_time_reversibility(
                            rev_atoms_dictionary
                        )
                        data_list.append(rev_atomic_graph_data)

                    # append to data_list
                    data_list.append(atomic_graph_data)

        torch.save(self.collate(data_list=data_list), self.processed_paths[0])
