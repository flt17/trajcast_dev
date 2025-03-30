"""
Preprocessing of MD trajectories to adjust labels and properties of interest.

Authors: Fabian Thiemann
"""

import os
from typing import Dict, List, Optional, Set, Union

import ase
import ase.io
import numpy as np

# Import all keys
from trajcast.data._keys import (
    ASE_ARRAY_FIELDS,
    ASE_INFO_FIELDS,
    ATOMIC_NUMBERS_KEY,
    CELL_KEY,
    DISPLACEMENTS_KEY,
    FORCES_KEY,
    INPUT_KEY_MAPPING,
    PBC_KEY,
    POSITIONS_KEY,
    TIMESTEP_KEY,
    TOTAL_ENERGY_KEY,
    UPDATE_KEY,
    VELOCITIES_KEY,
)
from trajcast.utils.atomic_computes import align_vectors_with_periodicity
from trajcast.utils.misc import (
    convert_ase_atoms_to_dictionary,
    invert_dictionary,
    string2index,
    truncate_dictionary,
)

from .wrappers._lammps import lammps_dump_to_ase_atoms

_TRAJECTORY_FIELDS: Set[str] = {
    POSITIONS_KEY,
    ATOMIC_NUMBERS_KEY,
    PBC_KEY,
    CELL_KEY,
    FORCES_KEY,
    VELOCITIES_KEY,
    TOTAL_ENERGY_KEY,
    DISPLACEMENTS_KEY,
    TIMESTEP_KEY,
}


class Trajectory:
    """_summary_"""

    def __init__(
        self,
        data: Union[List[ase.Atoms], Dict] = None,
        available_fields: Set[str] = _TRAJECTORY_FIELDS,
    ):
        """_summary_

        Args:
        """

        self.available_fields = available_fields
        self.data = data

    def _validate_chosen_fields(self, chosen_fields: Set[str]) -> bool:
        if not chosen_fields.issubset(
            self.available_fields
        ) or not chosen_fields.issuperset(
            set((POSITIONS_KEY, ATOMIC_NUMBERS_KEY, PBC_KEY, CELL_KEY))
        ):
            return False
        else:
            return True

    def compute_additional_fields(
        self,
        add_fields: Set[str] = {DISPLACEMENTS_KEY},
        time_step: int = 1,
        time_step_in_fs: float = None,
        truncate: bool = True,
    ):
        """This function takes a trajectory object and computes based on the input addtional fields.
        In the TrajCast context, this mainly involves the displacements vectors and the new velocities.
        However, other properties can be computed as well.

        Args:
            add_fields (Set[str], optional): Set of properties/fields to compute. Defaults to {DISPLACEMENTS_KEY}.
            time_step (int, optional):  How many frames to look forward when computing the properties. Defaults to 1 which means take the next frame.
            time_step_in_fs (float, optional): Time difference between prediction and reference in fs. Default is None as usually we take the time_step.
            truncate (bool, optional): Whether to truncate the trajectory such that all frames have the same fields.
              Defaults to True.

        Raises:
            KeyError: If certain fields cannot be computed because of missing inputs.
                For instance, we would like to compute the update_velocities but do not have the velocities.
        """
        time_between_frames = (
            self.time_between_frames if hasattr(self, "time_between_frames") else 1.0
        )

        if time_step_in_fs:
            # convert time_step from float to number of frames
            time_step = int(time_step_in_fs / time_between_frames)

        if DISPLACEMENTS_KEY in add_fields:
            self.data = compute_atomic_displacement_vectors(
                trajectory_data=self.data,
                time_step=time_step,
                time_between_frames=time_between_frames,
                key_mapping=self.mapping_available_fields,
            )
            self.available_fields.update({DISPLACEMENTS_KEY, TIMESTEP_KEY})

        # fields for which we would like to the information from the next frame
        update_fields = [field for field in add_fields if UPDATE_KEY in field]

        if update_fields:
            # get raw fields
            raw_fields = [field.split(f"{UPDATE_KEY}_")[-1] for field in update_fields]
            if not set(raw_fields).issubset(self.available_fields):
                raise KeyError(
                    "Some of the update fields cannot be used as we do not have the original field either."
                )
            self.data = get_desired_field_values_of_next_frame(
                fields=raw_fields,
                trajectory_data=self.data,
                time_step=time_step,
                key_mapping=self.mapping_available_fields,
            )

            self.available_fields.update(set(update_fields))

        if truncate:
            self._truncate_trajectory(time_step=time_step)


class ASETrajectory(Trajectory):
    def __init__(
        self,
        ase_atoms_list: List[ase.Atoms],
        key_mapping: Optional[Dict[str, str]] = invert_dictionary(INPUT_KEY_MAPPING),
        time_between_frames: Optional[float] = 1.0,
        apply_wrapping: Optional[bool] = False,
        apply_unwrapping: Optional[bool] = False,
    ):
        super().__init__()

        self.data = ase_atoms_list
        self.n_frames = len(ase_atoms_list)
        # convert ase_atoms_list into dictionary to get available fields and mapping
        dictionary = convert_ase_atoms_to_dictionary(self.data[0], rename=False)

        self.mapping_available_fields = {
            key_mapping.get(key, key): key for key in dictionary.keys()
        }
        self.available_fields = set(self.mapping_available_fields.keys())
        self.time_between_frames = time_between_frames

        if CELL_KEY in self.available_fields:
            self._guess_wrapping()

        # unwrapping
        if apply_unwrapping:
            self.unwrap()
        if apply_wrapping:
            self.wrap()

    def _guess_wrapping(self):
        # get scaled positions of last frame (this is why we call it guess and not check wrapping)
        # it is approximated
        scaled_pos = self.data[-1].get_scaled_positions(wrap=False)
        if np.min(scaled_pos) < 0 or np.max(scaled_pos) > 1:
            self._is_wrapped = False
        else:
            self._is_wrapped = True

    @property
    def is_wrapped(self):
        return self._is_wrapped

    @is_wrapped.setter
    def is_wrapped(self, value: bool):
        # given our guess of whether a trajectory is wrapped is not bullet proof
        # if a user requests wrapping we will also do the routine even if the trajectory
        # is already wrapped
        if not value:
            self.unwrap()
        else:
            self.wrap()

    def _truncate_trajectory(self, time_step):
        self.data = self.data[:-time_step]

    @classmethod
    def read_from_file(
        cls,
        root: str,
        filename: str,
        key_mapping: Optional[Dict[str, str]] = invert_dictionary(INPUT_KEY_MAPPING),
        frame_interval: Optional[float] = None,
        wrapper: Optional[str] = None,
        wrapper_kwargs: Optional[Dict] = None,
        apply_wrapping: Optional[bool] = False,
        apply_unwrapping: Optional[bool] = False,
        **ase_kwargs,
    ):
        """_summary_

        Args:
            root (str): _description_
            filename (str): _description_
            key_mapping (Optional[Dict[str, str]], optional): _description_. Defaults to invert_dictionary(INPUT_KEY_MAPPING).
            frame_interval (Optional[float], optional): Time in fs between frames in the trajectory file. Defaults to 1.0.

        Raises:
            FileNotFoundError: _description_
            TypeError: _description_

        Returns:
            _type_: _description_
        """
        path_to_file = os.path.join(root, filename)
        # check whether file exists
        if not os.path.exists(path_to_file):
            raise FileNotFoundError(f"Path:{path_to_file} does not exist.")

        # read in file via ase
        if not wrapper:
            ase_atoms_list = ase.io.read(path_to_file, **ase_kwargs)
            # do forces manually

            if hasattr(ase_atoms_list[0], "calc") and ase_atoms_list[0].calc:
                for frame in ase_atoms_list:
                    frame.arrays[FORCES_KEY] = frame.get_forces()

        else:
            # pick wrapper
            ase_atoms_list = {"lammps": lammps_dump_to_ase_atoms}[wrapper](
                path_to_file=path_to_file, **wrapper_kwargs, **ase_kwargs
            )
        # check whether more than one frame got read in
        if not isinstance(ase_atoms_list, List):
            raise TypeError(
                f"Expected a list but received {type(ase_atoms_list).__name__}"
            )

        # last infer the time between frames
        step_read = string2index(ase_kwargs.get("index")).step
        step_read = 1 if not step_read else step_read
        # to get the the time between frames in our object we have to multiply this
        # by the time between frames which can be adjusted by the user
        # If not given, it will check in the file for the timestep (assuming the positions are printed every frame)
        # and default to 1.0fs other.
        frame_interval = (
            frame_interval
            if frame_interval
            else ase_atoms_list[0].info.get(TIMESTEP_KEY, 1.0)
        )
        time_between_frames = step_read * frame_interval

        # instantiate class
        return cls(
            ase_atoms_list=ase_atoms_list,
            key_mapping=key_mapping,
            time_between_frames=time_between_frames,
            apply_wrapping=apply_wrapping,
            apply_unwrapping=apply_unwrapping,
        )

    def write_to_file(
        self,
        root: str = "./",
        filename_prefix: Optional[str] = "trajectory",
        chosen_fields: Optional[Set[str]] = set(),
        npz_format: Optional[bool] = False,
        **ase_kwargs,
    ):
        """Write a trajectory with all/a subset of frames to a file at the location of choice in the format of choice.
            This function builds on ase.io.write and further information can be found here:
            https://wiki.fysik.dtu.dk/ase/ase/io/io.html

        Args:
            root (str, optional): The directory where the file will be written to. Defaults to "./".
            filename_prefix (Optional[str], optional): Prefix of the trajectory name. Defaults to "trajectory".
            chosen_fields (Optional[Set[str]], optional): The set of fields we would like to save for each frame.
                Some fields like positions or atomic numbers are required while others like velocities or forces can be added.
                Defaults to set().
            npz_format (Optional[bool], optional): Whether the trajectory should saved as npz format.
                This is currently not implememnted. Defaults to False.
            ase_kwargs: Any arguments like format or index passed to the writer later.

        Raises:
            FileNotFoundError: If the directory does not exist.
            KeyError: If there is a field included which does not exist.
                Alternatively, this can also be raised if requirements like the positions or atomic numbers are not given.
            NotImplementedError: If npz format is required
        """
        # check whether root exists
        if not os.path.exists(root):
            raise FileNotFoundError("Directory does not exist, please create it!")

        if npz_format:
            raise NotImplementedError("NPZ format is currently not supported.")

        if not chosen_fields:
            chosen_fields = self.available_fields

        if not self._validate_chosen_fields(chosen_fields):
            raise KeyError(
                "Some elements of chosen fields are not available in the trajectory or minimum requirements (atomic numbers and positions) not satisfied."
            )

        # pop all fields not in chosen fields from data_objects
        modified_ase_atoms_list = self._modify_ase_atoms_list_based_on_chosen_fields(
            chosen_fields=chosen_fields
        )

        if not npz_format:
            # extract format from ase_kwargs
            file_format = (
                ase_kwargs.get("format") if "format" in ase_kwargs.keys() else "extxyz"
            )
            path_to_file = os.path.join(root, f"{filename_prefix}.{file_format}")

            ase.io.write(path_to_file, modified_ase_atoms_list, **ase_kwargs)

    def _modify_ase_atoms_list_based_on_chosen_fields(
        self, chosen_fields: Set[str]
    ) -> List[ase.Atoms]:
        """Modifies ase.Atoms list according to chosen fields which will then be saved.

        Args:
            chosen_fields (Set[str]): _description_

        Returns:
            List[ase.Atoms]: _description_
        """
        # simply loop over all frames
        local_ase_atoms_list = self.data.copy()
        for frame in local_ase_atoms_list:
            # turn off calculator
            frame.set_calculator()

            # start with info
            [
                frame.info.pop(self.mapping_available_fields[key])
                for key in ASE_INFO_FIELDS
                if {v: k for k, v in self.mapping_available_fields.items()}.get(key)
                in frame.info
                and key not in chosen_fields
            ]

            # do the same for arrays
            [
                frame.arrays.pop(self.mapping_available_fields[key])
                for key in ASE_ARRAY_FIELDS
                if {v: k for k, v in self.mapping_available_fields.items()}.get(key)
                in frame.arrays.keys()
                and key not in chosen_fields
            ]

        return local_ase_atoms_list

    def to_mdanalysis_universe(self):
        try:
            from MDAnalysis import Universe

        except ImportError:
            print(
                "To convert ASETrajectory into a MDAnalysis.Universe please install MDAnalysis."
            )

        # we'll build the universe from scratch, so first
        universe = Universe.empty(
            n_atoms=len(self.data[0]),
            trajectory=True,
            velocities=True if VELOCITIES_KEY in self.available_fields else False,
            forces=True if FORCES_KEY in self.available_fields else False,
        )

        # next we start adding topological information
        universe.add_TopologyAttr("name", self.data[0].get_chemical_symbols())
        universe.add_TopologyAttr("type", self.data[0].get_chemical_symbols())
        universe.add_TopologyAttr("masses", self.data[0].get_masses())

        # now we add the positions/trajectory
        # check whether forces and velocities are given
        # we are doing this in this crude way for the sake of efficiency
        vel_key = self.mapping_available_fields.get(VELOCITIES_KEY)
        force_key = self.mapping_available_fields.get(FORCES_KEY)

        # velocities but no forces
        if vel_key and not force_key:
            output = np.asarray(
                [
                    np.asarray([frame.positions, frame.arrays[vel_key]])
                    for frame in self.data
                ]
            )
            coordinates = output[:, 0, :, :]
            velocities = output[:, 1, :, :]

        # velocities and forces
        elif vel_key and force_key:
            output = np.asarray(
                [
                    np.asarray(
                        [
                            frame.positions,
                            frame.arrays[vel_key],
                            frame.arrays[force_key],
                        ]
                    )
                    for frame in self.data
                ]
            )
            coordinates = output[:, 0, :, :]
            velocities = output[:, 1, :, :]
            forces = output[:, 2, :, :]

        # forces but no velocities
        elif force_key and not vel_key:
            output = np.asarray(
                [
                    np.asarray([frame.positions, frame.arrays[force_key]])
                    for frame in self.data
                ]
            )
            coordinates = output[:, 0, :, :]
            forces = output[:, 1, :, :]

        # neither forces nor velocities
        else:
            output = np.asarray([frame.positions for frame in self.data])
            coordinates = output

        # load into trajectory
        # for now only nvt, so fix the box
        universe.load_new(
            coordinates,
            velocities=velocities if vel_key else None,
            forces=forces if force_key else None,
            dimensions=self.data[0].cell.cellpar(),
            dt=self.time_between_frames
            * 0.001,  # MDAnalysis takes the timestep in ps, that's why divided by 1000
        )

        return universe

    def unwrap(self):
        # let's first check there is a cell key given
        if not self.data[0].__getattribute__(CELL_KEY):
            raise KeyError(
                "Cell not found, please define otherwise how do you expect to unwrap?"
            )

        # we start by getting all coordinates
        coordinates = np.asarray([frame.positions for frame in self.data])

        # for now we assume nvt
        if not np.alltrue(self.data[0].cell == self.data[1].cell):
            raise NotImplementedError("Only NVT Ensemble so far.")

        # now we get the displacement vectors
        displacement_vectors = np.diff(
            coordinates, prepend=coordinates[0][np.newaxis, :, :], axis=0
        )
        displacement_vectors_unwrapped = align_vectors_with_periodicity(
            displacement_vectors, self.data[0].cell
        )
        # finally we only need to add all the displacement vectors additively to the initial frame
        coordinates = coordinates[0] + np.cumsum(displacement_vectors_unwrapped, axis=0)

        # now we update the coordinates
        for frame_index, frame in enumerate(self.data):
            frame.set_positions(coordinates[frame_index], apply_constraint=False)

        # set the keyword
        self._is_wrapped = False

    def wrap(self):
        # let's first check there is a cell key given
        if not self.data[0].__getattribute__(CELL_KEY):
            raise KeyError(
                "Cell not found, please define otherwise how do you expect to unwrap?"
            )

        # wrapping is easy and we can use ase for this:
        for frame_index, frame in enumerate(self.data):
            frame.wrap()

        # set the keyword
        self._is_wrapped = True


class NPZTrajectory(Trajectory):
    def __init__(
        self,
        npz_dictionary: Dict,
        key_mapping: Dict[str, str] = invert_dictionary(INPUT_KEY_MAPPING),
    ):
        super().__init__()

        self.data = npz_dictionary

        # quickly check available fields
        self.mapping_available_fields = {
            key_mapping.get(key, key): key for key in self.data.keys()
        }
        self.n_frames = self.data[self.mapping_available_fields[POSITIONS_KEY]].shape[0]
        self.available_fields = set(self.mapping_available_fields.keys())

    def _truncate_trajectory(self, time_step):
        self.data = truncate_dictionary(
            dictionary=self.data,
            n_values=self.n_frames - time_step,
        )

    @classmethod
    def read_from_file(
        cls,
        root: str,
        filename: str,
        indices: Optional[Union[str, List[int]]] = ":",
        key_mapping: Dict[str, str] = invert_dictionary(INPUT_KEY_MAPPING),
    ):
        path_to_file = os.path.join(root, filename)
        # check whether file exists
        if not os.path.exists(path_to_file):
            raise FileNotFoundError

        # read in data
        npz_object = np.load(path_to_file)
        # get the dictionary accounting for indices
        if isinstance(indices, str):
            new_indices = string2index(indices)
        else:
            new_indices = indices
        npz_dictionary = {
            key: npz_object[key][new_indices] for key in npz_object.keys()
        }

        # close file
        npz_object.close()

        return cls(npz_dictionary=npz_dictionary, key_mapping=key_mapping)


def compute_atomic_displacement_vectors(
    trajectory_data: Union[List[ase.Atoms], Dict],
    time_step: Optional[int] = 1,
    time_between_frames: Optional[float] = 1.0,
    key_mapping: Optional[Dict[str, str]] = {},
):
    """Computes atomic displacments vectors for a trajectory in the format of a ase_atoms_list or a numpy array (not implement yet).

    Args:
        ase_atoms_list (List[ase.Atoms]): Trajectory as list of ase.Atoms.
        time_step (Union[int], optional): Time step to use when computing displacement vectors, corresponding to the frame frequency.
            Defaults to 1 corresponding to calculate displacement to the next frame.
        truncate (bool, optional): Whether to truncate the trajectory according the calculation.
            If time_step=1 the displacement is calculated for all frames culminating in N-1 frames and the last one is discarded. Defaults to True.

    Raises:
        ValueError: _description_
    """
    trajectory_type = (
        "ase"
        if isinstance(trajectory_data, list)
        and isinstance(trajectory_data[0], ase.Atoms)
        else "npz"
    )
    # get total number of frames
    n_frames = (
        len(trajectory_data)
        if trajectory_type == "ase"
        else len(trajectory_data.get(key_mapping[POSITIONS_KEY]))
    )

    # check that time_step is smaller than actual length of trajectory
    if time_step >= n_frames:
        raise ValueError(
            "Unphysical value for time_step which should be smaller than the number of frames"
        )

    # get positions for each frame required to compute displacements
    positions = (
        np.asarray([frame.positions for frame in trajectory_data])
        if trajectory_type == "ase"
        else trajectory_data[key_mapping[POSITIONS_KEY]]
    )
    # compute displacement
    displacement_vectors = np.asarray(
        [
            frame2 - frame1
            for frame1, frame2 in zip(positions[:-time_step], positions[time_step:])
        ]
    )

    # check for periodicity, so far only constant box
    lattice_vectors = (
        trajectory_data[0].cell
        if trajectory_type == "ase"
        else trajectory_data.get(key_mapping.get(CELL_KEY), [])
    )
    pbc = (
        trajectory_data[0].pbc
        if trajectory_type == "ase"
        else trajectory_data.get(key_mapping.get(PBC_KEY), [False, False, False])
    )
    if all(pbc):
        displacement_vectors = align_vectors_with_periodicity(
            displacement_vectors, lattice_vectors
        )
    # update
    # first for ASETrajectory
    if trajectory_type == "ase":
        [
            frame.arrays.__setitem__(DISPLACEMENTS_KEY, displacement)
            for frame, displacement in zip(
                trajectory_data[:-time_step], displacement_vectors
            )
        ]
        # update timestep
        [
            frame.info.__setitem__(TIMESTEP_KEY, time_between_frames * time_step)
            for frame in trajectory_data[:-time_step]
        ]

    else:
        trajectory_data[DISPLACEMENTS_KEY] = displacement_vectors
        trajectory_data[TIMESTEP_KEY] = np.array([time_between_frames * time_step])

    return trajectory_data


def get_desired_field_values_of_next_frame(
    fields: List,
    trajectory_data: Union[List[ase.Atoms], Dict],
    time_step: Optional[int] = 1,
    key_mapping: Optional[Dict[str, str]] = {},
):
    trajectory_type = (
        "ase"
        if isinstance(trajectory_data, list)
        and isinstance(trajectory_data[0], ase.Atoms)
        else "npz"
    )

    # get total number of frames
    n_frames = (
        len(trajectory_data)
        if trajectory_type == "ase"
        else len(trajectory_data.get(key_mapping[POSITIONS_KEY]))
    )

    # check that time_step is smaller than actual length of trajectory
    if time_step >= n_frames:
        raise ValueError(
            "Unphysical value for time_step which should be smaller than the number of frames"
        )

    # update
    # first for ASETrajectory
    if trajectory_type == "ase":
        # let's loop over the fields:
        for field in fields:
            new_field = f"{UPDATE_KEY}_{field}"
            # check whether the field is an array or info field
            if field in ASE_INFO_FIELDS:
                [
                    frame.info.__setitem__(
                        new_field,
                        trajectory_data[count + time_step].info[key_mapping[field]],
                    )
                    for count, frame in enumerate(trajectory_data[:-time_step])
                ]

            else:
                [
                    frame.arrays.__setitem__(
                        new_field,
                        trajectory_data[count + time_step].arrays[key_mapping[field]],
                    )
                    for count, frame in enumerate(trajectory_data[:-time_step])
                ]

    else:
        raise NotImplementedError("Not implemented for NPZ data yet.")

    return trajectory_data
