import unittest

from trajcast.data.wrappers._lammps import lammps_dump_to_ase_atoms


class TestLammpsDumpToASEAtoms(unittest.TestCase):
    def test_raises_error_if_file_does_not_exist(self):
        with self.assertRaises(FileNotFoundError):
            lammps_dump_to_ase_atoms(path_to_file="fake_path")

    def test_returns_list_of_ase_atoms(self):
        path = "tests/unit/data/data/benzene_lammps_short.xyz"
        list_ase_atoms = lammps_dump_to_ase_atoms(
            path_to_file=path, lammps_units="real", type_mapping={1: 6, 2: 1}
        )

        self.assertTrue(
            set(list_ase_atoms[0].arrays.keys()).issuperset(
                {"positions", "velocities", "forces"}
            )
        )
        self.assertTrue("momenta" not in list_ase_atoms[0].arrays.keys())


if __name__ == "__main__":
    unittest.main()
