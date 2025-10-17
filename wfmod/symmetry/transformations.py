import numpy as np
import re


class Transformations:
    """store transformation of orbitals under symmetry operations in certain point groups."""
    @staticmethod
    def get_transformation_matrix(
         string_list: list[str],
         orb: str, mat: np.ndarray,
         orbital_basisfunctions: dict[str, list[str]]
    ):
        """Transforms string representation in transformation matrix"""
        mat = mat.copy()
        pattern = r"([+-])(\w+)"
        for op in string_list:
            func_val = op.split(" -> ")
            splitted = re.findall(
                pattern,
                func_val[1],
            )
            splitted = [
                (1 if sign == "+" else -1, word) for sign, word in splitted
            ]
            if splitted[0][1] == "0":
                return mat
            mat[
                orbital_basisfunctions[orb].index(func_val[0]),
                orbital_basisfunctions[orb].index(splitted[0][1]),
            ] = splitted[0][0]
        return mat

    @staticmethod
    def get_d2h_matrices() -> dict[str, object]:
        """load the opertion matrices for linear diatomics for
        s, px, py, pz orbitals."""

        orbital_basisfunctions = {
            "s": ["s1", "s2"],
            "p": ["px1", "py1", "pz1", "px2", "py2", "pz2"],
        }
        s_orbital_basis = [np.array([1, 0]), np.array([0, 1])]
        p_orbital_basis = [
            np.array([1, 0, 0, 0, 0, 0]),
            np.array([0, 1, 0, 0, 0, 0]),
            np.array([0, 0, 1, 0, 0, 0]),
            np.array([0, 0, 0, 1, 0, 0]),
            np.array([0, 0, 0, 0, 1, 0]),
            np.array([0, 0, 0, 0, 0, 1]),
        ]

        s_orbs = {
            "1 E": ["s1 -> +s1", "s2 -> +s2"],
            "1 C2_z": ["s1 -> +s1", "s2 -> +s2"],
            "1 C2_y": ["s1 -> +s2", "s2 -> +s1"],
            "1 C2_x": ["s1 -> +s2", "s2 -> +s1"],
            "1 i": ["s1 -> +s2", "s2 -> +s1"],
            "1 s_xy": ["s1 -> +s2", "s2 -> +s1"],
            "1 s_xz": ["s1 -> +s1", "s2 -> +s2"],
            "1 s_yz": ["s1 -> +s1", "s2 -> +s2"],
        }
        s_reducable_basis = [2, 2, 0, 0, 0, 0, 2, 2]
        orb_empty = np.zeros((len(s_orbital_basis), len(s_orbital_basis)))
        # convert to transformation matrix
        for mulliken, operations in s_orbs.items():
            s_orbs[mulliken] = Transformations.get_transformation_matrix(
                operations, "s", orb_empty, orbital_basisfunctions
            )

        px_orbs = {
            "1 E": ["px1 -> +px1", "px2 -> +px2"],
            "1 C2_z": ["px1 -> -px1", "px2 -> -px2"],
            "1 C2_y": ["px1 -> -px2", "px2 -> -px1"],
            "1 C2_x": ["px1 -> +px2", "px2 -> +px1"],
            "1 i": ["px1 -> -px2", "px2 -> -px1"],
            "1 s_xy": ["px1 -> +px2", "px2 -> +px1"],
            "1 s_xz": ["px1 -> +px1", "px2 -> +px2"],
            "1 s_yz": ["px1 -> -px1", "px2 -> -px2"],
        }
        orb_empty = np.zeros((len(p_orbital_basis), len(p_orbital_basis)))
        px_reducable_basis = [2, -2, 0, 0, 0, 0, 2, -2]
        # convert to transformation matrix
        for mulliken, operations in px_orbs.items():
            px_orbs[mulliken] = Transformations.get_transformation_matrix(
                operations, "p", orb_empty, orbital_basisfunctions
            )

        py_orbs = {
            "1 E": ["py1 -> +py1", "py2 -> +py2"],
            "1 C2_z": ["py1 -> -py1", "py2 -> -py2"],
            "1 C2_y": ["py1 -> +py2", "py2 -> +py1"],
            "1 C2_x": ["py1 -> -py2", "py2 -> -py1"],
            "1 i": ["py1 -> -py2", "py2 -> -py1"],
            "1 s_xy": ["py1 -> +py2", "py2 -> +py1"],
            "1 s_xz": ["py1 -> -py1", "py2 -> -py2"],
            "1 s_yz": ["py1 -> +py1", "py2 -> +py2"],
        }
        py_reducable_basis = [2, -2, 0, 0, 0, 0, -2, 2]
        # convert to transformation matrix
        for mulliken, operations in py_orbs.items():
            py_orbs[mulliken] = Transformations.get_transformation_matrix(
                operations, "p", orb_empty, orbital_basisfunctions
            )

        pz_orbs = {
            "1 E": ["pz1 -> +pz1", "pz2 -> +pz2"],
            "1 C2_z": ["pz1 -> +pz1", "pz2 -> +pz2"],
            "1 C2_y": ["pz1 -> -pz2", "pz2 -> -pz1"],
            "1 C2_x": ["pz1 -> -pz2", "pz2 -> -pz1"],
            "1 i": ["pz1 -> -pz2", "pz2 -> -pz1"],
            "1 s_xy": ["pz1 -> -pz2", "pz2 -> -pz1"],
            "1 s_xz": ["pz1 -> +pz1", "pz2 -> +pz2"],
            "1 s_yz": ["pz1 -> +pz1", "pz2 -> +pz2"],
        }
        pz_reducable_basis = [2, 2, 0, 0, 0, 0, 2, 2]
        # convert to transformation matrix
        for mulliken, operations in pz_orbs.items():
            pz_orbs[mulliken] = Transformations.get_transformation_matrix(
                operations, "p", orb_empty, orbital_basisfunctions
            )
        # load in variable

        operation_matrices = {
            "s": s_orbs,
            "px": px_orbs,
            "py": py_orbs,
            "pz": pz_orbs,
        }

        spanned_basis = {
            "s": s_reducable_basis,
            "px": px_reducable_basis,
            "py": py_reducable_basis,
            "pz": pz_reducable_basis,
        }

        orbital_basis = {
            "s": s_orbital_basis,
            "px": p_orbital_basis,
            "py": p_orbital_basis,
            "pz": p_orbital_basis,
        }
        return_dict = {
            "basis_functions": orbital_basisfunctions,
            "operation_matrices": operation_matrices,
            "spanned_basis": spanned_basis,
            "orbital_basis": orbital_basis,
        }
        return return_dict

    @staticmethod
    def get_d4h_matrices(cartesian: bool = False):
        """load the opertion matrices for linear diatomics for
        s, px, py, pz, dxy, dxz, dyz, dx2-y2, dz2 orbitals."""

        orbital_basisfunctions = {
            "s": ["s1", "s2"],
            "p": ["px1", "py1", "pz1", "px2", "py2", "pz2"],
            "pi": ["pi_x", "pi_y"],
            "d": ["dxy1", "dxz1", "dyz1", "dxxyy1", "dzz1", "dxy2", "dxz2", "dyz2", "dxxyy2", "dzz2"]
        }
        if cartesian:
            orbital_basisfunctions["d"] = [
                "dxy1", "dxz1", "dyz1", "dxx1", "dyy1", "dzz1", "dxy2", "dxz2", "dyz2", "dxx2", "dyy2", "dzz2",]
        s_orbital_basis = [np.array([1, 0]), np.array([0, 1])]
        p_orbital_basis = [
            np.array([1, 0, 0, 0, 0, 0]),
            np.array([0, 1, 0, 0, 0, 0]),
            np.array([0, 0, 1, 0, 0, 0]),
            np.array([0, 0, 0, 1, 0, 0]),
            np.array([0, 0, 0, 0, 1, 0]),
            np.array([0, 0, 0, 0, 0, 1]),
        ]
        pi_orbital_basis = [np.array([1, 0]), np.array([0, 1])]
        n_d_orbitals = len(orbital_basisfunctions["d"])
        d_orbital_basis = [
            np.eye(n_d_orbitals)[i] for i in range(n_d_orbitals)
        ]
        s_orbs = {
            "1 E": ["s1 -> +s1", "s2 -> +s2"],
            "1 C4_z+": ["s1 -> +s1", "s2 -> +s2"],
            "1 C4_z-": ["s1 -> +s1", "s2 -> +s2"],
            "1 C2": ["s1 -> +s1", "s2 -> +s2"],
            "1 C2''x": ["s1 -> +s2", "s2 -> +s1"],
            "1 C2''y": ["s1 -> +s2", "s2 -> +s1"],
            "1 C2'''1": ["s1 -> +s2", "s2 -> +s1"],
            "1 C2'''2": ["s1 -> +s2", "s2 -> +s1"],
            "1 i": ["s1 -> +s2", "s2 -> +s1"],
            "1 S4+": ["s1 -> +s2", "s2 -> +s1"],
            "1 S4-": ["s1 -> +s2", "s2 -> +s1"],
            "1 sh": ["s1 -> +s2", "s2 -> +s1"],
            "1 sv'": ["s1 -> +s1", "s2 -> +s2"],
            "1 sv''": ["s1 -> +s1", "s2 -> +s2"],
            "1 sd'": ["s1 -> +s1", "s2 -> +s2"],
            "1 sd''": ["s1 -> +s1", "s2 -> +s2"],
        }
        orb_empty = np.zeros((len(s_orbital_basis), len(s_orbital_basis)))
        s_reducable_basis = [2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2]
        # convert to transformation matrix
        for mulliken, operations in s_orbs.items():
            s_orbs[mulliken] = Transformations.get_transformation_matrix(
                operations, "s", orb_empty, orbital_basisfunctions
            )

        px_orbs = {
            "1 E": ["px1 -> +px1", "px2 -> +px2"],
            "1 C4_z+": ["px1 -> +py1", "px2 -> +py2"],
            "1 C4_z-": ["px1 -> -py1", "px2 -> -py2"],
            "1 C2": ["px1 -> -px1", "px2 -> -px2"],
            "1 C2''x": ["px1 -> +px2", "px2 -> +px1"],
            "1 C2''y": ["px1 -> -px2", "px2 -> -px1"],
            "1 C2'''1": ["px1 -> +py2", "px2 -> +py1"],
            "1 C2'''2": ["px1 -> -py2", "px2 -> -py1"],
            "1 i": ["px1 -> -px2", "px2 -> -px1"],
            "1 S4+": ["px1 -> +py2", "px2 -> +py1"],
            "1 S4-": ["px1 -> -py2", "px2 -> -py1"],
            "1 sh": ["px1 -> +px2", "px2 -> +px1"],
            "1 sv'": ["px1 -> -px1", "px2 -> -px2"],
            "1 sv''": ["px1 -> +px1", "px2 -> +px2"],
            "1 sd'": ["px1 -> +py1", "px2 -> +py2"],
            "1 sd''": ["px1 -> -py1", "px2 -> -py2"],
        }
        orb_empty = np.zeros((len(p_orbital_basis), len(p_orbital_basis)))
        px_reducable_basis = [2, 0, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, -2, -2, 2, 2]

        # convert to transformation matrix
        for mulliken, operations in px_orbs.items():
            px_orbs[mulliken] = Transformations.get_transformation_matrix(
                operations, "p", orb_empty, orbital_basisfunctions
            )

        py_orbs = {
            "1 E": ["py1 -> +py1", "py2 -> +py2"],
            "1 C4_z+": ["py1 -> +px1", "py2 -> +px2"],
            "1 C4_z-": ["py1 -> -px1", "py2 -> -px2"],
            "1 C2": ["py1 -> -py1", "py2 -> -py2"],
            "1 C2''x": ["py1 -> +py2", "py2 -> +py1"],
            "1 C2''y": ["py1 -> -py2", "py2 -> -py1"],
            "1 C2'''1": ["py1 -> +px2", "py2 -> +px1"],
            "1 C2'''2": ["py1 -> -px2", "py2 -> -px1"],
            "1 i": ["py1 -> -py2", "py2 -> -py1"],
            "1 S4+": ["py1 -> +px2", "py2 -> +px1"],
            "1 S4-": ["py1 -> -px2", "py2 -> -px1"],
            "1 sh": ["py1 -> +py2", "py2 -> +py1"],
            "1 sv'": ["py1 -> +py1", "py2 -> +py2"],
            "1 sv''": ["py1 -> -py1", "py2 -> -py2"],
            "1 sd'": ["py1 -> +px1", "py2 -> +px2"],
            "1 sd''": ["py1 -> -px1", "py2 -> -px2"],
        }
        py_reducable_basis = [2, 0, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, -2, -2]
        # convert to transformation matrix
        for mulliken, operations in py_orbs.items():
            py_orbs[mulliken] = Transformations.get_transformation_matrix(
                operations, "p", orb_empty, orbital_basisfunctions
            )

        pz_orbs = {
            "1 E": ["pz1 -> +pz1", "pz2 -> +pz2"],
            "1 C4_z+": ["pz1 -> +pz1", "pz2 -> +pz2"],
            "1 C4_z-": ["pz1 -> +pz1", "pz2 -> +pz2"],
            "1 C2": ["pz1 -> +pz1", "pz2 -> +pz2"],
            "1 C2''x": ["pz1 -> -pz2", "pz2 -> -pz1"],
            "1 C2''y": ["pz1 -> -pz2", "pz2 -> -pz1"],
            "1 C2'''1": ["pz1 -> -pz2", "pz2 -> -pz1"],
            "1 C2'''2": ["pz1 -> -pz2", "pz2 -> -pz1"],
            "1 i": ["pz1 -> -pz2", "pz2 -> -pz1"],
            "1 S4+": ["pz1 -> -pz2", "pz2 -> -pz1"],
            "1 S4-": ["pz1 -> -pz2", "pz2 -> -pz1"],
            "1 sh": ["pz1 -> -pz2", "pz2 -> -pz1"],
            "1 sv'": ["pz1 -> +pz1", "pz2 -> +pz2"],
            "1 sv''": ["pz1 -> +pz1", "pz2 -> +pz2"],
            "1 sd'": ["pz1 -> +pz1", "pz2 -> +pz2"],
            "1 sd''": ["pz1 -> +pz1", "pz2 -> +pz2"],
        }
        pz_reducable_basis = [2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2]
        # convert to transformation matrix
        for mulliken, operations in pz_orbs.items():
            pz_orbs[mulliken] = Transformations.get_transformation_matrix(
                operations, "p", orb_empty, orbital_basisfunctions
            )

        pi_x__u_orbs = {
            "1 E": ["pi_x -> +pi_x"],
            "1 C4_z+": ["pi_x -> +pi_y"],
            "1 C4_z-": ["pi_x -> -pi_y"],
            "1 C2": ["pi_x -> -pi_x"],
            "1 C2''x": ["pi_x -> +pi_x"],
            "1 C2''y": ["pi_x -> -pi_x"],
            "1 C2'''1": ["pi_x -> +pi_y"],
            "1 C2'''2": ["pi_x -> -pi_y"],
            "1 i": ["pi_x -> -pi_x"],
            "1 S4+": ["pi_x -> -pi_y"],
            "1 S4-": ["pi_x -> +pi_y"],
            "1 sh": ["pi_x -> +pi_x"],
            "1 sv'": ["pi_x -> +pi_x"],
            "1 sv''": ["pi_x -> -pi_x"],
            "1 sd'": ["pi_x -> +pi_y"],
            "1 sd''": ["pi_x -> -pi_y"],
        }
        orb_empty = np.zeros((len(pi_orbital_basis), len(pi_orbital_basis)))
        pi_x_u_reducable_basis = [1, 0, 0, -1, +1, -1, 0, 0, -1, 0, 0, 1, +1, -1, 0, 0]
        # convert to transformation matrix
        for mulliken, operations in pi_x__u_orbs.items():
            pi_x__u_orbs[mulliken] = Transformations.get_transformation_matrix(
                operations, "pi", orb_empty, orbital_basisfunctions
            )

        pi_y__u_orbs = {
            "1 E": ["pi_y -> +pi_y"],
            "1 C4_z+": ["pi_y -> -pi_x"],
            "1 C4_z-": ["pi_y -> +pi_x"],
            "1 C2": ["pi_y -> -pi_y"],
            "1 C2''x": ["pi_y -> -pi_y"],
            "1 C2''y": ["pi_y -> +pi_y"],
            "1 C2'''1": ["pi_y -> +pi_x"],
            "1 C2'''2": ["pi_y -> -pi_x"],
            "1 i": ["pi_y -> -pi_y"],
            "1 S4+": ["pi_y -> +pi_x"],
            "1 S4-": ["pi_y -> -pi_x"],
            "1 sh": ["pi_y -> +pi_y"],
            "1 sv'": ["pi_y -> -pi_y"],
            "1 sv''": ["pi_y -> +pi_y"],
            "1 sd'": ["pi_y -> +pi_x"],
            "1 sd''": ["pi_y -> -pi_x"],
        }
        pi_y_u_reducable_basis = [1, 0, 0, -1, -1, 1, 0, 0, -1, 0, 0, 1, -1, 1, 0, 0]
        # convert to transformation matrix
        for mulliken, operations in pi_y__u_orbs.items():
            pi_y__u_orbs[mulliken] = Transformations.get_transformation_matrix(
                operations, "pi", orb_empty, orbital_basisfunctions
            )

        pi_x_g_orbs = {
            "1 E": ["pi_x -> +pi_x"],
            "1 C4_z+": ["pi_x -> +pi_y"],
            "1 C4_z-": ["pi_x -> -pi_y"],
            "1 C2": ["pi_x -> -pi_x"],
            "1 C2''x": ["pi_x -> +pi_x"],
            "1 C2''y": ["pi_x -> -pi_x"],
            "1 C2'''1": ["pi_x -> +pi_y"],
            "1 C2'''2": ["pi_x -> -pi_y"],
            "1 i": ["pi_x -> +pi_x"],
            "1 S4+": ["pi_x -> +pi_y"],
            "1 S4-": ["pi_x -> -pi_y"],
            "1 sh": ["pi_x -> -pi_x"],
            "1 sv'": ["pi_x -> +pi_x"],
            "1 sv''": ["pi_x -> -pi_x"],
            "1 sd'": ["pi_x -> +pi_y"],
            "1 sd''": ["pi_x -> -pi_y"],
        }
        pi_x_g_reducable_basis = [1, 0, 0, -1, 1, -1, 0, 0, +1, 0, 0, -1, +1, -1, 0, 0]
        # convert to transformation matrix
        for mulliken, operations in pi_x_g_orbs.items():
            pi_x_g_orbs[mulliken] = Transformations.get_transformation_matrix(
                operations, "pi", orb_empty, orbital_basisfunctions
            )

        pi_y_g_orbs = {
            "1 E": ["pi_y -> +pi_y"],
            "1 C4_z+": ["pi_y -> -pi_x"],
            "1 C4_z-": ["pi_y -> +pi_x"],
            "1 C2": ["pi_y -> -pi_y"],
            "1 C2''x": ["pi_y -> -pi_y"],
            "1 C2''y": ["pi_y -> +pi_y"],
            "1 C2'''1": ["pi_y -> +pi_x"],
            "1 C2'''2": ["pi_y -> -pi_x"],
            "1 i": ["pi_y -> +pi_y"],
            "1 S4+": ["pi_y -> -pi_x"],
            "1 S4-": ["pi_y -> +pi_x"],
            "1 sh": ["pi_y -> -pi_y"],
            "1 sv'": ["pi_y -> -pi_y"],
            "1 sv''": ["pi_y -> +pi_y"],
            "1 sd'": ["pi_y -> +pi_x"],
            "1 sd''": ["pi_y -> -pi_x"],
        }
        pi_y_g_reducable_basis = [1, 0, 0, -1, -1, 1, 0, 0, +1, 0, 0, -1, -1, 1, 0, 0]
        # convert to transformation matrix
        for mulliken, operations in pi_y_g_orbs.items():
            pi_y_g_orbs[mulliken] = Transformations.get_transformation_matrix(
                operations, "pi", orb_empty, orbital_basisfunctions
            )

        dxy_orbs = {
            "1 E": ["dxy1 -> +dxy1", "dxy2 -> +dxy2"],
            "1 C4_z+": ["dxy1 -> -dxy1", "dxy2 -> -dxy2"],
            "1 C4_z-": ["dxy1 -> -dxy1", "dxy2 -> -dxy2"],
            "1 C2": ["dxy1 -> +dxy1", "dxy2 -> +dxy2"],
            "1 C2''x": ["dxy1 -> -dxy2", "dxy2 -> -dxy1"],
            "1 C2''y": ["dxy1 -> -dxy2", "dxy2 -> -dxy1"],
            "1 C2'''1": ["dxy1 -> +dxy2", "dxy2 -> +dxy1"],
            "1 C2'''2": ["dxy1 -> +dxy2", "dxy2 -> +dxy1"],
            "1 i": ["dxy1 -> +dxy2", "dxy2 -> +dxy1"],
            "1 S4+": ["dxy1 -> -dxy2", "dxy2 -> -dxy1"],
            "1 S4-": ["dxy1 -> -dxy2", "dxy2 -> -dxy1"],
            "1 sh": ["dxy1 -> +dxy2", "dxy2 -> +dxy1"],
            "1 sv'": ["dxy1 -> -dxy1", "dxy2 -> -dxy2"],
            "1 sv''": ["dxy1 -> -dxy1", "dxy2 -> -dxy2"],
            "1 sd'": ["dxy1 -> +dxy1", "dxy2 -> +dxy2"],
            "1 sd''": ["dxy1 -> +dxy1", "dxy2 -> +dxy2"],
        }
        orb_empty = np.zeros((len(d_orbital_basis), len(d_orbital_basis)))
        dxy_reducable_basis = [2, -2, -2, 2, 0, 0, 0, 0, 0, 0, 0, 0, -2, -2, +2, +2]

        dxz_orbs = {
            "1 E": ["dxz1 -> +dxz1", "dxz2 -> +dxz2"],
            "1 C4_z+": ["dxz1 -> +dyz1", "dxz2 -> +dyz2"],
            "1 C4_z-": ["dxz1 -> -dyz1", "dxz2 -> -dyz2"],
            "1 C2": ["dxz1 -> -dxz1", "dxz2 -> -dxz2"],
            "1 C2''x": ["dxz1 -> -dxz2", "dxz2 -> -dxz1"],
            "1 C2''y": ["dxz1 -> +dxz2", "dxz2 -> +dxz1"],
            "1 C2'''1": ["dxz1 -> +dyz2", "dxz2 -> +dyz1"],
            "1 C2'''2": ["dxz1 -> -dyz2", "dxz2 -> -dyz1"],
            "1 i": ["dxz1 -> +dxz2", "dxz2 -> +dxz1"],
            "1 S4+": ["dxz1 -> +dyz2", "dxz2 -> +dyz1"],
            "1 S4-": ["dxz1 -> -dyz2", "dxz2 -> -dyz1"],
            "1 sh": ["dxz1 -> -dxz2", "dxz2 -> -dxz1"],
            "1 sv'": ["dxz1 -> -dxz1", "dxz2 -> -dxz2"],
            "1 sv''": ["dxz1 -> +dxz1", "dxz2 -> +dxz2"],
            "1 sd'": ["dxz1 -> +dyz1", "dxz2 -> +dyz2"],
            "1 sd''": ["dxz1 -> -dyz1", "dxz2 -> -dyz2"],
        }
        dxz_reducable_basis = [2, 0, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, +2, -2]

        dyz_orbs = {
            "1 E": ["dyz1 -> +dyz1", "dyz2 -> +dyz2"],
            "1 C4_z+": ["dyz1 -> +dxz1", "dyz2 -> +dxz2"],
            "1 C4_z-": ["dyz1 -> -dxz1", "dyz2 -> -dxz2"],
            "1 C2": ["dyz1 -> -dyz1", "dyz2 -> -dyz2"],
            "1 C2''x": ["dyz1 -> -dyz2", "dyz2 -> -dyz1"],
            "1 C2''y": ["dyz1 -> +dyz2", "dyz2 -> +dyz1"],
            "1 C2'''1": ["dyz1 -> +dxz2", "dyz2 -> +dxz1"],
            "1 C2'''2": ["dyz1 -> -dxz2", "dyz2 -> -dxz1"],
            "1 i": ["dyz1 -> +dyz2", "dyz2 -> +dyz1"],
            "1 S4+": ["dyz1 -> -dxz2", "dyz2 -> -dxz1"],
            "1 S4-": ["dyz1 -> +dxz2", "dyz2 -> +dxz1"],
            "1 sh": ["dyz1 -> -dyz2", "dyz2 -> -dyz1"],
            "1 sv'": ["dyz1 -> +dyz1", "dyz2 -> +dyz1"],
            "1 sv''": ["dyz1 -> -dyz1", "dyz2 -> -dyz1"],
            "1 sd'": ["dyz1 -> +dxz1", "dyz2 -> +dxz2"],
            "1 sd''": ["dyz1 -> -dxz1", "dyz2 -> -dxz2"],
        }
        dyz_reducable_basis = [2, 0, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, -2, -2, 2, 2]

        dxxyy_orbs = {
            "1 E": ["dxxyy1 -> +dxxyy1", "dxxyy2 -> +dxxyy2"],
            "1 C4_z+": ["dxxyy1 -> -dxxyy1", "dxxyy2 -> -dxxyy2"],
            "1 C4_z-": ["dxxyy1 -> -dxxyy1", "dxxyy2 -> -dxxyy2"],
            "1 C2": ["dxxyy1 -> +dxxyy1", "dxxyy2 -> +dxxyy2"],
            "1 C2''x": ["dxxyy1 -> +dxxyy2", "dxxyy2 -> +dxxyy1"],
            "1 C2''y": ["dxxyy1 -> +dxxyy2", "dxxyy2 -> +dxxyy1"],
            "1 C2'''1": ["dxxyy1 -> -dxxyy2", "dxxyy2 -> -dxxyy1"],
            "1 C2'''2": ["dxxyy1 -> -dxxyy2", "dxxyy2 -> -dxxyy1"],
            "1 i": ["dxxyy1 -> +dxxyy2", "dxxyy2 -> +dxxyy1"],
            "1 S4+": ["dxxyy1 -> -dxxyy2", "dxxyy2 -> -dxxyy1"],
            "1 S4-": ["dxxyy1 -> -dxxyy2", "dxxyy2 -> -dxxyy1"],
            "1 sh": ["dxxyy1 -> +dxxyy2", "dxxyy2 -> +dxxyy1"],
            "1 sv'": ["dxxyy1 -> +dxxyy1", "dxxyy2 -> +dxxyy2"],
            "1 sv''": ["dxxyy1 -> +dxxyy1", "dxxyy2 -> +dxxyy2"],
            "1 sd'": ["dxxyy1 -> -dxxyy1", "dxxyy2 -> -dxxyy2"],
            "1 sd''": ["dxxyy1 -> -dxxyy1", "dxxyy2 -> -dxxyy2"],
        }
        dxxyy_reducable_basis = [2, -2, -2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, -2, -2]

        dzz_orbs = {
            "1 E": ["dzz1 -> +dzz1", "dzz2 -> +dzz2"],
            "1 C4_z+": ["dzz1 -> +dzz1", "dzz2 -> +dzz2"],
            "1 C4_z-": ["dzz1 -> +dzz1", "dzz2 -> +dzz2"],
            "1 C2": ["dzz1 -> +dzz1", "dzz2 -> +dzz2"],
            "1 C2''x": ["dzz1 -> +dzz2", "dzz2 -> +dzz1"],
            "1 C2''y": ["dzz1 -> +dzz2", "dzz2 -> +dzz1"],
            "1 C2'''1": ["dzz1 -> +dzz2", "dzz2 -> +dzz1"],
            "1 C2'''2": ["dzz1 -> +dzz2", "dzz2 -> +dzz1"],
            "1 i": ["dzz1 -> +dzz2", "dzz2 -> +dzz1"],
            "1 S4+": ["dzz1 -> +dzz2", "dzz2 -> +dzz1"],
            "1 S4-": ["dzz1 -> +dzz2", "dzz2 -> +dzz1"],
            "1 sh": ["dzz1 -> +dzz2", "dzz2 -> +dzz1"],
            "1 sv'": ["dzz1 -> +dzz1", "dzz2 -> +dzz2"],
            "1 sv''": ["dzz1 -> +dzz1", "dzz2 -> +dzz2"],
            "1 sd'": ["dzz1 -> +dzz1", "dzz2 -> +dzz2"],
            "1 sd''": ["dzz1 -> +dzz1", "dzz2 -> +dzz2"],
        }
        dzz_reducable_basis = [2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2]

        dxx_orbs = {
            "1 E": ["dxx1 -> +dxx1", "dxx2 -> +dxx2"],
            "1 C4_z+": ["dxx1 -> +dyy1", "dxx2 -> +dyy2"],
            "1 C4_z-": ["dxx1 -> +dyy1", "dxx2 -> +dyy2"],
            "1 C2": ["dxx1 -> +dxx1", "dxx2 -> +dxx2"],
            "1 C2''x": ["dxx1 -> +dxx2", "dxx2 -> +dxx1"],
            "1 C2''y": ["dxx1 -> +dxx2", "dxx2 -> +dxx1"],
            "1 C2'''1": ["dxx1 -> +dyy2", "dxx2 -> +dyy1"],
            "1 C2'''2": ["dxx1 -> +dyy2", "dxx2 -> +dyy1"],
            "1 i": ["dxx1 -> +dxx2", "dxx2 -> +dxx1"],
            "1 S4+": ["dxx1 -> +dyy2", "dxx2 -> +dyy1"],
            "1 S4-": ["dxx1 -> +dyy2", "dxx2 -> +dyy1"],
            "1 sh": ["dxx1 -> +dxx2", "dxx2 -> +dxx1"],
            "1 sv'": ["dxx1 -> +dxx1", "dxx2 -> +dxx2"],
            "1 sv''": ["dxx1 -> +dxx1", "dxx2 -> +dxx2"],
            "1 sd'": ["dxx1 -> +dyy1", "dxx2 -> +dyy2"],
            "1 sd''": ["dxx1 -> +dyy1", "dxx2 -> +dyy2"],
        }
        dxx_reducable_basis = [2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0,]

        dyy_orbs = {
            "1 E": ["dyy1 -> +dyy1", "dyy2 -> +dyy2"],
            "1 C4_z+": ["dyy1 -> +dxx1", "dyy2 -> +dxx2"],
            "1 C4_z-": ["dyy1 -> +dxx1", "dyy2 -> +dxx2"],
            "1 C2": ["dyy1 -> +dyy1", "dyy2 -> +dyy2"],
            "1 C2''x": ["dyy1 -> +dyy2", "dyy2 -> +dyy1"],
            "1 C2''y": ["dyy1 -> +dyy2", "dyy2 -> +dyy1"],
            "1 C2'''1": ["dyy1 -> +dxx2", "dyy2 -> +dxx1"],
            "1 C2'''2": ["dyy1 -> +dxx2", "dyy2 -> +dxx1"],
            "1 i": ["dyy1 -> +dyy2", "dyy2 -> +dyy1"],
            "1 S4+": ["dyy1 -> +dxx2", "dyy2 -> +dxx1"],
            "1 S4-": ["dyy1 -> +dxx2", "dyy2 -> +dxx1"],
            "1 sh": ["dyy1 -> +dyy2", "dyy2 -> +dyy1"],
            "1 sv'": ["dyy1 -> +dyy1", "dyy2 -> +dyy2"],
            "1 sv''": ["dyy1 -> +dyy1", "dyy2 -> +dyy2"],
            "1 sd'": ["dyy1 -> +dxx1", "dyy2 -> +dxx2"],
            "1 sd''": ["dyy1 -> +dxx1", "dyy2 -> +dxx2"],
        }
        dyy_reducable_basis = [2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0,]

        dzz_cart_orbs = {
            "1 E": ["dzz1 -> +dzz1", "dzz2 -> +dzz2"],
            "1 C4_z+": ["dzz1 -> +dzz1", "dzz2 -> +dzz2"],
            "1 C4_z-": ["dzz1 -> +dzz1", "dzz2 -> +dzz2"],
            "1 C2": ["dzz1 -> +dzz1", "dzz2 -> +dzz2"],
            "1 C2''x": ["dzz1 -> +dzz2", "dzz2 -> +dzz1"],
            "1 C2''y": ["dzz1 -> +dzz2", "dzz2 -> +dzz1"],
            "1 C2'''1": ["dzz1 -> +dzz2", "dzz2 -> +dzz1"],
            "1 C2'''2": ["dzz1 -> +dzz2", "dzz2 -> +dzz1"],
            "1 i": ["dzz1 -> +dzz2", "dzz2 -> +dzz1"],
            "1 S4+": ["dzz1 -> +dzz2", "dzz2 -> +dzz1"],
            "1 S4-": ["dzz1 -> +dzz2", "dzz2 -> +dzz1"],
            "1 sh": ["dzz1 -> +dzz2", "dzz2 -> +dzz1"],
            "1 sv'": ["dzz1 -> +dzz1", "dzz2 -> +dzz2"],
            "1 sv''": ["dzz1 -> +dzz1", "dzz2 -> +dzz2"],
            "1 sd'": ["dzz1 -> +dzz1", "dzz2 -> +dzz2"],
            "1 sd''": ["dzz1 -> +dzz1", "dzz2 -> +dzz2"],
        }
        dzz_cart_reducable_basis = [2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2,]
        cart_bas = [
            dyz_orbs,
            dxz_orbs,
            dxy_orbs,
            dxx_orbs,
            dyy_orbs,
            dzz_cart_orbs,
        ]
        sph_bas = [dyz_orbs, dxz_orbs, dxy_orbs, dxxyy_orbs, dzz_orbs]
        if cartesian:
            for orbs in cart_bas:
                for mulliken, operations in orbs.items():
                    orbs[mulliken] = Transformations.get_transformation_matrix(
                        operations, "d", orb_empty, orbital_basisfunctions
                    )
        else:
            for orbs in sph_bas:
                for mulliken, operations in orbs.items():
                    orbs[mulliken] = Transformations.get_transformation_matrix(
                        operations, "d", orb_empty, orbital_basisfunctions
                    )

        operation_matrices = {
            "s": s_orbs,
            "px": px_orbs,
            "py": py_orbs,
            "pz": pz_orbs,
            "pi_x_u": pi_x__u_orbs,
            "pi_y_u": pi_y__u_orbs,
            "pi_x_g": pi_x_g_orbs,
            "pi_y_g": pi_y_g_orbs,
            "dxy": dxy_orbs,
            "dxz": dxz_orbs,
            "dyz": dyz_orbs,
        }
        if cartesian:
            operation_matrices["dxx"] = dxx_orbs
            operation_matrices["dyy"] = dyy_orbs
            operation_matrices["dzz"] = dzz_cart_orbs
        else:
            operation_matrices["dxxyy"] = dxxyy_orbs
            operation_matrices["dzz"] = dzz_orbs

        spanned_basis = {
            "s": s_reducable_basis,
            "px": px_reducable_basis,
            "py": py_reducable_basis,
            "pz": pz_reducable_basis,
            "pi_x_u": pi_x_u_reducable_basis,
            "pi_y_u": pi_y_u_reducable_basis,
            "pi_x_g": pi_x_g_reducable_basis,
            "pi_y_g": pi_y_g_reducable_basis,
            "dxy": dxy_reducable_basis,
            "dxz": dxz_reducable_basis,
            "dyz": dyz_reducable_basis,
        }
        if cartesian:
            spanned_basis["dxx"] = dxx_reducable_basis
            spanned_basis["dyy"] = dyy_reducable_basis
            spanned_basis["dzz"] = dzz_cart_reducable_basis
        else:
            spanned_basis["dxxyy"] = dxxyy_reducable_basis
            spanned_basis["dzz"] = dzz_reducable_basis

        orbital_basis = {
            "s": s_orbital_basis,
            "px": p_orbital_basis,
            "py": p_orbital_basis,
            "pz": p_orbital_basis,
            "pi_x": pi_orbital_basis,
            "pi_y": pi_orbital_basis,
            "pi": pi_orbital_basis,
            "dxy": d_orbital_basis,
            "dxz": d_orbital_basis,
            "dyz": d_orbital_basis,
        }
        if cartesian:
            orbital_basis["dxx"] = d_orbital_basis
            orbital_basis["dyy"] = d_orbital_basis
            orbital_basis["dzz"] = d_orbital_basis
        else:
            orbital_basis["dxxyy"] = d_orbital_basis
            orbital_basis["dzz"] = d_orbital_basis

        return_dict = {
            "basis_functions": orbital_basisfunctions,
            "operation_matrices": operation_matrices,
            "spanned_basis": spanned_basis,
            "orbital_basis": orbital_basis,
        }
        return return_dict
