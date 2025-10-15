import numpy as np
import re
import yaml
from wfmod.charactertables import CharacterTable
from wfmod.symmetry.transformations import Transformations


class SALC:
    def __init__(self, point_group: str, basis: list, cartesian: bool = False):
        """"""
        self.point_group = point_group
        self.characTab = CharacterTable(point_group)
        self.mo_basis = basis
        self.cartesian = cartesian
        self.operation_matrices: dict[str, list] = {}
        self.spanned_basis: dict[str, list] = {}
        self.orbital_basis: dict[str, list] = {}
        self.orbital_basisfunctions: dict[str, list] = {}
        self.proj_results: dict[str, dict[str, list]] = {}
        self.get_operations()

    def get_operations(self):
        """"""
        # get for each operation the corresponding matrix. load for now the
        # matrices for d2h, d4h or d8h from memory
        # TODO get by symmetry tools
        if self.point_group == "d2h":
            self.load_d2h_matrices()
        elif self.point_group == "d4h_expanded":
            self.load_d4h_matrices()

    def load_d2h_matrices(self):
        """load the opertion matrices for linear diatomics for
        s, px, py, pz orbitals."""
        transformations = Transformations.get_d2h_matrices()

        self.operation_matrices["s"] = transformations["operation_matrices"]["s"]
        self.operation_matrices["px"] = transformations["operation_matrices"]["px"]
        self.operation_matrices["py"] = transformations["operation_matrices"]["py"]
        self.operation_matrices["pz"] = transformations["operation_matrices"]["pz"]
        self.spanned_basis["s"] = transformations["spanned_basis"]["s"]
        self.spanned_basis["px"] = transformations["spanned_basis"]["px"]
        self.spanned_basis["py"] = transformations["spanned_basis"]["py"]
        self.spanned_basis["pz"] = transformations["spanned_basis"]["pz"]
        self.orbital_basis["s"] = transformations["orbital_basis"]["s"]
        self.orbital_basis["p"] = transformations["orbital_basis"]["px"] # add basis also for general p
        self.orbital_basis["px"] = transformations["orbital_basis"]["px"]
        self.orbital_basis["py"] = transformations["orbital_basis"]["py"]
        self.orbital_basis["pz"] = transformations["orbital_basis"]["pz"]
        self.orbital_basisfunctions = transformations["basis_functions"]

    def load_d4h_matrices(self):
        """load the opertion matrices for linear diatomics for
        s, px, py, pz, dxy, dxz, dyz, dx2-y2, dz2 orbitals."""
        transformations = Transformations.get_d4h_matrices(self.cartesian)

        self.operation_matrices["s"] = transformations["operation_matrices"]["s"]
        self.operation_matrices["px"] = transformations["operation_matrices"]["px"]
        self.operation_matrices["py"] = transformations["operation_matrices"]["py"]
        self.operation_matrices["pz"] = transformations["operation_matrices"]["pz"]
        self.operation_matrices["pi_x_u"] = transformations["operation_matrices"]["pi_x_u"]
        self.operation_matrices["pi_y_u"] = transformations["operation_matrices"]["pi_y_u"]
        self.operation_matrices["pi_x_g"] = transformations["operation_matrices"]["pi_x_g"]
        self.operation_matrices["pi_y_g"] = transformations["operation_matrices"]["pi_y_g"]
        self.operation_matrices["dxy"] = transformations["operation_matrices"]["dxy"]
        self.operation_matrices["dxz"] = transformations["operation_matrices"]["dxz"]
        self.operation_matrices["dyz"] = transformations["operation_matrices"]["dyz"]
        if self.cartesian:
            self.operation_matrices["dxx"] = transformations["operation_matrices"]["dxx"]
            self.operation_matrices["dyy"] = transformations["operation_matrices"]["dyy"]
            self.operation_matrices["dzz"] = transformations["operation_matrices"]["dzz"]
        else:
            self.operation_matrices["dxxyy"] = transformations["operation_matrices"]["dxxyy"]
            self.operation_matrices["dzz"] = transformations["operation_matrices"]["dzz"]
        self.spanned_basis["s"] = transformations["spanned_basis"]["s"]
        self.spanned_basis["px"] = transformations["spanned_basis"]["px"]
        self.spanned_basis["py"] = transformations["spanned_basis"]["py"]
        self.spanned_basis["pz"] = transformations["spanned_basis"]["pz"]
        self.spanned_basis["pi_x_u"] = transformations["spanned_basis"]["pi_x_u"]
        self.spanned_basis["pi_y_u"] = transformations["spanned_basis"]["pi_y_u"]
        self.spanned_basis["pi_x_g"] = transformations["spanned_basis"]["pi_x_g"]
        self.spanned_basis["pi_y_g"] = transformations["spanned_basis"]["pi_y_g"]
        self.spanned_basis["dxy"] = transformations["spanned_basis"]["dxy"]
        self.spanned_basis["dxz"] = transformations["spanned_basis"]["dxz"]
        self.spanned_basis["dyz"] = transformations["spanned_basis"]["dyz"]
        if self.cartesian:
            self.spanned_basis["dxx"] = transformations["spanned_basis"]["dxx"]
            self.spanned_basis["dyy"] = transformations["spanned_basis"]["dyy"]
            self.spanned_basis["dzz"] = transformations["spanned_basis"]["dzz"]
        else:
            self.spanned_basis["dxxyy"] = transformations["spanned_basis"]["dxxyy"]
            self.spanned_basis["dzz"] = transformations["spanned_basis"]["dzz"]
        self.orbital_basis["s"] = transformations["orbital_basis"]["s"]
        self.orbital_basis["px"] = transformations["orbital_basis"]["px"]
        self.orbital_basis["py"] = transformations["orbital_basis"]["py"]
        self.orbital_basis["pz"] = transformations["orbital_basis"]["pz"]
        self.orbital_basis["p"] = transformations["orbital_basis"]["px"] # add basis also for general p
        self.orbital_basis["pi_x"] = transformations["orbital_basis"]["pi_x"]
        self.orbital_basis["pi_y"] = transformations["orbital_basis"]["pi_y"]
        self.orbital_basis["pi"] = transformations["orbital_basis"]["pi"]
        self.orbital_basis["dxy"] = transformations["orbital_basis"]["dxy"]
        self.orbital_basis["dxz"] = transformations["orbital_basis"]["dxz"]
        self.orbital_basis["dyz"] = transformations["orbital_basis"]["dyz"]
        if self.cartesian:
            self.orbital_basis["dxx"] = transformations["orbital_basis"]["dxx"]
            self.orbital_basis["dyy"] = transformations["orbital_basis"]["dyy"]
            self.orbital_basis["dzz"] = transformations["orbital_basis"]["dzz"]
        else:
            self.orbital_basis["dxxyy"] = transformations["orbital_basis"]["dxxyy"]
            self.orbital_basis["dzz"] = transformations["orbital_basis"]["dzz"]
        self.orbital_basisfunctions = transformations["basis_functions"]


    def get_symmetry_adapted_basis(self, orbital):
        """
        Get symmetry adapted basis for the given orbitals by computing
        irreps that contribute to the reducible representation. The projection
        operator is applied for each contributing irrep to get the basis
        corresponding to each contributing irrep.

        The function returns the Mulliken label of the contributing irrep and
        the corresponding symmetry adapted basis as list of numpy arrays.
        """

        # get irreps by reduction from reducible representation of the spanned
        # orbital basis
        contributions, mulliken_labels = self.characTab.get_reduction(
            self.spanned_basis[orbital]
        )
        orb_basis = self.orbital_basis[orbital]
        order = self.characTab.order

        mulliken_label_res = []
        projection_res = []

        # get symmetry adapted basis by applying the projection operator for
        # each contributing irrep
        # (eq. 5.24 Atkins Friedman 2011. Ed. 5; Example 5.9)

        for contribution, label in zip(contributions, mulliken_labels):
            if contribution != 0:
                dim = self.characTab.get_dimension(label)
                counter = 0
                tmp = [np.zeros(len(orb_basis)) for _ in orb_basis]
                for operation, matrix in self.operation_matrices[
                    orbital
                ].items():
                    res = (
                        np.dot(matrix, orb_basis)
                        * self.characTab.characters[label][counter]
                        * int(operation.split()[0])
                    )
                    tmp += res
                    counter += 1

                projection = dim / order * tmp
                if not np.all(projection == 0):
                    mulliken_label_res.append(label)
                    projection_res.append(projection)
        return mulliken_label_res, projection_res

    def apply_symmetry_operator_on_product(self, prod: list, inversion: str = ""):
        """
        Returns the product of basis functions for each irrep after applying
        the symmetry operations.
        """

        for fac in prod:
            print("fac:", fac)
            spherical_harmonic, _ = self.get_spherical_harmonic(fac)
            # get l type of the orbitals
            angular_momentum = ""
            for l_type in self.orbital_basisfunctions.keys():
                if l_type in fac:
                    angular_momentum = l_type
                    break
            assert angular_momentum, f"Could not find angular momentum (s, p, d ...)type for {fac}"
            print("angular momentum:", angular_momentum)

            # get index of respective basis_function
            print("basis functions:", self.orbital_basisfunctions)
            index = self.orbital_basisfunctions[angular_momentum].index(fac)
            print("index in angular momentum list:", index)

            # get representation of orbital input in basis function
            basis_function = self.orbital_basis[angular_momentum][index]
            print("basis function:", basis_function)

            # now apply projection operator on this basis function
            for i, operation_symbol in enumerate(self.characTab.operations):
                print(operation_symbol)
                dot = np.dot(
                    self.operation_matrices[spherical_harmonic][operation_symbol].T,
                    basis_function,
                ) * int(operation_symbol.split()[0])
                print(dot)
            print(self.orbital_basisfunctions)
            self.get_ao_name(basis_function, angular_momentum)
            return


        operation_results = []
        for i, operation_symbol in enumerate(self.characTab.operations):
            print(operation_symbol)
            print("HERE")
            print(self.operation_matrices[fac][operation_symbol])
            print(self.orbital_basis[fac])

            return
            tmp_res = []
            for func in prod:
                # get index of respective basis_function
                index = next(
                    j
                    for j, arr in enumerate(basis_functions)
                    if np.array_equal(arr, func)
                )
                dot = np.dot(
                    self.operation_matrices[lab][operation_symbol],
                    basis_functions,
                ) * int(operation_symbol.split()[0])
                # do not append the zero vector
                for arr in dot:
                    if np.any(arr):
                        tmp_res.append(arr)
            operation_results.append(tmp_res)
        return operation_results

    def apply_symmetry_operator_on_product_pi(self, prod: list, inversion: str = ""):
        """
        Returns the product of basis functions for each irrep after applying
        the symmetry operations.
        """
        # labels according to the basis functions
        label = ["pi_x", "pi_y"]

        # reshape basis functions
        basis_functions = []
        for i, elem in enumerate(self.orbital_basis["pi"]):
            basis_functions.append(elem.reshape(-1, 1))

        # iterate over operations
        operation_results = []
        for i, operation_symbol in enumerate(self.characTab.operations):
            tmp_res = []
            for func in prod:
                # get index of respective basis_function
                index = next(
                    j
                    for j, arr in enumerate(basis_functions)
                    if np.array_equal(arr, func)
                )
                lab = label[index] + inversion
                dot = np.dot(
                    self.operation_matrices[lab][operation_symbol],
                    basis_functions,
                ) * int(operation_symbol.split()[0])
                # do not append the zero vector
                for arr in dot:
                    if np.any(arr):
                        tmp_res.append(arr)
            operation_results.append(tmp_res)
        return operation_results

    def get_idx_in_basis(self, label, func):
        """Get the index of a basis function in the basis list."""
        # check at which postion the orbital is in the basis
        for func in self.orbital_basisfunctions.values():
            for i, f in enumerate(func):
                if label in f:
                    j = i
                    break

    def get_ao_name(self, basis_function, angular_momentum):
        "Get the ao name from the basis function e.g. [1, 0] -> +1s]"
        basis = self.orbital_basisfunctions[angular_momentum]
        for function_value, string_rep in zip(basis_function, basis):
            if function_value != 0:
                sgn = ""
                if np.sign(function_value) == 1:
                    sgn = "+"
                else:
                    sgn = "-"
                return sgn + string_rep

    def get_spherical_harmonic(self, ao):
        """Gets from input AOs the pure spherical harmonic e.g. pz from C2_2pz"""
        try:
            orb_species = re.search(r"[A-Za-z]+", ao.split("_")[-1]).group(0)
        except AttributeError:
            orb_species = None
        try:
            location = re.search(r"\d+", ao.split("_")[0]).group()
        except AttributeError:
            location = None
        return orb_species, location

    def get_indices_of_same_basis(self,):
        """Get indices of basis functions that are the same but on different atoms."""
        orb_idx = {}
        orb_xyz = []
        # count number of different orbitals and save indices
        for i, orb in enumerate(self.mo_basis):
            ao = orb.split("_")[-1]
            if ao not in orb_idx:
                orb_idx[ao] = []
            orb_idx[ao].append(i)

            # get the different orbital species in terms of angular momentum l
            orb_species = re.search(r"\d+(\D+)", ao).group(1)
            if orb_species not in orb_xyz:
                orb_xyz.append(orb_species)
        return orb_idx, orb_xyz

    def get_salcs(self):
        """"""
        orb_idx, orb_xyz = self.get_indices_of_same_basis()
        # count number of different orbitals and save indices
        for i, orb in enumerate(self.mo_basis):
            ao = orb.split("_")[-1]
            if ao not in orb_idx:
                orb_idx[ao] = []
            orb_idx[ao].append(i)

            # get the different orbital species in terms of angular momentum l
            orb_species = re.search(r"\d+(\D+)", ao).group(1)
            if orb_species not in orb_xyz:
                orb_xyz.append(orb_species)
        for orb in orb_xyz:
            lab, op = self.get_symmetry_adapted_basis(orb)
            for label, operation in zip(lab, op):
                if label not in self.proj_results:
                    self.proj_results[label] = {
                        "labels": [],
                        "operations": [],
                    }
                self.proj_results[label]["labels"].append(orb)
                self.proj_results[label]["operations"].append(operation)
        # construct for each reducible representation the salcs as
        # linear combination
        # TODO write this section more elegant and readable
        lst = [0 for _ in self.mo_basis]
        for mulliken, data in self.proj_results.items():
            summands = []
            # seperate summands if degenerate
            # TODO Adapt for other orbital labels or types.
            # only valid for x,y degeneracy
            deg = False
            if "E" in mulliken:
                deg = True
                summands_x = []
                summands_y = []

            for orb, idx in orb_idx.items():
                for label, operation in zip(
                    data["labels"], data["operations"]
                ):
                    j = None
                    if label in orb:
                        # check at which postion the orbital is in the basis
                        for bas in self.orbital_basisfunctions.values():
                            for i, f in enumerate(bas):
                                if label in f:
                                    j = i
                                    break

                        tmp = lst.copy()
                        # assign linear combinations to the orbitals to
                        # the input orbital list
                        ops = []
                        for i, val in enumerate(operation[j]):
                            if val:
                                # TODO This solution is really not nice
                                # and should be changed
                                bas_func = re.search(
                                    r"([a-zA-Z]+)(\d+)",
                                    self.orbital_basisfunctions[
                                        list(label)[0]
                                    ][i],
                                ).group(1)
                                if bas_func in orb:
                                    ops.append(i)
                        for id, o in zip(idx, ops):
                            tmp[id] = np.sign(operation[j][o])
                        summands.append(np.array(tmp))
                        # for E group
                        if deg:
                            if "x" in label:
                                summands_x.append(np.array(tmp))
                            elif "y" in label:
                                summands_y.append(np.array(tmp))
            if deg:
                self.proj_results[mulliken]["salcs"] = (
                    self.generate_combinations(summands_x)
                )
                self.proj_results[mulliken][
                    "salcs"
                ] += self.generate_combinations(summands_y)
            else:
                self.proj_results[mulliken]["salcs"] = (
                    self.generate_combinations(summands)
                )

    def generate_combinations(self, vectors):
        """generate all linear combinations of list of vectors"""
        signs = [-1, 1]  # Possible signs
        num_vectors = len(vectors)
        combinations = []

        # Generate all combinations using only + and -
        for i in range(2**num_vectors):  # 2^n combinations
            combination = np.zeros_like(vectors[0]).astype(float)

            for j in range(num_vectors):
                # Use bitwise operations to decide + or - for each vector
                sign = signs[(i >> j) & 1]
                combination += sign * vectors[j]
            combinations.append(combination)

        return combinations

    def assign_mo_coefficients(self, mos):
        """Assigns the molecular orbital coefficients to the generated SALCs
        with respect to the signs"""
        symmetries = ["" for _ in mos]
        for i, mo in enumerate(mos):
            for mul, data in self.proj_results.items():
                if not symmetries[i]:
                    for j, salc in enumerate(data["salcs"]):
                        same = np.all(np.sign(mo) == np.sign(salc))
                        if same:
                            symmetries[i] = mul
                            break
        return symmetries
        # print()
        # print(symmetries)
        # print(len(symmetries))

    def assign_mo_symmetry_species(self, mos):
        """
        Assign symmetries sigma, pi or delta to molecular orbtals.
        sigma orbitals: s, p_z, d_zz        orbitals contribute
        pi orbitals:    p_x, p_y, dyz, dxz  orbitals contribute
        delta orbitals: d_xy, d_xx-yy | dxx, dyy  orbitals contribute
        """
        symmetry_species_res = []
        symmetry_species_bas = {
            "sigma": {"orb": ["s", "pz", "dzz"], "found_all": False},
            "pi": {"orb": ["px", "py", "dyz", "dxz"], "found_all": False},
            "delta": {
                "orb": ["dxx", "dyy", "dxxyy", "dxy"],
                "found_all": False,
            },
        }
        for mo in mos:
            for i, contribution in enumerate(mo):
                if contribution:
                    orb_species = re.search(
                        r"\d+(\D+)", self.mo_basis[i].split("_")[-1]
                    ).group(1)
                    for key, symm in symmetry_species_bas.items():
                        symm["found_all"] = orb_species in symm["orb"]
            for key, value in symmetry_species_bas.items():
                if value["found_all"]:
                    symmetry_species_res.append(key)
        return symmetry_species_res
