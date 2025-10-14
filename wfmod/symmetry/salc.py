import numpy as np
import re
import yaml
from wfmod.charactertables import CharacterTable


class SALC:
    def __init__(self, point_group: str, basis: list, cartesian: bool = False):
        """"""
        self.point_group = point_group
        self.characTab = CharacterTable(point_group)
        self.basis = basis
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

    def apply_symmetry_operator_on_product(self, prod, inversion=""):
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

    def get_salcs(self):
        """"""
        orb_idx = {}
        orb_xyz = []
        # count number of different orbitals and save indices
        for i, orb in enumerate(self.basis):
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
        lst = [0 for _ in self.basis]
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
                        r"\d+(\D+)", self.basis[i].split("_")[-1]
                    ).group(1)
                    for key, symm in symmetry_species_bas.items():
                        symm["found_all"] = orb_species in symm["orb"]
            for key, value in symmetry_species_bas.items():
                if value["found_all"]:
                    symmetry_species_res.append(key)
        return symmetry_species_res


# tests in development stage
data_set = "c2_tz"
cartesian = False
if data_set == "c2_sz":
    # C2 in minimal basis
    path = "../docs/orca_sto3g.yaml"
    point_group = "d2h"
    orbital_basis = ["C1_1s", "C1_2s", "C1_1px", "C1_1py", "C1_1pz", "C2_1s", "C2_2s", "C2_1px", "C2_1py", "C2_1pz"]
    orca_reference = ["Ag", "B1u", "Ag", "B1u", "B3u", "B2u", "Ag", "B2g", "B3g", "B1u"]
    # parse MO coefficients
    with open(path, "r") as file:
        data = yaml.safe_load(file)
    mos = data["molecularOrbitals"]["coefficients"].values()

elif data_set == "c2_dz":
    # C2 in double zeta
    path = "../docs/orca_dzae.yaml"
    point_group = "d2h"
    orbital_basis = ["C1_1s", "C1_2s", "C1_3s", "C1_4s", "C1_1px", "C1_1py", "C1_1pz", "C1_2px", "C1_2py", "C1_2pz", "C2_1s", "C2_2s", "C2_3s", "C2_4s", "C2_1px", "C2_1py", "C2_1pz", "C2_2px", "C2_2py", "C2_2pz"]
    orca_reference = ["Ag", "B1u", "Ag", "B1u", "B2u", "B3u", "Ag", "B2g", "B3g", "B1u", "Ag", "B2u", "B3u", "Ag", "B2g", "B3g", "B1u", "B1u", "Ag", "B1u"]

    # parse MO coefficients
    with open(path, "r") as file:
        data = yaml.safe_load(file)
    mos = data["molecularOrbitals"]["coefficients"].values()

elif data_set == "c2_tz":
    path = "../docs/orca_tzpae.yaml"
    point_group = "d4h_expanded"
    if not cartesian:
        orbital_basis = ["C1_1s", "C1_2s", "C1_3s", "C1_4s", "C1_5s", "C1_1px", "C1_1py", "C1_1pz", "C1_2px", "C1_2py", "C1_2pz", "C1_3px", "C1_3py", "C1_3pz", "C1_1dzz", "C1_1dxz", "C1_1dyz", "C1_1dxxyy", "C1_1dxy", "C2_1s", "C2_2s", "C2_3s", "C2_4s", "C2_5s", "C2_1px", "C2_1py", "C2_1pz", "C2_2px", "C2_2py", "C2_2pz", "C2_3px", "C2_3py", "C2_3pz", "C2_1dzz", "C2_1dxz", "C2_1dyz", "C2_1dxxyy", "C2_1dxy"]
    else:
        orbital_basis = ["C1_1s", "C1_2s", "C1_3s", "C1_4s", "C1_5s", "C1_1px", "C1_1py", "C1_1pz", "C1_2px", "C1_2py", "C1_2pz", "C1_3px", "C1_3py", "C1_3pz", "C1_1dxx", "C1_1dyy", "C1_1dzz", "C1_1dxy", "C1_1dxz", "C1_1dyz", "C2_1s", "C2_2s", "C2_3s", "C2_4s", "C2_5s", "C2_1px", "C2_1py", "C2_1pz", "C2_2px", "C2_2py", "C2_2pz", "C2_3px", "C2_3py", "C2_3pz", "C2_1dxx", "C2_1dyy", "C2_1dzz", "C2_1dxy", "C2_1dxz", "C2_1dyz"]
    orca_reference = ["Ag", "B1u", "Ag", "B1u", "B2u", "B3u", "Ag", "B3g", "B2g", "B1u", "B2u", "B3u", "Ag", "B3g", "B2g", "Ag", "B1u", "B1u", "B1g", "Ag", "B2u", "B3u", "Ag", "Au", "B1u", "B3u", "B2u", "B3g", "B2g", "B1u", "B2g", "B3g", "Ag", "B1u", "Ag", "B1u", "Ag", "B1u",
    ]
    gamess_reference = ["A1G", "A2U", "A1G", "A2U", "EU", "EU", "A1G", "EG", "EG", "A2U", "EU", "EU", "A1G", "A1G", "EG", "EG", "A2U", "A2U", "B1G", "B2G", "EU", "EU", "A1G", "B2U", "B1U", "A1G", "EU", "EU", "EG", "EG", "A2U", "A2U", "EG", "EG", "A1G", "A2U", "A1G", "A2U", "A1G", "A2U",
    ]

    # parse orca mos
    data = [[] for _ in orbital_basis]
    with open(
        "../docs/orca_tzpae.mkl", "r"
    ) as reffile:
        found = False
        for line in reffile:
            if "$END" in line:
                found = False
            if "$COEFF_ALPHA" in line:
                found = True
                continue
            if "a1g" in line:
                counter = 0
                line = reffile.readline()
                continue
            if found:
                items = line.split()
                for val in items:
                    data[counter].append(float(val))
                counter += 1
    mos = list(map(list, zip(*data)))
else:
    print("Invalid input.")
    exit()

if __name__ == "__main__":
    if cartesian:
        data = [[] for _ in orbital_basis]
        # parse gamess mos
        counter = 0
        with open(
            "../docs/gamess_tzpae.out", "r"
        ) as reffile:
            found = False
            for line in reffile:
                if "EIGENVECTORS" in line:
                    found = True
                    for _ in range(6):
                        line = reffile.readline()
                if "...... END OF RHF CALCULATION ......" in line:
                    found = False
                    break
                if counter == len(orbital_basis):
                    counter = 0
                    for _ in range(4):
                        line = reffile.readline()
                if found:
                    items = line.split()
                    items = items[4:]
                    for val in items:
                        data[counter].append(float(val))
                    counter += 1
        mos = list(map(list, zip(*data)))

    salc = SALC(
        point_group,
        orbital_basis,
        cartesian=cartesian,
    )

    salc.get_salcs()
    res = salc.assign_mo_coefficients(mos)
    print("Mulliken symbols")
    print(res)
    print(len(res))

    print()
    print("Symmetry species")
    res = salc.assign_mo_symmetry_species(mos)
    print(res)

    print()
    print("Orca reference:")
    print(orca_reference)
    print(len(orca_reference))

    if data_set == "c2_tz":
        print()
        print("Gamess reference (cartesian):")
        print(gamess_reference)
        print(len(gamess_reference))
