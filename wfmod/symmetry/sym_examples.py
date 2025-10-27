import numpy as np
import yaml
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from mo_product import MOProduct
from salc import SALC

if __name__ == "__main__":
    # tests in development stage
    data_set = "c2_tz"
    cartesian = False
    if data_set == "c2_sz":
        # C2 in minimal basis
        path = "docs/orca_sto3g.yaml"
        point_group = "d2h"
        orbital_basis = ["C1_1s", "C1_2s", "C1_1px", "C1_1py", "C1_1pz", "C2_1s", "C2_2s", "C2_1px", "C2_1py", "C2_1pz"]
        orca_reference = ["Ag", "B1u", "Ag", "B1u", "B3u", "B2u", "Ag", "B2g", "B3g", "B1u"]
        # parse MO coefficients
        with open(path, "r") as file:
            data = yaml.safe_load(file)
        mos = list(map(list, zip(*data["molecularOrbitals"]["coefficients"].values())))

    elif data_set == "c2_dz":
        # C2 in double zeta
        path = "docs/orca_dzae.yaml"
        point_group = "d2h"
        orbital_basis = ["C1_1s", "C1_2s", "C1_3s", "C1_4s", "C1_1px", "C1_1py", "C1_1pz", "C1_2px", "C1_2py", "C1_2pz", "C2_1s", "C2_2s", "C2_3s", "C2_4s", "C2_1px", "C2_1py", "C2_1pz", "C2_2px", "C2_2py", "C2_2pz"]
        orca_reference = ["Ag", "B1u", "Ag", "B1u", "B2u", "B3u", "Ag", "B2g", "B3g", "B1u", "Ag", "B2u", "B3u", "Ag", "B2g", "B3g", "B1u", "B1u", "Ag", "B1u"]

        # parse MO coefficients
        with open(path, "r") as file:
            data = yaml.safe_load(file)
        mos = data["molecularOrbitals"]["coefficients"].values()
        mos = list(map(list, zip(*data)))

    elif data_set == "c2_tz":
        path = "docs/orca_tzpae.yaml"
        point_group = "d4h_expanded"
        if not cartesian:
            orbital_basis = ["C1_1s", "C1_2s", "C1_3s", "C1_4s", "C1_5s", "C1_1px", "C1_1py", "C1_1pz", "C1_2px", "C1_2py", "C1_2pz", "C1_3px", "C1_3py", "C1_3pz", "C1_1dzz", "C1_1dxz", "C1_1dyz", "C1_1dxxyy", "C1_1dxy", "C2_1s", "C2_2s", "C2_3s", "C2_4s", "C2_5s", "C2_1px", "C2_1py", "C2_1pz", "C2_2px", "C2_2py", "C2_2pz", "C2_3px", "C2_3py", "C2_3pz", "C2_1dzz", "C2_1dxz", "C2_1dyz", "C2_1dxxyy", "C2_1dxy"]
        else:
            orbital_basis = ["C1_1s", "C1_2s", "C1_3s", "C1_4s", "C1_5s", "C1_1px", "C1_1py", "C1_1pz", "C1_2px", "C1_2py", "C1_2pz", "C1_3px", "C1_3py", "C1_3pz", "C1_1dxx", "C1_1dyy", "C1_1dzz", "C1_1dxy", "C1_1dxz", "C1_1dyz", "C2_1s", "C2_2s", "C2_3s", "C2_4s", "C2_5s", "C2_1px", "C2_1py", "C2_1pz", "C2_2px", "C2_2py", "C2_2pz", "C2_3px", "C2_3py", "C2_3pz", "C2_1dxx", "C2_1dyy", "C2_1dzz", "C2_1dxy", "C2_1dxz", "C2_1dyz"]
        orca_reference = ["Ag", "B1u", "Ag", "B1u", "B2u", "B3u", "Ag", "B3g", "B2g", "B1u", "B2u", "B3u", "Ag", "B3g", "B2g", "Ag", "B1u", "B1u", "B1g", "Ag", "B2u", "B3u", "Ag", "Au", "B1u", "B3u", "B2u", "B3g", "B2g", "B1u", "B2g", "B3g", "Ag", "B1u", "Ag", "B1u", "Ag", "B1u",]
        gamess_reference = ["A1G", "A2U", "A1G", "A2U", "EU", "EU", "A1G", "EG", "EG", "A2U", "EU", "EU", "A1G", "A1G", "EG", "EG", "A2U", "A2U", "B1G", "B2G", "EU", "EU", "A1G", "B2U", "B1U", "A1G", "EU", "EU", "EG", "EG", "A2U", "A2U", "EG", "EG", "A1G", "A2U", "A1G", "A2U", "A1G", "A2U",]

        # parse orca mos
        data = [[] for _ in orbital_basis]
        with open(
            "docs/orca_tzpae.mkl", "r"
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

    if cartesian:
        data = [[] for _ in orbital_basis]
        # parse gamess mos
        counter = 0
        with open(
            "docs/gamess_tzpae.out", "r"
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

    # test salcs for mo symmetry assignment

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
        print()

    # test mo products
    moProd = MOProduct(point_group, orbital_basis, cartesian=cartesian)
    print("mo")
    _ = moProd.get_transformations_in_mo_basis()
    ao_1 = np.eye(1, 38, 6).flatten()  # 1px orbital on atom 1 of a linear molecule
    # ao_1 = moProd.get_sign(ao_1)
    ao_2 = np.eye(1, 38, 5).flatten()
    # ao_2 = moProd.get_sign(ao_2)
    print("ao_1", ao_1)
    print("ao_2", ao_2)
    mo_product = [ao_2, ao_2]
    same_irrep_mos = [ao_1, ao_2]

    print("start projection")
    linear_combination = moProd.get_all_ao_product_projections(mo_product, "A1g", same_irrep_mos)
    print(linear_combination)

    # moProd.assign_ao_products_to_mos(prod, labels, [mos[4], mos[5]])

    # print(len(moProd.mo_basis))

    # test for identical terms in mo product
