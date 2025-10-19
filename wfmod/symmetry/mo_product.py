import re

import numpy as np
import itertools as itertools
from wfmod.symmetry.salc import SALC


class MOProduct(SALC):
    """Class to get symmetry adapted product basis functions of molecular orbitals."""
    def __init__(self, point_group: str, basis: list, cartesian: bool = False):
        super().__init__(point_group, basis, cartesian)

    def add_two_functions(self,
                            f1: list[int, np.ndarray],
                            f2: list[int, np.ndarray]
                          ) -> list[int, np.ndarray] | None:
        """
        Add two functions represented as lists of numpy arrays.
        Add them only if the 'variables' in the function part are identical.

        Returns the summed function or None if they cannot be added.
        """
        equal = True

        # check if same number of factors
        if len(f1[1]) != len(f2[1]):
            return None

        for factor_1, factor_2 in zip(f1[1], f2[1]):
            # check if they variables have the same shape
            if factor_1.shape != factor_2.shape:
                return None

            # check if the arrays are identical
            if not np.array_equal(factor_1, factor_2):
                equal = False

        if equal:
            prefactor = f1[0] + f2[0]
            return [prefactor, f1[1]]
        else:
            return None

    def term_key(self, term):
        """Convert a list of arrays into a hashable key."""
        return tuple(tuple(a) for a in term)

    def get_sign(self, term: list[np.ndarray]):
        """Get the overall sign of a term represented as a list of arrays."""
        sign = 1
        positive_terms = []
        for arr in term:
            arr_sign = np.sign(np.prod(arr[arr != 0]))
            positive_terms.append(abs(arr))
            sign *= arr_sign
        return sign, positive_terms

    def sum_identical_terms(self, terms):
        """Sum identical terms in a list of terms represented as lists of arrays."""
        summed_terms = []
        used = set()  # track indices we already summed
        for i, term in enumerate(terms):
            if i in used:
                continue

            # Find all identical terms regardless of sign
            key = self.term_key(term[1])

            sign = term[0]
            for j in range(i+1, len(terms)):
                if self.term_key(terms[j][1]) == key:
                    sign += terms[j][0]
                    used.add(j)

            # Multiply the term by the count
            summed = [sign, term[1]]

            summed_terms.append(summed)

        return summed_terms

    def compute_mo_product(self, mos: list, zero=1e-12):
        """
        Compute the products of molecular orbitals in terms of atomic orbitals.
        Returns a list with products of atomic orbitals given in indices of the
        MO input list (indice corresponds to AO).

        Returns: list of tuples with indices of the MO basis functions.
        """
        res = []

        # Enumerate over all index combinations
        for combo in itertools.product(*[enumerate(mo) for mo in mos]):
            indices = tuple(idx for idx, _ in combo)
            values = [val for _, val in combo]

            result = np.prod(values)  # multiply all values together
            if result > zero:
                res.append(indices)

        return res

    def get_ao_basis_function_by_idx(self, idx: int) -> np.array:
        """Assign the transformations in ao basis to mo basis."""
        ao = np.zeros(len(self.mo_basis))
        ao[idx] = 1.0
        return ao

    def get_transformations_in_mo_basis(self) -> dict:
        """
        Get transformation matrices for each symmetry operation in the mo basis.

        Returns a dictionary with one operation matrix per symmetry operation.
        """

        # transform mo basis in a more comparable basis
        # assigned as 1 s1 , 2 s1 , 1 px1, ... , 1 s2, 2 s2, 1 px2 ...
        # so they belong together if first number is the same
        ao_label = []
        ao_number = []
        for mo in self.mo_basis:
            ao, location = self.get_spherical_harmonic(mo)
            number = re.search(r'\d+', mo.split("_")[-1]).group(0)
            ao_label.append(ao + location)
            ao_number.append(number)

        # iterate over different angular momenta types  (s, px, py, pz, dxy ...)
        transformation_groups: dict[str, dict[str, list[int]]] = {}
        for l_type in self.operation_matrices.keys():
            transformation_groups[l_type] = {}
            # find that in the ao_label and group them by their number
            # to distinguish between them (e.g. 1s1 1s2 and 2s1 2s2)
            for i, (number, label) in enumerate(zip(ao_number, ao_label)):
                spherical, index = self.get_spherical_harmonic(label)
                if l_type == spherical:
                    transformation_groups.setdefault(l_type, {}).setdefault(f"{index}", []).append(i)

        # get operation matrices in mo basis by mapping the string transformations
        # allocate space for operation matrices in mo basis
        operation_matrices_mo_basis = {
            operation: np.zeros((len(self.mo_basis), len(self.mo_basis))) for operation in self.characTab.operations
            }
        # now get the transformtaions from the human readable string representation
        # and map them into the mo basis that we have only one operation matrix for
        # each symmetry operation for the full basis
        pattern = r"([+-])(\w+)"
        for l_type, _ in transformation_groups.items():
            for mulliken, transformation in self.string_transformations[l_type].items():
                for operation in transformation:
                    # parse transformations
                    func_val = operation.split(" -> ")
                    sign, orbital = re.findall(
                        pattern,
                        func_val[1],
                    )[0]
                    start = func_val[0]  # e.g., "s1"
                    sign, end = (1 if sign == "+" else -1, orbital)  # e.g., [(1, 's1')] where one is the sign
                    # start orbital
                    l_type, idx = self.get_spherical_harmonic(start)
                    start_indices = transformation_groups[l_type][idx]
                    # end orbital after operation
                    l_type, idx = self.get_spherical_harmonic(end)
                    end_indices = transformation_groups[l_type][idx]
                    for s_idx, e_idx in zip(start_indices, end_indices):
                        operation_matrices_mo_basis[mulliken][e_idx][s_idx] = sign
        return operation_matrices_mo_basis


    def transform_angular_basis_to_mo_basis(self, ao_product, aos_labels_in_product, mo_labels):
        """Transform from angular basis to mo basis."""
        res = []
        # determine angular momentum
        angular_momentum = ""
        for i, function  in enumerate(ao_product):
            for l_type in self.orbital_basisfunctions.keys():
                if l_type in aos_labels_in_product[i]:
                    angular_momentum = l_type
                    break
            # transform ao basis in orbital label (str)
            tmp = self.get_ao_name(function, angular_momentum)

            # now reassign to mo label
            # the angular part remains. the number corresponds to the number of the atom
            # the digit in the angular part in mo string (_1px) is taken
            # from the initial mo label

            # get labels from mo input
            atom, angular = mo_labels[i].split("_")
            mo_number = re.search(r'\d+', angular).group(0)
            atom_letter = re.search(r'([A-Za-z]+)', atom).group(1)

            # get labels from ao basis
            ao, location = self.get_spherical_harmonic(tmp)

            # add everything together
            orbital_name = f"{atom_letter}{location}_{mo_number}{ao}"

            # get sign
            res.append(orbital_name)

        sign = self.get_sign(ao_product)
        return [sign] + res

    def get_projection_of_mo_product(self, mo_list: list, target_symmetry, zero=1e-12):
        for mo in mo_list:
            self.print_mo(mo)

        res = []

        # Enumerate over all index combinations
        for combo in itertools.product(*[enumerate(mo) for mo in mo_list]):
            indices = tuple(idx for idx, _ in combo)
            values = [val for _, val in combo]

            result = np.prod(values)  # multiply all values together
            if result > zero:
                res.append(indices)

        # get AO for each index
        products_in_input_basis = []
        for tup in res:
            tmp = []
            for idx in tup:
                tmp.append(self.mo_basis[idx])
            products_in_input_basis.append(tuple(tmp))


        # apply symmetry operator to each factor of each product
        # as example do for first product
        results = []
        results_mo_labels = []
        for product in products_in_input_basis:

            aos_labels_in_product = []
            for factor in product:
                ao, location = self.get_spherical_harmonic(factor)
                aos_labels_in_product.append(ao + location)


            print("Applying symmetry operators to product:")
            projection_result = self.apply_symmetry_operator_on_product(aos_labels_in_product, irrep)
            print(projection_result)
            print(aos_labels_in_product)
            print(product)

            projection_result_labels = []
            for i, ao in enumerate(projection_result):
                res = []

                # determine angular momentum
                angular_momentum = ""
                for l_type in self.orbital_basisfunctions.keys():
                    if l_type in aos_labels_in_product[i]:
                        angular_momentum = l_type
                        break


                for function in ao:
                    res.append(self.get_ao_name(function, angular_momentum))
                projection_result_labels.append(res)

            print("reshape")

            print(np.column_stack(projection_result_labels))
            print()

            # for tmp in projection_result:
            #     for t in tmp:
            #         for val in t:
            #             print(f"{val:8.4f}", end=" ")
            #         print()
            #     print()

            combined = [list(x) for x in zip(*projection_result)]
            print(combined[0])
            # reassign aos to in the mo list
            results.append(self.sum_identical_terms(combined))

            print("combined results:")
            mo_labels = []
            for res in combined:
                tmp = self.transform_angular_basis_to_mo_basis(res, aos_labels_in_product, product)
                mo_labels.append(tmp)
            results_mo_labels.append(mo_labels)
        return results, results_mo_labels

    def assign_ao_products_to_mos(self, ao_products: list, ao_products_labels: list, mo_list: list, zero = 1e-12):
        """Assign ao products to mo products and return linear combinations of mos."""

        product_list = []
        print()
        for prod in ao_products_labels:
            for p in prod:
                lab = []
                for label in p[1:]:
                    lab.append(self.mo_basis.index(label))
                product_list.append(tuple(lab))
        product_list = set(product_list)

        print("Product list")
        print(product_list)
        print("Start with assignment of ao products to mo products")
        # generate all mo combinations
        res = []
        # generate full basis of mo products
        # Enumerate over all index combinations

        for mo_1 in mo_list:
            for mo_2 in mo_list:
                tmp = []
                for combo in itertools.product(*[enumerate(mo) for mo in [mo_1, mo_2]]):
                    indices = tuple(idx for idx, _ in combo)
                    values = [val for _, val in combo]

                    result = np.prod(values)
                    if abs(result) > zero:
                        tmp.append(indices)
                res.append(tmp)
        print(len(res))
        print(res)

        assign = np.zeros((len(product_list)), dtype=int)
        res_been_found = np.zeros((4, len(res[0])), dtype=bool)
        for j, product in enumerate(product_list):
            found = False
            for i, mo_product in enumerate(res):
                for k, mo in enumerate(mo_product):
                    if mo == product:
                        assign[j] = i
                        res_been_found[i][k] = True
                        found = True
                        break
                if found:
                    break
            if not found:
                print("Not found in mo products!")
                print(product)
        false_indices = np.where(res_been_found[0] == False)[0]
        print(false_indices)
        print(res[0][31])
        print(res[0][59])
        print(self.mo_basis[16])
        print(self.mo_basis[35])
        # print(res_been_found[0][false_indices])
        # print(ao_products[0])
        print(assign)
