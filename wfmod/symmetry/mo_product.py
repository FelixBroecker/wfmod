import re

import numpy as np
import itertools as itertools
import time
from wfmod.symmetry.salc import SALC


class MOProduct(SALC):
    """Class to get symmetry adapted product basis functions of molecular orbitals."""
    def __init__(self, point_group: str, basis: list, cartesian: bool = False):
        super().__init__(point_group, basis, cartesian)
        self.mo_operation_matrices: dict[str, np.array] = {}

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

    def scalar_multiplication(self, scalar: float, term: list[np.ndarray]):
        """Multiply a term represented as a list of arrays by a scalar."""
        return [scalar * term[0], term[1]]

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

    def compute_mo_product(self, mos: list, zero=1e-16):
        """
        Compute the products of molecular orbitals in terms of atomic orbitals.
        Returns a list with products of atomic orbitals given in indices of the
        MO input list (indice corresponds to AO).

        Returns: list of tuples with indices of the MO basis functions.
                 list of signs corresponding to each tuple.
        """
        ao_products = []
        signs = []

        # Enumerate over all index combinations
        for combo in itertools.product(*[enumerate(mo) for mo in mos]):
            indices = tuple(idx for idx, _ in combo)
            values = [val for _, val in combo]

            result = np.prod(values)  # multiply all values together
            if np.abs(result) > zero:
                # print("indices:", indices)
                # print("result:", result)
                # print("sign:", np.sign(result))

                ao_products.append(indices)
                signs.append(np.sign(result))
        return ao_products, signs

    def compute_all_possible_mo_products(self, mos: list):
        """
        Compute all possible products of input molecular orbitals.
        e.g. for two mos: mo1 * mo1, mo1 * mo2, mo2 * mo1, mo2 * mo2"""
        product_list = []
        indices = []
        signs = []
        for i, mo_1 in enumerate(mos):
            for j, mo_2 in enumerate(mos):
                prod, sign = self.compute_mo_product([mo_1, mo_2])
                idx = (i, j)
                product_list.append(prod)
                indices.append(idx)
                signs.append(sign)
                print(f"Computed mo product for indices {idx}")
                print(f"Product: {prod}, Sign: {sign}")
        # remove same lists:
        for i in range(len(product_list)-1, -1, -1):
            for j in range(i-1, -1, -1):
                if product_list[i] == product_list[j]:
                    del product_list[i]
                    del indices[i]
                    del signs[i]
                    break

        return product_list, indices, signs

    def get_ao_basis_function_by_idx(self, idx: int) -> np.array:
        """Assign the transformations in ao basis to mo basis."""
        ao = np.zeros(len(self.mo_basis))
        ao[idx] = 1.0
        return ao

    def get_idx_by_ao_basis_function(self, ao_func) -> int:
        """Get the index of an ao basis function in the mo basis."""
        if not isinstance(ao_func, np.ndarray):
            ao_func = np.array(ao_func)
        idx = int(np.where(ao_func == 1.0)[0])
        return idx

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
        self.mo_operation_matrices = operation_matrices_mo_basis
        return operation_matrices_mo_basis

    def get_projection_of_ao_product(self, ao_product: list, target_symmetry: str, zero=1e-12):
        """
        Use Projection operator to get symmetry adapted mo product. Apply the projection operator therefore
        on each factor of the product of aos that result from a product of mos.

        Returns the projected mo products as list of lists of numpy arrays.

        """

        # apply projection operator
        # P^irrep = ( dimension/order ) sum_over_operations [ chi^irrep(op) * R(op) ]
        # (eq. 5.24 Atkins Friedman 2011. Ed. 5; Example 5.9)
        order = self.characTab.order
        mulliken_letter = re.search(r'([A-Z]+)', target_symmetry).group(0)
        dim = self.characTab.get_dimension(mulliken_letter)
        factor = dim / order
        result = [[factor, []] for _ in range(len(self.mo_operation_matrices.values()))]  # initialize result
        for i, ao in enumerate(ao_product):
            for j, operation in enumerate(self.mo_operation_matrices.values()):
                if not i:
                    character = self.characTab.characters[target_symmetry][j]
                    result[j][0] *= character
                function = np.dot(operation, ao[-1])
                sign, function = self.get_sign(function)
                result[j][0] *= sign
                result[j][1].append(function)
        result = self.sum_identical_terms(result)
        return result

    def get_all_ao_product_projections(
            self, mo_product: list,
            target_symmetry: str,
            same_irrep_mos: list,
            timings: bool = False
            ) -> list[tuple[int, int]]:
        """
        Compute all product of mos, project each ao product in the mo product
        with the projection operator. Assign the results back to linear combination of mo products.

        Return: list of tuples (sign, mo index) that specify the linear combination of mos from the same_irrep_mos input.
        The sign indicates the sign in the linear combination.
        """
        t_start = time.time()
        t1 = time.time()

        # compute mo product and get a sum of ao products
        ao_product_factors, signs = self.compute_mo_product(mo_product)

        t2 = time.time()
        t_compute_mo_product = t2 - t1
        print("t_compute_mo_product:", t_compute_mo_product)

        # convert ao indices to ao basis functions
        ao_product_factors_converted = []
        for ao_product in ao_product_factors:
            ao_funcs = []
            for idx in ao_product:
                ao_func = self.get_ao_basis_function_by_idx(idx)
                ao_funcs.append(self.get_sign(ao_func))
            ao_product_factors_converted.append(ao_funcs)

        # load projection matrices
        self.get_transformations_in_mo_basis()

        t1 = time.time()

        # project each ao product to the target symmetries
        projected_ao_products: list = []
        for sign, ao_product in zip(signs, ao_product_factors_converted):
            projection_result = self.get_projection_of_ao_product(ao_product, target_symmetry)

            # if the mo product had a sign, multiply now with sign that has
            # been excluded before applying the projection operator
            for j, proj in enumerate(projection_result):
                projection_result[j] = self.scalar_multiplication(sign, proj)
            projected_ao_products += projection_result

        t2 = time.time()
        t_project_ao_products = t2 - t1
        print("t_project_ao_products:", t_project_ao_products)

        # print("projected_ao_products before sum:", projected_ao_products)
        # sum identical terms from different mo products
        t1 = time.time()
        projected_ao_products = self.sum_identical_terms(projected_ao_products)
        t2 = time.time()
        t_sum_identical_terms = t2 - t1

        t1 = time.time()

        # convert projected ao basis functions back to indices
        for i, term in enumerate(projected_ao_products):
            tmp = []
            for ao in term[1]:
                idx = self.get_idx_by_ao_basis_function(ao)
                tmp.append(idx)
            projected_ao_products[i][1] = tuple(tmp)
        print("projected_ao_products:", projected_ao_products)

        t2 = time.time()
        t_convert_ao_to_idx = t2 - t1
        print("t_convert_ao_to_idx:", t_convert_ao_to_idx)

        t1 = time.time()
        # compare to all combinations from the input mo product and assign terms
        # to the corresponding mo product to get linear combinations of mo products.
        # pass here all mos that belong to one irrep (e.g. all degenerate mos of one E)
        mo_combinations = self.compute_all_possible_mo_products(same_irrep_mos)
        # print("mo_combinations:", mo_combinations[0][-1])
        # print("signs:", mo_combinations[2][-1])

        # for i, _ in enumerate(mo_combinations[0]):
        #     for j, ao_product in enumerate(mo_combinations[0][i]):
        #         print(f"{i}:{j}:", mo_combinations[2][i][j], ao_product)

        t2 = time.time()
        t_compute_all_mo_products = t2 - t1
        print("t_compute_all_mo_products:", t_compute_all_mo_products)

        #  mo_combinations[0] stores the mo combinations (e.g. 4 for 2 mos)
        #  mo_combinations[0][0] stores the tuples of ao indices for the first mo product (e.g. mo1 * mo1)

        t1 = time.time()
        # save the sign and the mo index for each term
        # deduce from that the linear combination of mos
        # use i as index +1 for mo index to distinguish between +0 and -0
        print("projection results:", projected_ao_products)
        mo_assignments = []
        for term in projected_ao_products:
            found = False
            for i, mo_product in enumerate(mo_combinations[0]):
                for j, ao_product in enumerate(mo_product):
                    if ao_product == term[1] and term[0] != 0:
                        # Assign mo index and consider sign of projection
                        # result and of mo product. If both match the sign
                        # of the mo product is positive.
                        mo_assignments.append(int(np.sign(term[0]) * mo_combinations[2][i][j] * (i+1)))
                        found = True
                        break
                if found:
                    break
        # print("mo combinations indices:", mo_combinations[0])
        print(mo_assignments)
        unique_assignments = list(set(mo_assignments))
        t2 = time.time()
        t_assign_ao_products_to_mos = t2 - t1
        print("t_assign_ao_products_to_mos:", t_assign_ao_products_to_mos)

        # return empty list if the result is zero
        if len(unique_assignments) == 1 and unique_assignments[0] == 0:
            return []

        # subtract 1 to get the real mo indices again and keep signs
        return_format = []
        for val in unique_assignments:
            return_format.append((np.sign(val), abs(val)-1))

        t_end = time.time()
        if timings:
            print("Timings for get_all_ao_product_projections:")
            print(f"  {'compute_mo_product':<30}: {t_compute_mo_product:.6f} s")
            print(f"  {'project_ao_products':<30}: {t_project_ao_products:.6f} s")
            print(f"  {'sum_identical_terms':<30}: {t_sum_identical_terms:.6f} s")
            print(f"  {'convert_ao_to_idx':<30}: {t_convert_ao_to_idx:.6f} s")
            print(f"  {'compute_all_mo_products':<30}: {t_compute_all_mo_products:.6f} s")
            print(f"  {'assign_ao_products_to_mos':<30}: {t_assign_ao_products_to_mos:.6f} s")
            print(f"  {'total':<30}: {t_end - t_start:.6f} s")

        return return_format
