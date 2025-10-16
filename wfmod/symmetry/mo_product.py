import re
import numpy as np
import itertools as itertools
from wfmod.symmetry.salc import SALC

class MOProduct(SALC):
    def __init__(self, point_group: str, basis: list, cartesian: bool = False):
        super().__init__(point_group, basis, cartesian)

    def add_functions(self, f1, f2):
        """Add two functions represented as lists of numpy arrays.
        Add them only if they are identical."""

        # must have same number of factors
        if len(f1) != len(f2):
            return None

        result = []
        for a, b in zip(f1, f2):
            if np.array_equal(a, b):   # only if arrays are identical
                result.append(a + b)
            else:
                return None
        return result

    def term_key(self, term):
        """Convert a list of arrays into a hashable key."""
        return tuple(tuple(abs(a)) for a in term)

    def get_sign(self, term):
        """Get the overall sign of a term represented as a list of arrays."""
        sign = 1
        for arr in term:
            arr_sign = np.sign(np.prod(arr[arr != 0]))
            sign *= arr_sign
        return sign

    def sum_identical_terms(self, terms):
        """Sum identical terms in a list of terms represented as lists of arrays."""
        summed_terms = []
        used = set()  # track indices we already summed
        for i, term in enumerate(terms):
            if i in used:
                continue

            sign = self.get_sign(term)

            # Find all identical terms regardless of sign
            key = self.term_key(term)

            count = 1
            for j in range(i+1, len(terms)):
                if self.term_key(terms[j]) == key:
                    sign_j = self.get_sign(terms[j])
                    count += 1
                    sign += sign_j
                    used.add(j)

            # Multiply the term by the count
            summed = [arr * count * np.sign(sign) for arr in term]

            # check if terms are all positive
            # if not all(np.all(arr >= 0) for arr in summed):
            #     print(summed)
            #     print(sign_i, sign_j)
            #     print("not positive")
            #     exit()

            summed_terms.append(summed)

        return summed_terms


    def print_mo(self, mo):
        print("MO:")
        print(mo)

    def get_mo_product(self, mo_list: list, zero=1e-12):

        irrep="A1g"
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
        for product in products_in_input_basis:

            aos_labels_in_product = []
            for factor in product:
                ao, location = self.get_spherical_harmonic(factor)
                aos_labels_in_product.append(ao + location)

            print("Applying symmetry operators to product:")
            projection_result = self.apply_symmetry_operator_on_product(aos_labels_in_product, irrep)


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

            result = self.sum_identical_terms(combined)

            for tmp in result:
                print(tmp)


        return res
