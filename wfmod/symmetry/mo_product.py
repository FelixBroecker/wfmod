import re
import numpy as np
import itertools as itertools
from wfmod.symmetry.salc import SALC

class MOProduct(SALC):
    def __init__(self, point_group: str, basis: list, cartesian: bool = False):
        super().__init__(point_group, basis, cartesian)

    def print_mo(self, mo):
        print("MO:")
        print(mo)

    def get_mo_product(self, mo_list: list):

        for mo in mo_list:
            self.print_mo(mo)

        res = []
        # get indices that form non-zero products

        # Enumerate over all index combinations
        for combo in itertools.product(*[enumerate(mo) for mo in mo_list]):
            indices = tuple(idx for idx, _ in combo)
            values = [val for _, val in combo]

            result = np.prod(values)  # multiply all values together
            if result > 1e-12:
                res.append(indices)

        # get AO for each index
        products_in_input_basis = []
        for tup in res:
            tmp = ()
            for idx in tup:
                print(self.mo_basis[idx])
                tmp += (self.mo_basis[idx],)
            products_in_input_basis.append(tmp)
        print(products_in_input_basis)

        # apply symmetry operator to each factor of each product
        # as example do for first product
        product = products_in_input_basis[15]
        print("Product:")
        print(product)

        aos_labels_in_product = []
        for factor in product:
            ao, location = self.get_spherical_harmonic(factor)
            aos_labels_in_product.append(ao + location)

        print("aos_labels_in_product:")
        print(aos_labels_in_product)
        print()

        print("Applying symmetry operators to product:")
        projection_result = self.apply_symmetry_operator_on_product(aos_labels_in_product)


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

        return res
