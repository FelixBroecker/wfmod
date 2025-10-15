import re
from wfmod.symmetry.salc import SALC

class MOProduct(SALC):
    def __init__(self, point_group: str, basis: list, cartesian: bool = False):
        super().__init__(point_group, basis, cartesian)

    def print_mos(self, mo1, mo2):
        print("MO 1:")
        print(mo1)
        print("MO 2:")
        print(mo2)

    def get_mo_product(self, mo1, mo2):
        self.print_mos(mo1, mo2)

        res = []
        # get indices that form non-zero products
        for i, val1 in enumerate(mo1):
            for j, val2 in enumerate(mo2):
                result = val1 * val2
                if result > 1e-12:
                    res.append(((i, j)))

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
        product = products_in_input_basis[0]
        print(product)
        res = []
        for factor in product:
            ao, location = self.get_spherical_harmonic(factor)
            res.append(ao + location)

        print(res)
        print()

        print("Applying symmetry operators to product:")
        self.apply_symmetry_operator_on_product(res)


        print()
        print(res)
        return res
