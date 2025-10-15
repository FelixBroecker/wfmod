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
                print(self.basis[idx])
                tmp += (self.basis[idx],)
            products_in_input_basis.append(tmp)
        print(products_in_input_basis)

        # apply symmetry operator to each factor of each product
        # as example do for first product
        product = products_in_input_basis[0]
        for factor in product:
            print(factor)
            sph_harmonic = self.get_spherical_harmonic(factor)
            mulliken, projection = self.get_symmetry_adapted_basis(sph_harmonic)
            print(mulliken, projection)


        print()
        print(res)
        return res
