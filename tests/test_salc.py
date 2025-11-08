import pytest
import numpy as np
from wfmod.symmetry.salc import SALC


@pytest.fixture
def salc():
    """Fixture to provide a SALC instance for all tests"""
    point_group = "d4h_expanded"
    cartesian = False
    orbital_basis = [
        "C1_1s", "C1_2s", "C1_3s", "C1_4s", "C1_5s", "C1_1px", "C1_1py", "C1_1pz",
        "C1_2px", "C1_2py", "C1_2pz", "C1_3px", "C1_3py", "C1_3pz", "C1_1dzz", "C1_1dxz",
        "C1_1dyz", "C1_1dxxyy", "C1_1dxy", "C2_1s", "C2_2s", "C2_3s", "C2_4s", "C2_5s", "C2_1px",
        "C2_1py", "C2_1pz", "C2_2px", "C2_2py", "C2_2pz", "C2_3px", "C2_3py", "C2_3pz", "C2_1dzz",
        "C2_1dxz", "C2_1dyz", "C2_1dxxyy", "C2_1dxy"
        ]
    salc = SALC(
        point_group,
        orbital_basis,
        cartesian=cartesian,
    )
    return salc


# test sign assignment of several variables in term
@pytest.mark.parametrize("a,expected", [
   ("s",
    (['A1g', 'A2u'],
     [
    np.array(
         [[0.5, 0.5],
         [0.5, 0.5]]
         ),
    np.array(
        [[ 0.5, -0.5],
        [-0.5,  0.5]]
        )
    ]
    ))
])
def test_get_symmetry_adapted_basis(salc, a, expected):
    """Test get_symmetry_adapted_basis method of SALC class"""
    result = salc.get_symmetry_adapted_basis(a)
    assert result[0] == expected[0], f"Expected irreps {expected[0]}, got {result[0]}"

    # Compare arrays elementwise
    for arr_r, arr_e in zip(result[1], expected[1]):
        assert np.array_equal(arr_r, arr_e), f"Arrays differ: {arr_r} vs {arr_e}"
