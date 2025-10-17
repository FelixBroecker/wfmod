import pytest
import numpy as np
from wfmod.symmetry.mo_product import MOProduct


@pytest.fixture
def moProd():
    """Fixture to provide a MOProduct instance for all tests"""
    point_group = "d2h"
    mo_basis = ["C1_1s", "C1_2s", "C1_3s", "C1_4s", "C1_5s", "C1_1px", "C1_1py", "C1_1pz", "C1_2px", "C1_2py", "C1_2pz", "C1_3px", "C1_3py", "C1_3pz", "C1_1dzz", "C1_1dxz", "C1_1dyz", "C1_1dxxyy", "C1_1dxy", "C2_1s", "C2_2s", "C2_3s", "C2_4s", "C2_5s", "C2_1px", "C2_1py", "C2_1pz", "C2_2px", "C2_2py", "C2_2pz", "C2_3px", "C2_3py", "C2_3pz", "C2_1dzz", "C2_1dxz", "C2_1dyz", "C2_1dxxyy", "C2_1dxy"]
    moProd = MOProduct(point_group, mo_basis)
    return moProd


# test sign assignment of several variables in term
@pytest.mark.parametrize("a,expected", [
   (
        [np.array([1, 0, 0, 0, 0, 0]), np.array([-1, 0, 0, 0, 0, 0])],
        (-1, [np.array([1, 0, 0, 0, 0, 0]), np.array([1, 0, 0, 0, 0, 0])])
    ),
   (
       [np.array([1, 0, 0, 0, 0, 0]), np.array([1, 0, 0, 0, 0, 0])],
       (1, [np.array([1, 0, 0, 0, 0, 0]), np.array([1, 0, 0, 0, 0, 0])])
    ),
   (
       [np.array([-1, 0, 0, 0, 0, 0]), np.array([-1, 0, 0, 0, 0, 0])],
       (1, [np.array([1, 0, 0, 0, 0, 0]), np.array([1, 0, 0, 0, 0, 0])])
    ),
   (
       [np.array([-1, 0, -1, 0, 0, 0]), np.array([-1, 0, 1, 0, 0, 0])],
       (-1, [np.array([1, 0, 1, 0, 0, 0]), np.array([1, 0, 1, 0, 0, 0])])
    ),
])
def test_get_sign(moProd, a, expected):
    """Test get_sign method of MOProduct class"""
    sign_result, arrays_result = moProd.get_sign(a)
    sign_expected, arrays_expected = expected

    # Compare sign
    assert sign_result == sign_expected

    # Compare arrays elementwise
    for arr_r, arr_e in zip(arrays_result, arrays_expected):
        assert np.array_equal(arr_r, arr_e), f"Arrays differ: {arr_r} vs {arr_e}"


# test summation of two functions
@pytest.mark.parametrize("a,b,expected", [
   (
       [-1, [np.array([1, 0, 0, 0, 0, 0])]],
       [-1, [np.array([1, 0, 0, 0, 0, 0])]],
       [-2, [np.array([1, 0, 0, 0, 0, 0])]],
   ),
   (
       [+1, [np.array([1, 0, 0, 0, 0, 0]), np.array([0, 1, 0, 0, 0, 0])]],
       [-1, [np.array([1, 0, 0, 0, 0, 0]), np.array([0, 1, 0, 0, 0, 0])]],
       [0, [np.array([1, 0, 0, 0, 0, 0]), np.array([0, 1, 0, 0, 0, 0])]],
   ),
   (
       [-1, [np.array([1, 0, 0, 0, 0, 0]), np.array([0, 1, 0, 0, 0, 0])]],
       [-1, [np.array([1, 0, 0, 0, 0, 0]), np.array([0, 1, 0, 0, 0, 0])]],
       [-2, [np.array([1, 0, 0, 0, 0, 0]), np.array([0, 1, 0, 0, 0, 0])]],
   ),
   (
       [-1, [np.array([1, 0, 0, 0, 0, 0]), np.array([0, 1, 0, 0, 0, 0])]],
       [-1, [np.array([1, 0, 0, 0, 0, 0])]],
       None,
   ),
   (
       [-1, [np.array([1, 0, 0, 0, 0, 0]), np.array([0, 1, 0, 0, 0, 0])]],
       [-1, [np.array([1, 0, 0, 0, 0, 0]), np.array([0, 1, 0, 0, 0])]],
       None,
   ),
])
def test_add_two_functions(moProd, a, b, expected):
    """Test add_functions method of MOProduct class"""
    result = moProd.add_two_functions(a, b)

    if expected is not None:
        assert result[0] == expected[0], f"Prefactors differ: {result[0]} vs {expected[0]}"
        for res_arr, exp_arr in zip(result[1], expected[1]):
            assert np.array_equal(res_arr, exp_arr), f"Arrays differ: {res_arr} vs {exp_arr}"
    else:
        assert result is None


# test summation of several functions
@pytest.mark.parametrize("a,expected", [
   (
    [
        [+1, [np.array([1, 0, 0, 0, 0, 0]), np.array([0, 1, 0, 0, 0, 0])]],
        [-1, [np.array([1, 0, 0, 0, 0, 0]), np.array([0, 1, 0, 0, 0, 0])]],
        [+1, [np.array([1, 0, 0, 0, 0, 0]), np.array([0, 1, 0, 0, 0, 0])]],
        [+1, [np.array([1, 0, 0, 0, 0, 0]), np.array([0, 1, 0, 0, 0, 0])]]
    ],
    [[+2, [np.array([1, 0, 0, 0, 0, 0]), np.array([0, 1, 0, 0, 0, 0])]]]
   ),
   (
    [
        [+1, [np.array([1, 0, 0, 0, 0, 0]), np.array([0, 1, 0, 0, 0, 0])]],
        [-1, [np.array([1, 0, 0, 0, 0, 0]), np.array([0, 1, 0, 0, 0, 0])]],
        [-1, [np.array([1, 0, 0, 0, 0, 0]), np.array([0, 1, 0, 0, 0, 0])]],
        [+1, [np.array([1, 0, 0, 0, 0, 0]), np.array([0, 1, 0, 0, 0, 0])]]
    ],
    [[0, [np.array([1, 0, 0, 0, 0, 0]), np.array([0, 1, 0, 0, 0, 0])]]]
    ),
    (
    [
        [+1, [np.array([1, 0, 0, 0, 0, 0]), np.array([0, 1, 0, 0, 0, 0])]],
        [-1, [np.array([0, 1, 0, 0, 0, 0]), np.array([0, 1, 0, 0, 0, 0])]],
        [+1, [np.array([1, 0, 0, 0, 0, 0]), np.array([0, 1, 0, 0, 0, 0])]],
        [+1, [np.array([1, 0, 0, 0, 0, 0]), np.array([0, 1, 0, 0, 0, 0])]]
    ],
    [
        [3, [np.array([1, 0, 0, 0, 0, 0]), np.array([0, 1, 0, 0, 0, 0])]],
        [-1, [np.array([0, 1, 0, 0, 0, 0]), np.array([0, 1, 0, 0, 0, 0])]]
    ]
   ),
   (
    [
        [+1, [np.array([1, 0, 0, 0, 0, 0]), np.array([0, 1, 0, 0, 0, 0])]],
        [-1, [np.array([0, 1, 0, 0, 0, 0]), np.array([0, 1, 0, 0, 0, 0])]],
        [+1, [np.array([1, 0, 1, 0, 0, 0]), np.array([0, 1, 0, 0, 0, 0])]],
        [+1, [np.array([1, 0, 0, 0, 0, 0]), np.array([0, 1, 0, 0, 1, 0])]]
    ],
    [
        [+1, [np.array([1, 0, 0, 0, 0, 0]), np.array([0, 1, 0, 0, 0, 0])]],
        [-1, [np.array([0, 1, 0, 0, 0, 0]), np.array([0, 1, 0, 0, 0, 0])]],
        [+1, [np.array([1, 0, 1, 0, 0, 0]), np.array([0, 1, 0, 0, 0, 0])]],
        [+1, [np.array([1, 0, 0, 0, 0, 0]), np.array([0, 1, 0, 0, 1, 0])]]
    ],
   )
])
def test_sum_all_functions(moProd, a, expected):
    """Test sum_identical_terms method of MOProduct class"""
    result = moProd.sum_identical_terms(a)

    for res, exp in zip(result, expected):
        assert res[0] == exp[0], f"Prefactors differ: {res[0]} vs {exp[0]}"
        for res_arr, exp_arr in zip(res[1], exp[1]):
            assert np.array_equal(res_arr, exp_arr), f"Arrays differ: {res_arr} vs {exp_arr}"
