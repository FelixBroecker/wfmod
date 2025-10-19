import pytest
import numpy as np
from wfmod.symmetry.mo_product import MOProduct


@pytest.fixture
def moProd():
    """Fixture to provide a MOProduct instance for all tests"""
    point_group = "d4h_expanded"
    mo_basis = [
        "C1_1s", "C1_2s", "C1_3s", "C1_4s", "C1_5s", "C1_1px", "C1_1py", "C1_1pz",
        "C1_2px", "C1_2py", "C1_2pz", "C1_3px", "C1_3py", "C1_3pz", "C1_1dzz", "C1_1dxz",
        "C1_1dyz", "C1_1dxxyy", "C1_1dxy", "C2_1s", "C2_2s", "C2_3s", "C2_4s", "C2_5s", "C2_1px",
        "C2_1py", "C2_1pz", "C2_2px", "C2_2py", "C2_2pz", "C2_3px", "C2_3py", "C2_3pz", "C2_1dzz",
        "C2_1dxz", "C2_1dyz", "C2_1dxxyy", "C2_1dxy"
        ]
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


@pytest.mark.parametrize("a,expected", [
   (
        [
           [-0.7015951, -0.7011436, -0.1929879, -0.180916, -0.0, 0.0, -0.0653454, -0.0, 0.0, -0.1407968],
           [0.0010617, -0.0111153, -0.2077491, -0.2999345, 0.0, 0.0, 0.6158199, -0.0, 0.0, 1.2070551]
        ],
        [
            (0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3), (2, 1), (2, 2),
            (2, 3), (3, 1), (3, 2), (3, 3), (6, 1), (6, 2), (6, 3), (9, 1), (9, 2), (9, 3)
         ]
    ),
    (
        [
            [-0.7015951, -0.7011436, -0.1929879, -0.180916, -0.0, 0.0, -0.0653454, -0.0, 0.0, -0.1407968],
            [-0.7015951, -0.7011436, -0.1929879, -0.180916, -0.0, 0.0, -0.0653454, -0.0, 0.0, -0.1407968]
        ],
        [
             (0, 0), (0, 1), (0, 2), (0, 3), (0, 6), (0, 9), (1, 0), (1, 1), (1, 2), (1, 3), (1, 6),
             (1, 9), (2, 0), (2, 1), (2, 2), (2, 3), (2, 6), (2, 9), (3, 0), (3, 1), (3, 2), (3, 3),
             (3, 6), (3, 9), (6, 0), (6, 1), (6, 2), (6, 3), (6, 6), (6, 9), (9, 0), (9, 1), (9, 2),
             (9, 3), (9, 6), (9, 9)
         ]
    ),
    (
        [
            [-0.7015951, -0.7011436, -0.1929879, -0.180916, -0.0, 0.0, -0.0653454, -0.0, 0.0, -0.1407968],
            [0.0, -0.0, 0.0, 0.0, 0.6247297, 0.0, 0.0, -0.8339753, 0.0, 0.0],
            [0.0, 0.0, 0.0, -0.0, 0.6247297, 0.0, 0.0, 0.8339753, 0.0, -0.0]
        ],
        [
            (0, 7, 4), (0, 7, 7), (1, 7, 4), (1, 7, 7), (2, 7, 4), (2, 7, 7), (3, 7, 4), (3, 7, 7),
            (6, 7, 4), (6, 7, 7), (9, 7, 4), (9, 7, 7)
         ]
    ),
   ]
)
# test mo product computation
def test_compute_mo_product(moProd, a, expected):
    """Test compute_mo_product method of MOProduct class"""
    result = moProd.compute_mo_product(a)

    for res, exp in zip(result, expected):
        assert res == exp, f"Tuples differ: {res} vs {exp}"


# test transformation matrix by performing some transformations with it
@pytest.mark.parametrize("a,b,expected", [
   (
        "1 E",
        np.eye(1, 38, 0).flatten(),  # 1s orbital on atom 1 of a linear molecule
        np.eye(1, 38, 0).flatten(),
    ),
    (
        "1 C2",
        np.eye(1, 38, 5).flatten(),  # 1px orbital on atom 1 of a linear molecule
        np.eye(1, 38, 5).flatten() * -1,
    ),
    (
        "1 i",
        np.eye(1, 38, 13).flatten(),  # 3pz orbital on atom 1 of a linear molecule
        np.eye(1, 38, 32).flatten() * -1,
    ),
    (
        "1 S4-",
        np.eye(1, 38, 16).flatten(),  # 1dyz orbital on atom 1 of a linear molecule
        np.eye(1, 38, 34).flatten(),
    )

])
def test_get_transformations_in_mo_basis(moProd, a, b, expected):
    """Test get_transformations_in_mo_basis method of MOProduct class"""
    transformations = moProd.get_transformations_in_mo_basis()
    array_result = np.dot(transformations[a], b)

    # Compare arrays elementwise
    assert np.array_equal(array_result, expected), f"Arrays differ: {array_result} vs {expected}"
