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


# test scalar multiplication of a term
@pytest.mark.parametrize("a,b,expected", [
   (
       4,
        (-1, [np.array([1, 0, 0, 0, 0, 0]), np.array([1, 0, 0, 0, 0, 0])]),
        (-4, [np.array([1, 0, 0, 0, 0, 0]), np.array([1, 0, 0, 0, 0, 0])])
    ),
   (
       -1,
       (1, [np.array([1, 0, 0, 0, 0, 0]), np.array([1, 0, 0, 0, 0, 0])]),
       (-1, [np.array([1, 0, 0, 0, 0, 0]), np.array([1, 0, 0, 0, 0, 0])])
    ),
   (
       0.5,
       (1, [np.array([1, 0, 0, 0, 0, 0]), np.array([1, 0, 0, 0, 0, 0])]),
       (0.5, [np.array([1, 0, 0, 0, 0, 0]), np.array([1, 0, 0, 0, 0, 0])])
    ),
])
def test_scalar_multiplication(moProd, a, b, expected):
    """Test scalar_multiplication method of MOProduct class"""

    result = moProd.scalar_multiplication(a, b)
    # prefactor is same
    assert result[0] == expected[0]

    # Compare arrays elementwise
    for arr_r, arr_e in zip(result[1], expected[1]):
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
            # 4th MO of C2 triple zeta (pi_x)
            [-0.0, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0678866, -0.0, -0.0, 0.4420203,
             -0.0, -0.0, 0.1339298, 0.0, 0.0, -0.0, -0.0306001, 0.0, 0.0, -0.0,
             0.0, -0.0, 0.0, -0.0, 0.0, 0.0678866, 0.0, -0.0, 0.4420203, -0.0,
             -0.0, 0.1339298, 0.0, -0.0, -0.0, 0.0306001, 0.0, 0.0],
            # 5th MO of C2 triple zeta (pi_y)
            [0.0, -0.0, 0.0, -0.0, 0.0, 0.0678866, 0.0, 0.0, 0.4420203, -0.0,
             -0.0, 0.1339298, 0.0, 0.0, -0.0, -0.0306001, -0.0, 0.0, 0.0, -0.0,
             0.0, -0.0, 0.0, -0.0, 0.0678866, 0.0, -0.0, 0.4420203, -0.0, 0.0,
             0.1339298, 0.0, -0.0, 0.0, 0.0306001, 0.0, -0.0, 0.0],
        ],
        [
            [(6, 5), (6, 8), (6, 11), (6, 15), (6, 24), (6, 27), (6, 30),
             (6, 34), (9, 5), (9, 8), (9, 11), (9, 15), (9, 24), (9, 27),
             (9, 30), (9, 34), (12, 5), (12, 8), (12, 11), (12, 15), (12, 24),
             (12, 27), (12, 30), (12, 34), (16, 5), (16, 8), (16, 11),
             (16, 15), (16, 24), (16, 27), (16, 30), (16, 34), (25, 5),
             (25, 8), (25, 11), (25, 15), (25, 24), (25, 27), (25, 30),
             (25, 34), (28, 5), (28, 8), (28, 11), (28, 15), (28, 24),
             (28, 27), (28, 30), (28, 34), (31, 5), (31, 8), (31, 11),
             (31, 15), (31, 24), (31, 27), (31, 30), (31, 34), (35, 5),
             (35, 8), (35, 11), (35, 15), (35, 24), (35, 27), (35, 30),
             (35, 34)],
            [1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0,
             1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0,
             -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0,
             -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0,
             1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
             -1.0, 1.0, 1.0, 1.0, 1.0]
         ]
    ),
   ]
)
# test mo product computation
def test_compute_mo_product(moProd, a, expected):
    """Test compute_mo_product method of MOProduct class"""
    result = moProd.compute_mo_product(a)

    # check tuples
    for res, exp in zip(result[0], expected[0]):
        assert res == exp, f"Tuples differ: {res} vs {exp}"

    # check signs
    for res, exp in zip(result[1], expected[1]):
        assert res == exp, f"Signs differ: {res} vs {exp}"


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
    ),
    (
        "1 C4_z+",
        np.eye(1, 38, 5).flatten() + np.eye(1, 38, 24).flatten(),  # 1px1 and 1px2 orbital on atom 1 + 2 of a linear molecule
        (np.eye(1, 38, 6).flatten() + np.eye(1, 38, 25).flatten()) * -1,
    )
])
def test_get_transformations_in_mo_basis(moProd, a, b, expected):
    """Test get_transformations_in_mo_basis method of MOProduct class"""
    transformations = moProd.get_transformations_in_mo_basis()
    array_result = np.dot(transformations[a], b)

    # Compare arrays elementwise
    assert np.array_equal(array_result, expected), f"Arrays differ: {array_result} vs {expected}"


@pytest.mark.parametrize("a,b,expected", [
   (
        [np.eye(1, 38, 5).flatten(), np.eye(1, 38, 5).flatten()],   # px1 * px1
        "A1g",
        [
            [0.25, [np.eye(1, 38, 5).flatten()], np.eye(1, 38, 5).flatten()],
            [0.25, [np.eye(1, 38, 6).flatten()], np.eye(1, 38, 6).flatten()],
            [0.25, [np.eye(1, 38, 24).flatten()], np.eye(1, 38, 24).flatten()],
            [0.25, [np.eye(1, 38, 25).flatten()], np.eye(1, 38, 25).flatten()]
            ]
    ),
    (
        [np.eye(1, 38, 5).flatten(), np.eye(1, 38, 5).flatten()],   # px1 * px1
        "A2g",
        [
            [0.0, [np.eye(1, 38, 5).flatten()], np.eye(1, 38, 5).flatten()],
            [0.0, [np.eye(1, 38, 6).flatten()], np.eye(1, 38, 6).flatten()],
            [0.0, [np.eye(1, 38, 24).flatten()], np.eye(1, 38, 24).flatten()],
            [0.0, [np.eye(1, 38, 25).flatten()], np.eye(1, 38, 25).flatten()]
            ]
    ),
    (
        [np.eye(1, 38, 5).flatten(), np.eye(1, 38, 5).flatten()],   # px1 * px1
        "B1g",
        [
            [0.25, [np.eye(1, 38, 5).flatten()], np.eye(1, 38, 5).flatten()],
            [-0.25, [np.eye(1, 38, 6).flatten()], np.eye(1, 38, 6).flatten()],
            [0.25, [np.eye(1, 38, 24).flatten()], np.eye(1, 38, 24).flatten()],
            [-0.25, [np.eye(1, 38, 25).flatten()], np.eye(1, 38, 25).flatten()]
            ]
    ),
])
def test_get_projection_of_ao_product(moProd, a, b, expected):
    """Test get_projection_of_ao_product method of MOProduct class"""
    for i, term in enumerate(a):
        a[i] = moProd.get_sign(term)
    moProd.get_transformations_in_mo_basis()
    result = moProd.get_projection_of_ao_product(a, b)


    assert len(result) == len(expected), f"Number of terms differ: {len(result)} vs {len(expected)}"
    for res, exp in zip(result, expected):
        assert res[0] == exp[0], f"Prefactors differ: {res[0]} vs {exp[0]}"
        for res_arr, exp_arr in zip(res[1], exp[1]):
            assert np.array_equal(res_arr, exp_arr), f"Arrays differ: {res_arr} vs {exp_arr}"
