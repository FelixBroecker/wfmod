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


@pytest.mark.parametrize("a,expected", [
   ([np.array([1, 0, 0, 0, 0, 0]), np.array([-1, 0, 0, 0, 0, 0])], (-1, [np.array([1, 0, 0, 0, 0, 0]), np.array([1, 0, 0, 0, 0, 0])])),
   ([np.array([1, 0, 0, 0, 0, 0]), np.array([1, 0, 0, 0, 0, 0])], (1, [np.array([1, 0, 0, 0, 0, 0]), np.array([1, 0, 0, 0, 0, 0])])),
   ([np.array([-1, 0, 0, 0, 0, 0]), np.array([-1, 0, 0, 0, 0, 0])], (1, [np.array([1, 0, 0, 0, 0, 0]), np.array([1, 0, 0, 0, 0, 0])])),
   ([np.array([-1, 0, -1, 0, 0, 0]), np.array([-1, 0, 1, 0, 0, 0])], (-1, [np.array([1, 0, 1, 0, 0, 0]), np.array([1, 0, 1, 0, 0, 0])])),
])
def test_get_sign(moProd, a, expected):
    """Test get_sign method of MOProduct class"""
    result = moProd.get_sign(a)
    sign_result, arrays_result = result
    sign_expected, arrays_expected = expected

    # Compare sign
    assert sign_result == sign_expected

    # Compare arrays elementwise
    for arr_r, arr_e in zip(arrays_result, arrays_expected):
        assert np.array_equal(arr_r, arr_e), f"Arrays differ: {arr_r} vs {arr_e}"
