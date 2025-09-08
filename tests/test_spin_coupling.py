import pytest
from wfmod.spin_coupling import SpinCoupling


@pytest.fixture
def sc():
    """Fixture to provide a SpinCoupling instance for all tests"""
    return SpinCoupling()


@pytest.mark.parametrize("a,expected", [
    ([[1, 2], [2, 1], [1, 2]], [[1, 2], [2, 1]]),
    ([[1, 2], [1, 2], [1, 2]], [[1, 2]]),
    ([], []),
])
def test_remove_duplicates(sc, a, expected):
    """Test remove_duplicates method of SpinCoupling class"""
    assert sc.remove_duplicates(a) == expected
