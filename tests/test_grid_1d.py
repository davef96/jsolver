import jax
import pytest
from jsolver.grid_1d import grid_1D

@pytest.fixture
def grid():
    return grid_1D(mx=128, xb=0., xe=1., rim=1, grid_type=0, centered=True)

def test_init_values(grid):
    assert grid.mx == 128
    assert grid.nx == 129
    assert grid.centered
    assert grid.rim == 1
    assert grid.xb == 0.0
    assert grid.xe == 1.0
    assert grid.grid_type == 0

def test_dependent_values(grid):
    assert grid.length == 1.0
    assert grid.del_ == pytest.approx(0.0078125)
    assert grid.idel == pytest.approx(128.0)

def test_array_values(grid):
    assert grid.centers[0] == pytest.approx(-0.0078125)
    assert grid.centers[-1] == pytest.approx(1.0078125)
    assert grid.edges[0] == pytest.approx(-0.01171875)
    assert grid.edges[-1] == pytest.approx(1.01171875)
    assert grid.widths[0] == pytest.approx(0.0078125)
    assert grid.widths[-1] == pytest.approx(0.0078125)

def test_sizes(grid):
    assert grid.centers.size == grid.widths.size
    assert grid.edges.size == grid.centers.size + 1

def test_pytree_static(grid):
    assert jax.tree.map(lambda x: x + 1, grid) == grid

@pytest.fixture
def sinusoidal_grid():
    return grid_1D(mx=128, xb=0., xe=1., rim=1, grid_type=1)

def test_array_values_sinusoidal(sinusoidal_grid):
    assert sinusoidal_grid.centers[0] == pytest.approx(-0.00428662)
    assert sinusoidal_grid.centers[-1] == pytest.approx(1.0121)
    assert sinusoidal_grid.edges[0] == pytest.approx(-0.00857324)
    assert sinusoidal_grid.edges[-1] == pytest.approx(1.01639, rel=1e-5)
    assert sinusoidal_grid.widths[0] == pytest.approx(0.00857324)
    assert sinusoidal_grid.widths[-1] == pytest.approx(0.00857324)

def test_unknown_grid_type():
    with pytest.raises(Exception):
        grid = grid_1D(mx=128, xb=0., xe=1., rim=1, grid_type=42)

if __name__ == "__main__":
    pytest.main()
