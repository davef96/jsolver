import numpy as np
import jax.numpy as jnp
import jax
from jax.test_util import check_grads
import pytest
import time
from jsolver.grid_1d import grid_1D
from jsolver.solver_3d import solve_3d_fcycle_simple, solve_3d_fixed_fcycle_simple, solve_3d_fixed_simple, solve_3d_simple, solve_3d

# some configuration
check_autodiff_fwd = True # done for all tests
check_autodiff_rev = True # currently, only done for test1 (if 'check_fixed' is also True)
check_fixed = True # fixed number of iterations; currently, only done for test1
check_w     = True # W-cycle (gamma=2); currently, only done for test1
check_f     = True # F-cycle; currently, only done for test1


@jax.jit
def rmse(phi_ana, phi_res):
    error = (phi_ana - phi_res)[1:-1,1:-1,1:-1]
    return jnp.linalg.norm(error) / jnp.sqrt(error.size)

def test_numpy_default_dtype():
    arr = np.zeros(42)
    assert arr.dtype == np.dtype('float64')


@pytest.fixture
def setup():
    x_res = 64
    y_res = 64
    z_res = 64
    arr_shape = (y_res + 3, x_res + 3, z_res + 3)

    phi = np.zeros(arr_shape)

    grid_x = grid_1D(mx=x_res, xb=0., xe=1., rim=1, grid_type=0, centered=True)
    grid_y = grid_1D(mx=y_res, xb=0., xe=1., rim=1, grid_type=0, centered=True)
    grid_z = grid_1D(mx=z_res, xb=0., xe=1., rim=1, grid_type=0, centered=True)

    acceptable_error_abs = 2e-15
    acceptable_error_rel = 1e-9

    return grid_x, grid_y, grid_z, phi, acceptable_error_abs, acceptable_error_rel

@pytest.fixture
def setup_32():
    x_res = 32
    y_res = 32
    z_res = 32
    arr_shape = (y_res + 3, x_res + 3, z_res + 3)

    phi = np.zeros(arr_shape)

    grid_x = grid_1D(mx=x_res, xb=0., xe=1., rim=1, grid_type=0, centered=True)
    grid_y = grid_1D(mx=y_res, xb=0., xe=1., rim=1, grid_type=0, centered=True)
    grid_z = grid_1D(mx=z_res, xb=0., xe=1., rim=1, grid_type=0, centered=True)

    acceptable_error_abs = 2e-15
    acceptable_error_rel = 1e-9

    return grid_x, grid_y, grid_z, phi, acceptable_error_abs, acceptable_error_rel


@pytest.fixture
def setup_mixed():
    x_res = 32
    y_res = 64
    z_res = 32
    arr_shape = (y_res + 3, x_res + 3, z_res + 3)

    phi = np.zeros(arr_shape)

    grid_x = grid_1D(mx=x_res, xb=0., xe=1., rim=1, grid_type=0, centered=True)
    grid_y = grid_1D(mx=y_res, xb=0., xe=1., rim=1, grid_type=0, centered=True)
    grid_z = grid_1D(mx=z_res, xb=0., xe=1., rim=1, grid_type=0, centered=True)

    acceptable_error_abs = 2e-15
    acceptable_error_rel = 1e-9

    return grid_x, grid_y, grid_z, phi, acceptable_error_abs, acceptable_error_rel


def test_not_power_of_two_setup_x():
    with pytest.raises(Exception) as exec_info:
        solve_3d_simple(grid_x=grid_1D(mx=100), grid_y=grid_1D(mx=128), grid_z=grid_1D(mx=128), phi=np.zeros(131), rhs=np.zeros(131), lam=np.zeros(131), D_xx=1.0, D_yy=1.0, D_zz=1.0)
    # make sure, we get the correct exception:
    assert "Resolution of x-grid is not a power of 2" in str(exec_info.value)

def test_not_power_of_two_setup_y():
    with pytest.raises(Exception) as exec_info:
        solve_3d_simple(grid_x=grid_1D(mx=128), grid_y=grid_1D(mx=100), grid_z=grid_1D(mx=32), phi=np.zeros(131), rhs=np.zeros(131), lam=np.zeros(131), D_xx=1.0, D_yy=1.0, D_zz=1.0)
    # make sure, we get the correct exception:
    assert "Resolution of y-grid is not a power of 2" in str(exec_info.value)

def test_not_power_of_two_setup_z():
    with pytest.raises(Exception) as exec_info:
        solve_3d_simple(grid_x=grid_1D(mx=128), grid_y=grid_1D(mx=32), grid_z=grid_1D(mx=55), phi=np.zeros(131), rhs=np.zeros(131), lam=np.zeros(131), D_xx=1.0, D_yy=1.0, D_zz=1.0)
    # make sure, we get the correct exception:
    assert "Resolution of z-grid is not a power of 2" in str(exec_info.value)


def test_inefficient_setup_x():
    with pytest.raises(Exception) as exec_info:
        solve_3d_simple(grid_x=grid_1D(mx=32), grid_y=grid_1D(mx=4), grid_z=grid_1D(mx=4), phi=np.zeros(131), rhs=np.zeros(131), lam=np.zeros(131), D_xx=1.0, D_yy=1.0, D_zz=1.0)
    assert "Resolution of x-grid not efficient for multigrid configuration" in str(exec_info.value)

def test_inefficient_setup_y():
    with pytest.raises(Exception) as exec_info:
        solve_3d_simple(grid_x=grid_1D(mx=4), grid_y=grid_1D(mx=32), grid_z=grid_1D(mx=4), phi=np.zeros(131), rhs=np.zeros(131), lam=np.zeros(131), D_xx=1.0, D_yy=1.0, D_zz=1.0)
    assert "Resolution of y-grid not efficient for multigrid configuration" in str(exec_info.value)    

def test_inefficient_setup_z():
    with pytest.raises(Exception) as exec_info:
        solve_3d_simple(grid_x=grid_1D(mx=4), grid_y=grid_1D(mx=4), grid_z=grid_1D(mx=32), phi=np.zeros(131), rhs=np.zeros(131), lam=np.zeros(131), D_xx=1.0, D_yy=1.0, D_zz=1.0)
    assert "Resolution of z-grid not efficient for multigrid configuration" in str(exec_info.value)      


def test_illegal_setup_x():
    with pytest.raises(Exception) as exec_info:
        solve_3d_simple(grid_x=grid_1D(mx=32, grid_type=1), grid_y=grid_1D(mx=32), grid_z=grid_1D(mx=32), phi=np.zeros(131), rhs=np.zeros(131), lam=np.zeros(131), D_xx=1.0, D_yy=1.0, D_zz=1.0)
    assert "Only linear grids are supported" in str(exec_info.value)      

def test_illegal_setup_y():
    with pytest.raises(Exception) as exec_info:
        solve_3d_simple(grid_x=grid_1D(mx=32), grid_y=grid_1D(mx=32, grid_type=1), grid_z=grid_1D(mx=32), phi=np.zeros(131), rhs=np.zeros(131), lam=np.zeros(131), D_xx=1.0, D_yy=1.0, D_zz=1.0)
    assert "Only linear grids are supported" in str(exec_info.value)    

def test_illegal_setup_z():
    with pytest.raises(Exception) as exec_info:
        solve_3d_simple(grid_x=grid_1D(mx=32), grid_y=grid_1D(mx=32), grid_z=grid_1D(mx=32, grid_type=1), phi=np.zeros(131), rhs=np.zeros(131), lam=np.zeros(131), D_xx=1.0, D_yy=1.0, D_zz=1.0)
    assert "Only linear grids are supported" in str(exec_info.value)        
