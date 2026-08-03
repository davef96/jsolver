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
    arr_shape = (z_res + 3, y_res + 3, x_res + 3)

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
    arr_shape = (z_res + 3, y_res + 3, x_res + 3)

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
    arr_shape = (z_res + 3, y_res + 3, x_res + 3)

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


@pytest.fixture
def grad_func_solve(setup):
    grid_x, grid_y, grid_z, phi, _, _ = setup

    def gsolve(grid_x, grid_y, grid_z, phi, rhs, lam, D_xx, D_yy, D_zz):
        phi_res, _ = solve_3d_simple(grid_x=grid_x, grid_y=grid_y, grid_z=grid_z, phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy, D_zz=D_zz)
        return phi_res

    return gsolve


@pytest.fixture
def grad_func_solve_32(setup_32):
    grid_x, grid_y, grid_z, phi, _, _ = setup_32

    def gsolve(grid_x, grid_y, grid_z, phi, rhs, lam, D_xx, D_yy, D_zz):
        phi_res, _ = solve_3d_simple(grid_x=grid_x, grid_y=grid_y, grid_z=grid_z, phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy, D_zz=D_zz)
        return phi_res

    return gsolve    

@pytest.fixture
def grad_func_solve_mixed(setup_mixed):
    grid_x, grid_y, grid_z, phi, _, _ = setup_mixed

    def gsolve(grid_x, grid_y, grid_z, phi, rhs, lam, D_xx, D_yy, D_zz):
        phi_res, _ = solve_3d_simple(grid_x=grid_x, grid_y=grid_y, grid_z=grid_z, phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy, D_zz=D_zz)
        return phi_res

    return gsolve      


def problem_1(x_cen, y_cen, z_cen, shape):
    D_xx = 1.0
    D_yy = 1.0
    D_zz = 1.0
    sx = np.sin(np.pi * x_cen)[np.newaxis, np.newaxis, :]
    sy = np.sin(np.pi * y_cen)[np.newaxis, :, np.newaxis]
    sz = np.sin(np.pi * z_cen)[:, np.newaxis, np.newaxis]
    phi_ana = sx * sy * sz
    lam = np.zeros(shape)
    rhs = -(np.pi ** 2) * (D_xx + D_yy + D_zz) * phi_ana

    return phi_ana, rhs, lam, D_xx, D_yy, D_zz

@pytest.fixture
def test1_setup(setup):
    grid_x, grid_y, grid_z, phi, _, _ = setup
    return problem_1(grid_x.centers, grid_y.centers, grid_z.centers, phi.shape)

@pytest.fixture
def test1_setup_32(setup_32):
    grid_x, grid_y, grid_z, phi, _, _ = setup_32
    return problem_1(grid_x.centers, grid_y.centers, grid_z.centers, phi.shape)

@pytest.fixture
def test1_setup_mixed(setup_mixed):
    grid_x, grid_y, grid_z, phi, _, _ = setup_mixed
    return problem_1(grid_x.centers, grid_y.centers, grid_z.centers, phi.shape)

def test1(setup, test1_setup, grad_func_solve):
    grid_x, grid_y, grid_z, phi, acceptable_error_abs, acceptable_error_rel = setup
    phi_ana, rhs, lam, D_xx, D_yy, D_zz = test1_setup
    gsolve = grad_func_solve

    curr_time = time.time()
    phi_res, iterations = solve_3d_simple(grid_x=grid_x, grid_y=grid_y, grid_z=grid_z, phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy, D_zz=D_zz)
    print("solve took {} s".format(time.time() - curr_time))

    assert iterations == 19
    assert pytest.approx(6.936905980033114e-05, abs=acceptable_error_abs, rel=acceptable_error_rel) == rmse(phi_ana, phi_res)

    if check_autodiff_fwd:
        curr_time = time.time()
        check_grads(gsolve, (grid_x, grid_y, grid_z, phi, rhs, lam, D_xx, D_yy, D_zz), order=1, modes=["fwd"])
        print("check_grads(gsolve, ...) took {} s".format(time.time() - curr_time))

def test1_32(setup_32, test1_setup_32, grad_func_solve_32):
    grid_x, grid_y, grid_z, phi, acceptable_error_abs, acceptable_error_rel = setup_32
    phi_ana, rhs, lam, D_xx, D_yy, D_zz = test1_setup_32
    gsolve = grad_func_solve_32

    curr_time = time.time()
    phi_res, iterations = solve_3d_simple(grid_x=grid_x, grid_y=grid_y, grid_z=grid_z, phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy, D_zz=D_zz)
    print("solve took {} s".format(time.time() - curr_time))

    assert iterations == 19
    assert pytest.approx(0.0002712919636528891, abs=acceptable_error_abs, rel=acceptable_error_rel) == rmse(phi_ana, phi_res)

    if check_autodiff_fwd:
        curr_time = time.time()
        check_grads(gsolve, (grid_x, grid_y, grid_z, phi, rhs, lam, D_xx, D_yy, D_zz), order=1, modes=["fwd"])
        print("check_grads(gsolve, ...) took {} s".format(time.time() - curr_time))

def test1_32(setup_mixed, test1_setup_mixed, grad_func_solve_mixed):
    grid_x, grid_y, grid_z, phi, acceptable_error_abs, acceptable_error_rel = setup_mixed
    phi_ana, rhs, lam, D_xx, D_yy, D_zz = test1_setup_mixed
    gsolve = grad_func_solve_mixed

    curr_time = time.time()
    phi_res, iterations = solve_3d_simple(grid_x=grid_x, grid_y=grid_y, grid_z=grid_z, phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy, D_zz=D_zz)
    print("solve took {} s".format(time.time() - curr_time))

    assert iterations == 33
    assert pytest.approx(0.0002049924560414506, abs=acceptable_error_abs, rel=acceptable_error_rel) == rmse(phi_ana, phi_res)

    if check_autodiff_fwd:
        curr_time = time.time()
        check_grads(gsolve, (grid_x, grid_y, grid_z, phi, rhs, lam, D_xx, D_yy, D_zz), order=1, modes=["fwd"])
        print("check_grads(gsolve, ...) took {} s".format(time.time() - curr_time))        


def problem_2(x_cen, y_cen, z_cen, shape):
    D_xx = 1.0
    D_yy = 1.0
    D_zz = 1.0
    x = x_cen[np.newaxis, np.newaxis, :]
    y = y_cen[np.newaxis, :, np.newaxis]
    z = z_cen[:, np.newaxis, np.newaxis]
    sx = np.sin(np.pi * x)
    sy = np.sin(np.pi * y)
    sz = np.sin(np.pi * z)
    phi_ana = sx * sy * sz
    lam = 0.2 * x * y ** 2 * z**3
    print("\n shape of lambda",lam.shape)
    rhs = -((np.pi ** 2) * (D_xx + D_yy + D_zz) + lam) * phi_ana

    return phi_ana, rhs, lam, D_xx, D_yy, D_zz

@pytest.fixture
def test2_setup(setup):
    grid_x, grid_y, grid_z, phi, _, _ = setup
    return problem_2(grid_x.centers, grid_y.centers, grid_z.centers, phi.shape)

@pytest.fixture
def test2_setup_32(setup_32):
    grid_x, grid_y, grid_z, phi, _, _ = setup_32
    return problem_2(grid_x.centers, grid_y.centers, grid_z.centers, phi.shape)

@pytest.fixture
def test2_setup_mixed(setup_mixed):
    grid_x, grid_y, grid_z, phi, _, _ = setup_mixed
    return problem_2(grid_x.centers, grid_y.centers, grid_z.centers, phi.shape)    

def test2(setup, test2_setup, grad_func_solve):
    grid_x, grid_y, grid_z, phi, acceptable_error_abs, acceptable_error_rel = setup
    phi_ana, rhs, lam, D_xx, D_yy, D_zz = test2_setup
    gsolve = grad_func_solve

    curr_time = time.time()
    phi_res, iterations = solve_3d_simple(grid_x=grid_x, grid_y=grid_y, grid_z=grid_z, phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy, D_zz=D_zz)
    print("solve took {} s".format(time.time() - curr_time))

    assert iterations == 19
    assert pytest.approx(6.935753745686714e-05, abs=acceptable_error_abs, rel=acceptable_error_rel) == rmse(phi_ana, phi_res)

    if check_autodiff_fwd:
        curr_time = time.time()
        check_grads(gsolve, (grid_x, grid_y, grid_z, phi, rhs, lam, D_xx, D_yy, D_zz), order=1, modes=["fwd"])
        print("check_grads(gsolve, ...) took {} s".format(time.time() - curr_time))

def test2_32(setup_32, test2_setup_32, grad_func_solve_32):
    grid_x, grid_y, grid_z, phi, acceptable_error_abs, acceptable_error_rel = setup_32
    phi_ana, rhs, lam, D_xx, D_yy, D_zz = test2_setup_32
    gsolve = grad_func_solve_32

    curr_time = time.time()
    phi_res, iterations = solve_3d_simple(grid_x=grid_x, grid_y=grid_y, grid_z=grid_z, phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy, D_zz=D_zz)
    print("solve took {} s".format(time.time() - curr_time))

    assert iterations == 19
    assert pytest.approx(0.00027124687452780976, abs=acceptable_error_abs, rel=acceptable_error_rel) == rmse(phi_ana, phi_res)

    if check_autodiff_fwd:
        curr_time = time.time()
        check_grads(gsolve, (grid_x, grid_y, grid_z, phi, rhs, lam, D_xx, D_yy, D_zz), order=1, modes=["fwd"])
        print("check_grads(gsolve, ...) took {} s".format(time.time() - curr_time))        

def test2_mixed(setup_mixed, test2_setup_mixed, grad_func_solve_mixed):
    grid_x, grid_y, grid_z, phi, acceptable_error_abs, acceptable_error_rel = setup_mixed
    phi_ana, rhs, lam, D_xx, D_yy, D_zz = test2_setup_mixed
    gsolve = grad_func_solve_mixed

    curr_time = time.time()
    phi_res, iterations = solve_3d_simple(grid_x=grid_x, grid_y=grid_y, grid_z=grid_z, phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy, D_zz=D_zz)
    print("solve took {} s".format(time.time() - curr_time))

    assert iterations == 33
    assert pytest.approx(0.00020495839279424837, abs=acceptable_error_abs, rel=acceptable_error_rel) == rmse(phi_ana, phi_res)

    if check_autodiff_fwd:
        curr_time = time.time()
        check_grads(gsolve, (grid_x, grid_y, grid_z, phi, rhs, lam, D_xx, D_yy, D_zz), order=1, modes=["fwd"])
        print("check_grads(gsolve, ...) took {} s".format(time.time() - curr_time))           

def problem_3(x_cen, y_cen, z_cen, shape):
    D_xx = 1.0
    D_yy = 0.5
    D_zz = 2.0
    x = x_cen[np.newaxis, np.newaxis, :]
    y = y_cen[np.newaxis, :, np.newaxis]
    z = z_cen[:, np.newaxis, np.newaxis]
    sx = np.sin(np.pi * x)
    sy = np.sin(np.pi * y)
    sz = np.sin(np.pi * z)
    phi_ana = sx * sy * sz
    lam = 0.2 * x * y ** 2 * z**3
    print("\n shape of lambda",lam.shape)
    rhs = -((np.pi ** 2) * (D_xx + D_yy + D_zz) + lam) * phi_ana

    return phi_ana, rhs, lam, D_xx, D_yy, D_zz

@pytest.fixture
def test3_setup(setup):
    grid_x, grid_y, grid_z, phi, _, _ = setup
    return problem_3(grid_x.centers, grid_y.centers, grid_z.centers, phi.shape)

@pytest.fixture
def test3_setup_32(setup_32):
    grid_x, grid_y, grid_z, phi, _, _ = setup_32
    return problem_3(grid_x.centers, grid_y.centers, grid_z.centers, phi.shape)

@pytest.fixture
def test3_setup_mixed(setup_mixed):
    grid_x, grid_y, grid_z, phi, _, _ = setup_mixed
    return problem_3(grid_x.centers, grid_y.centers, grid_z.centers, phi.shape)    

def test3(setup, test3_setup, grad_func_solve):
    grid_x, grid_y, grid_z, phi, acceptable_error_abs, acceptable_error_rel = setup
    phi_ana, rhs, lam, D_xx, D_yy, D_zz = test3_setup
    gsolve = grad_func_solve

    curr_time = time.time()
    phi_res, iterations = solve_3d_simple(grid_x=grid_x, grid_y=grid_y, grid_z=grid_z, phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy, D_zz=D_zz)
    print("solve took {} s".format(time.time() - curr_time))

    assert iterations == 36
    assert pytest.approx(6.93591831010889e-05, abs=acceptable_error_abs, rel=acceptable_error_rel) == rmse(phi_ana, phi_res)

    if check_autodiff_fwd:
        curr_time = time.time()
        check_grads(gsolve, (grid_x, grid_y, grid_z, phi, rhs, lam, D_xx, D_yy, D_zz), order=1, modes=["fwd"])
        print("check_grads(gsolve, ...) took {} s".format(time.time() - curr_time))

def test3_32(setup_32, test3_setup_32, grad_func_solve_32):
    grid_x, grid_y, grid_z, phi, acceptable_error_abs, acceptable_error_rel = setup_32
    phi_ana, rhs, lam, D_xx, D_yy, D_zz = test3_setup_32
    gsolve = grad_func_solve_32

    curr_time = time.time()
    phi_res, iterations = solve_3d_simple(grid_x=grid_x, grid_y=grid_y, grid_z=grid_z, phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy, D_zz=D_zz)
    print("solve took {} s".format(time.time() - curr_time))

    assert iterations == 35
    assert pytest.approx(0.0002712533139540335, abs=acceptable_error_abs, rel=acceptable_error_rel) == rmse(phi_ana, phi_res)

    if check_autodiff_fwd:
        curr_time = time.time()
        check_grads(gsolve, (grid_x, grid_y, grid_z, phi, rhs, lam, D_xx, D_yy, D_zz), order=1, modes=["fwd"])
        print("check_grads(gsolve, ...) took {} s".format(time.time() - curr_time))   

def test3_mixed(setup_mixed, test3_setup_mixed, grad_func_solve_mixed):
    grid_x, grid_y, grid_z, phi, acceptable_error_abs, acceptable_error_rel = setup_mixed
    phi_ana, rhs, lam, D_xx, D_yy, D_zz = test3_setup_mixed
    gsolve = grad_func_solve_mixed

    curr_time = time.time()
    phi_res, iterations = solve_3d_simple(grid_x=grid_x, grid_y=grid_y, grid_z=grid_z, phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy, D_zz=D_zz)
    print("solve took {} s".format(time.time() - curr_time))

    assert iterations == 27
    assert pytest.approx(0.00024402769848607482, abs=acceptable_error_abs, rel=acceptable_error_rel) == rmse(phi_ana, phi_res)

    if check_autodiff_fwd:
        curr_time = time.time()
        check_grads(gsolve, (grid_x, grid_y, grid_z, phi, rhs, lam, D_xx, D_yy, D_zz), order=1, modes=["fwd"])
        print("check_grads(gsolve, ...) took {} s".format(time.time() - curr_time))           