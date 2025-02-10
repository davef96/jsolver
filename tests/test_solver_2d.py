import numpy as np
import jax.numpy as jnp
import jax
from jax.test_util import check_grads
import pytest
import time
from jsolver.grid_1d import grid_1D
from jsolver.solver_2d import solve_2d_fcycle_simple, solve_2d_fixed_fcycle_simple, solve_2d_fixed_simple, solve_2d_simple, solve_2d

# already done in 'solver_2d.py'
#jax.config.update('jax_enable_x64', True)

# some configuration
check_autodiff_fwd = True # done for all tests
check_autodiff_rev = True # currently, only done for test1 (if 'check_fixed' is also True)
check_fixed = True # fixed number of iterations; currently, only done for test1
check_w     = True # W-cycle (gamma=2); currently, only done for test1
check_f     = True # F-cycle; currently, only done for test1

@jax.jit
def rmse(phi_ana, phi_res):
    error = (phi_ana - phi_res)[1:-1,1:-1]
    return jnp.linalg.norm(error) / jnp.sqrt(error.size)

def test_numpy_default_dtype():
    arr = np.zeros(42)
    assert arr.dtype == np.dtype('float64')

@pytest.fixture
def setup():
    x_res = 128
    y_res = 128
    arr_shape = (y_res + 3, x_res + 3)

    phi = np.zeros(arr_shape)

    grid_x = grid_1D(mx=x_res, xb=0., xe=1., rim=1, grid_type=0, centered=True)
    grid_y = grid_1D(mx=y_res, xb=0., xe=1., rim=1, grid_type=0, centered=True)

    acceptable_error_abs = 2e-15
    acceptable_error_rel = 1e-9

    return grid_x, grid_y, phi, acceptable_error_abs, acceptable_error_rel

@pytest.fixture
def setup_64():
    x_res = 64
    y_res = 64
    arr_shape = (y_res + 3, x_res + 3)

    phi = np.zeros(arr_shape)

    grid_x = grid_1D(mx=x_res, xb=0., xe=1., rim=1, grid_type=0, centered=True)
    grid_y = grid_1D(mx=y_res, xb=0., xe=1., rim=1, grid_type=0, centered=True)

    acceptable_error_abs = 2e-15
    acceptable_error_rel = 1e-9

    return grid_x, grid_y, phi, acceptable_error_abs, acceptable_error_rel

@pytest.fixture
def setup_64_128():
    x_res = 64
    y_res = 128
    arr_shape = (y_res + 3, x_res + 3)

    phi = np.zeros(arr_shape)

    grid_x = grid_1D(mx=x_res, xb=0., xe=1., rim=1, grid_type=0, centered=True)
    grid_y = grid_1D(mx=y_res, xb=0., xe=1., rim=1, grid_type=0, centered=True)

    acceptable_error_abs = 2e-15
    acceptable_error_rel = 1e-9

    return grid_x, grid_y, phi, acceptable_error_abs, acceptable_error_rel

def test_not_power_of_two_setup_y():
    with pytest.raises(Exception):
        solve_2d_simple(grid_x=grid_1D(mx=128), grid_y=grid_1D(mx=100), phi=np.zeros(131), rhs=np.zeros(131), lam=np.zeros(131), D_xx=1.0, D_yy=1.0)

def test_not_power_of_two_setup_x():
    with pytest.raises(Exception):
        solve_2d_simple(grid_x=grid_1D(mx=100), grid_y=grid_1D(mx=128), phi=np.zeros(131), rhs=np.zeros(131), lam=np.zeros(131), D_xx=1.0, D_yy=1.0)

def test_inefficient_setup_y():
    with pytest.raises(Exception):
        solve_2d_simple(grid_x=grid_1D(mx=128), grid_y=grid_1D(mx=4), phi=np.zeros(131), rhs=np.zeros(131), lam=np.zeros(131), D_xx=1.0, D_yy=1.0)

def test_inefficient_setup_x():
    with pytest.raises(Exception):
        solve_2d_simple(grid_x=grid_1D(mx=4), grid_y=grid_1D(mx=128), phi=np.zeros(131), rhs=np.zeros(131), lam=np.zeros(131), D_xx=1.0, D_yy=1.0)

def test_illegal_setup_y():
    with pytest.raises(Exception):
        solve_2d_simple(grid_x=grid_1D(mx=128), grid_y=grid_1D(mx=128, grid_type=1), phi=np.zeros(131), rhs=np.zeros(131), lam=np.zeros(131), D_xx=1.0, D_yy=1.0)

def test_illegal_setup_x():
    with pytest.raises(Exception):
        solve_2d_simple(grid_x=grid_1D(mx=128, grid_type=1), grid_y=grid_1D(mx=128), phi=np.zeros(131), rhs=np.zeros(131), lam=np.zeros(131), D_xx=1.0, D_yy=1.0)

@pytest.fixture
def grad_func_solve(setup):
    grid_x, grid_y, phi, _, _ = setup

    def gsolve(grid_x, grid_y, phi, rhs, lam, D_xx, D_yy, D_xy=None):
        phi_res, _ = solve_2d_simple(grid_x=grid_x, grid_y=grid_y, phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy, D_xy=D_xy)
        return phi_res

    return gsolve

@pytest.fixture
def grad_func_solve_64(setup_64):
    grid_x, grid_y, phi, _, _ = setup_64

    def gsolve(grid_x, grid_y, phi, rhs, lam, D_xx, D_yy, D_xy=None):
        phi_res, _ = solve_2d_simple(grid_x=grid_x, grid_y=grid_y, phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy, D_xy=D_xy)
        return phi_res

    return gsolve

@pytest.fixture
def grad_func_solve_64_128(setup_64_128):
    grid_x, grid_y, phi, _, _ = setup_64_128

    def gsolve(grid_x, grid_y, phi, rhs, lam, D_xx, D_yy, D_xy=None):
        phi_res, _ = solve_2d_simple(grid_x=grid_x, grid_y=grid_y, phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy, D_xy=D_xy)
        return phi_res

    return gsolve

def problem_1(x_cen, y_cen, shape):
    D_xx = 1.0
    D_yy = 1.0
    x = x_cen[np.newaxis]
    y = y_cen[:, np.newaxis]
    ox = 1. - x
    oy = 1. - y
    prod = y * oy
    phi_ana = x * ox ** 3 * prod
    lam = np.zeros(shape)
    rhs = -2 * x * ox ** 3 - 6 * ox * (ox - x) * prod

    return phi_ana, rhs, lam, D_xx, D_yy

@pytest.fixture
def test1_setup(setup):
    grid_x, grid_y, phi, _, _ = setup
    return problem_1(grid_x.centers, grid_y.centers, phi.shape)

@pytest.fixture
def test1_setup_64(setup_64):
    grid_x, grid_y, phi, _, _ = setup_64
    return problem_1(grid_x.centers, grid_y.centers, phi.shape)

@pytest.fixture
def test1_setup_64_128(setup_64_128):
    grid_x, grid_y, phi, _, _ = setup_64_128
    return problem_1(grid_x.centers, grid_y.centers, phi.shape)

def test1(setup, test1_setup, grad_func_solve):
    grid_x, grid_y, phi, acceptable_error_abs, acceptable_error_rel = setup
    phi_ana, rhs, lam, D_xx, D_yy = test1_setup
    gsolve = grad_func_solve

    curr_time = time.time()
    phi_res, iterations = solve_2d_simple(grid_x=grid_x, grid_y=grid_y, phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy)
    print("solve took {} s".format(time.time() - curr_time))

    assert iterations == 13
    assert pytest.approx(1.010315424011278e-06, abs=acceptable_error_abs, rel=acceptable_error_rel) == rmse(phi_ana, phi_res)

    if check_autodiff_fwd:
        curr_time = time.time()
        check_grads(gsolve, (grid_x, grid_y, phi, rhs, lam, D_xx, D_yy), order=1, modes=["fwd"])
        print("check_grads(gsolve, ...) took {} s".format(time.time() - curr_time))

    if check_f:
        phi_res, iterations = solve_2d_fcycle_simple(grid_x=grid_x, grid_y=grid_y, phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy)
        assert iterations <= 8
        assert pytest.approx(1.010315414037897e-06, rel=acceptable_error_rel) == rmse(phi_ana, phi_res)

    if check_w:
        phi_res, iterations = solve_2d(gamma=2, grid_x=grid_x, grid_y=grid_y, phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy)
        assert iterations <= 13
        assert pytest.approx(1.010315424011278e-06, rel=acceptable_error_rel) == rmse(phi_ana, phi_res)

    if check_fixed:
        curr_time = time.time()
        phi_res_fixed, iterations_fixed = solve_2d_fixed_simple(iters=13, grid_x=grid_x, grid_y=grid_y, phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy)
        print("solve_fixed took {} s".format(time.time() - curr_time))
        assert iterations_fixed == 13
        assert pytest.approx(1.010315424011278e-06, abs=acceptable_error_abs, rel=acceptable_error_rel) == rmse(phi_ana, phi_res_fixed)

        if check_f:
            phi_res, iterations = solve_2d_fixed_fcycle_simple(iters=13, grid_x=grid_x, grid_y=grid_y, phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy)
            assert iterations <= 13
            assert pytest.approx(1.010315424011278e-06, rel=acceptable_error_rel) == rmse(phi_ana, phi_res)

        if check_autodiff_rev:
            def gsolve_rev(grid_x, grid_y, phi, rhs, lam, D_xx, D_yy, D_xy=None):
                phi_res, _ = solve_2d_fixed_simple(iters=13, grid_x=grid_x, grid_y=grid_y, phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy, D_xy=D_xy)
                return phi_res

            curr_time = time.time()
            check_grads(gsolve_rev, (grid_x, grid_y, phi, rhs, lam, D_xx, D_yy), order=1, modes=["rev"])
            print("check_grads(gsolve_rev, ...) took {} s".format(time.time() - curr_time))

def test1_64(setup_64, test1_setup_64, grad_func_solve_64):
    grid_x, grid_y, phi, acceptable_error_abs, acceptable_error_rel = setup_64
    phi_ana, rhs, lam, D_xx, D_yy = test1_setup_64
    gsolve = grad_func_solve_64

    curr_time = time.time()
    phi_res, iterations = solve_2d_simple(grid_x=grid_x, grid_y=grid_y, phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy)
    print("solve took {} s".format(time.time() - curr_time))

    assert iterations == 13
    assert pytest.approx(4.010172727197344e-06, abs=acceptable_error_abs, rel=acceptable_error_rel) == rmse(phi_ana, phi_res)

    if check_autodiff_fwd:
        curr_time = time.time()
        check_grads(gsolve, (grid_x, grid_y, phi, rhs, lam, D_xx, D_yy), order=1, modes=["fwd"])
        print("check_grads(gsolve, ...) took {} s".format(time.time() - curr_time))

def test1_64_128(setup_64_128, test1_setup_64_128, grad_func_solve_64_128):
    grid_x, grid_y, phi, acceptable_error_abs, acceptable_error_rel = setup_64_128
    phi_ana, rhs, lam, D_xx, D_yy = test1_setup_64_128
    gsolve = grad_func_solve_64_128

    curr_time = time.time()
    phi_res, iterations = solve_2d_simple(grid_x=grid_x, grid_y=grid_y, phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy)
    print("solve took {} s".format(time.time() - curr_time))

    assert iterations == 27
    assert pytest.approx(4.025383882544280e-06, abs=acceptable_error_abs, rel=acceptable_error_rel) == rmse(phi_ana, phi_res)

    if check_autodiff_fwd:
        curr_time = time.time()
        check_grads(gsolve, (grid_x, grid_y, phi, rhs, lam, D_xx, D_yy), order=1, modes=["fwd"])
        print("check_grads(gsolve, ...) took {} s".format(time.time() - curr_time))

def problem_2(x_cen, y_cen, shape):
    D_xx = 1.0
    D_yy = 1.0
    sy = np.sin(np.pi * y_cen)[:, np.newaxis]
    sx = np.sin(np.pi * x_cen)[np.newaxis]
    phi_ana = sx * sy
    lam = np.zeros(shape)
    rhs = -(np.pi ** 2) * (D_xx + D_yy) * phi_ana

    return phi_ana, rhs, lam, D_xx, D_yy

@pytest.fixture
def test2_setup(setup):
    grid_x, grid_y, phi, _, _ = setup
    return problem_2(grid_x.centers, grid_y.centers, phi.shape)

@pytest.fixture
def test2_setup_64(setup_64):
    grid_x, grid_y, phi, _, _ = setup_64
    return problem_2(grid_x.centers, grid_y.centers, phi.shape)

@pytest.fixture
def test2_setup_64_128(setup_64_128):
    grid_x, grid_y, phi, _, _ = setup_64_128
    return problem_2(grid_x.centers, grid_y.centers, phi.shape)

def test2(setup, test2_setup, grad_func_solve):
    grid_x, grid_y, phi, acceptable_error_abs, acceptable_error_rel = setup
    phi_ana, rhs, lam, D_xx, D_yy = test2_setup
    gsolve = grad_func_solve

    curr_time = time.time()
    phi_res, iterations = solve_2d_simple(grid_x=grid_x, grid_y=grid_y, phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy)
    print("solve took {} s".format(time.time() - curr_time))

    assert iterations == 14
    assert pytest.approx(2.490588070275737e-05, abs=acceptable_error_abs, rel=acceptable_error_rel) == rmse(phi_ana, phi_res)

    if check_autodiff_fwd:
        curr_time = time.time()
        check_grads(gsolve, (grid_x, grid_y, phi, rhs, lam, D_xx, D_yy), order=1, modes=["fwd"])
        print("check_grads(gsolve, ...) took {} s".format(time.time() - curr_time))

def test2_64(setup_64, test2_setup_64, grad_func_solve_64):
    grid_x, grid_y, phi, acceptable_error_abs, acceptable_error_rel = setup_64
    phi_ana, rhs, lam, D_xx, D_yy = test2_setup_64
    gsolve = grad_func_solve_64

    curr_time = time.time()
    phi_res, iterations = solve_2d_simple(grid_x=grid_x, grid_y=grid_y, phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy)
    print("solve took {} s".format(time.time() - curr_time))

    assert iterations == 14
    assert pytest.approx(9.886612163935591e-05, abs=acceptable_error_abs, rel=acceptable_error_rel) == rmse(phi_ana, phi_res)

    if check_autodiff_fwd:
        curr_time = time.time()
        check_grads(gsolve, (grid_x, grid_y, phi, rhs, lam, D_xx, D_yy), order=1, modes=["fwd"])
        print("check_grads(gsolve, ...) took {} s".format(time.time() - curr_time))

def test2_64_128(setup_64_128, test2_setup_64_128, grad_func_solve_64_128):
    grid_x, grid_y, phi, acceptable_error_abs, acceptable_error_rel = setup_64_128
    phi_ana, rhs, lam, D_xx, D_yy = test2_setup_64_128
    gsolve = grad_func_solve_64_128

    curr_time = time.time()
    phi_res, iterations = solve_2d_simple(grid_x=grid_x, grid_y=grid_y, phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy)
    print("solve took {} s".format(time.time() - curr_time))

    assert iterations == 28
    assert pytest.approx(6.202644125853758e-05, abs=acceptable_error_abs, rel=acceptable_error_rel) == rmse(phi_ana, phi_res)

    if check_autodiff_fwd:
        curr_time = time.time()
        check_grads(gsolve, (grid_x, grid_y, phi, rhs, lam, D_xx, D_yy), order=1, modes=["fwd"])
        print("check_grads(gsolve, ...) took {} s".format(time.time() - curr_time))

def problem_3(x_cen, y_cen, shape):
    sy = np.sin(np.pi * y_cen)[:, np.newaxis]
    x = x_cen[np.newaxis]
    sx = np.sin(np.pi * x)
    cx = np.cos(np.pi * x)
    D_xx = np.full(shape, x)
    D_yy = np.full(shape, x)
    phi_ana = sx * sy
    lam = np.zeros(shape)
    rhs = np.pi * cx * sy - 2 * np.pi ** 2 * x * phi_ana

    return phi_ana, rhs, lam, D_xx, D_yy

@pytest.fixture
def test3_setup(setup):
    grid_x, grid_y, phi, _, _ = setup
    return problem_3(grid_x.centers, grid_y.centers, phi.shape)

@pytest.fixture
def test3_setup_64(setup_64):
    grid_x, grid_y, phi, _, _ = setup_64
    return problem_3(grid_x.centers, grid_y.centers, phi.shape)

@pytest.fixture
def test3_setup_64_128(setup_64_128):
    grid_x, grid_y, phi, _, _ = setup_64_128
    return problem_3(grid_x.centers, grid_y.centers, phi.shape)

def test3(setup, test3_setup, grad_func_solve):
    grid_x, grid_y, phi, acceptable_error_abs, acceptable_error_rel = setup
    phi_ana, rhs, lam, D_xx, D_yy = test3_setup
    gsolve = grad_func_solve

    curr_time = time.time()
    phi_res, iterations = solve_2d_simple(grid_x=grid_x, grid_y=grid_y, phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy)
    print("solve took {} s".format(time.time() - curr_time))

    assert iterations == 41
    assert pytest.approx(2.197736631958706e-05, abs=acceptable_error_abs, rel=acceptable_error_rel) == rmse(phi_ana, phi_res)

    if check_autodiff_fwd:
        curr_time = time.time()
        check_grads(gsolve, (grid_x, grid_y, phi, rhs, lam, D_xx, D_yy), order=1, modes=["fwd"])
        print("check_grads(gsolve, ...) took {} s".format(time.time() - curr_time))

def test3_64(setup_64, test3_setup_64, grad_func_solve_64):
    grid_x, grid_y, phi, acceptable_error_abs, acceptable_error_rel = setup_64
    phi_ana, rhs, lam, D_xx, D_yy = test3_setup_64
    gsolve = grad_func_solve_64

    curr_time = time.time()
    phi_res, iterations = solve_2d_simple(grid_x=grid_x, grid_y=grid_y, phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy)
    print("solve took {} s".format(time.time() - curr_time))

    assert iterations == 35
    assert pytest.approx(8.725944838558505e-05, abs=acceptable_error_abs, rel=acceptable_error_rel) == rmse(phi_ana, phi_res)

    if check_autodiff_fwd:
        curr_time = time.time()
        check_grads(gsolve, (grid_x, grid_y, phi, rhs, lam, D_xx, D_yy), order=1, modes=["fwd"])
        print("check_grads(gsolve, ...) took {} s".format(time.time() - curr_time))

def test3_64_128(setup_64_128, test3_setup_64_128, grad_func_solve_64_128):
    grid_x, grid_y, phi, acceptable_error_abs, acceptable_error_rel = setup_64_128
    phi_ana, rhs, lam, D_xx, D_yy = test3_setup_64_128
    gsolve = grad_func_solve_64_128

    curr_time = time.time()
    phi_res, iterations = solve_2d_simple(grid_x=grid_x, grid_y=grid_y, phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy)
    print("solve took {} s".format(time.time() - curr_time))

    assert iterations == 77
    assert pytest.approx(5.910207160483823e-05, abs=acceptable_error_abs, rel=acceptable_error_rel) == rmse(phi_ana, phi_res)

    if check_autodiff_fwd:
        curr_time = time.time()
        check_grads(gsolve, (grid_x, grid_y, phi, rhs, lam, D_xx, D_yy), order=1, modes=["fwd"])
        print("check_grads(gsolve, ...) took {} s".format(time.time() - curr_time))

def problem_4(x_cen, y_cen, shape):
    y = y_cen[:, np.newaxis]
    sy = np.sin(np.pi * y)
    x = x_cen[np.newaxis]
    sx = np.sin(np.pi * x)
    D_xx = np.full(shape, x)
    D_yy = np.zeros(shape) + y
    phi_ana = sx * sy
    lam = 0.2 * x * y ** 2
    xysum = x + y
    rhs = np.pi * np.sin(np.pi * xysum) - (xysum * np.pi ** 2 + lam) * phi_ana

    return phi_ana, rhs, lam, D_xx, D_yy

@pytest.fixture
def test4_setup(setup):
    grid_x, grid_y, phi, _, _ = setup
    return problem_4(grid_x.centers, grid_y.centers, phi.shape)

@pytest.fixture
def test4_setup_64(setup_64):
    grid_x, grid_y, phi, _, _ = setup_64
    return problem_4(grid_x.centers, grid_y.centers, phi.shape)

@pytest.fixture
def test4_setup_64_128(setup_64_128):
    grid_x, grid_y, phi, _, _ = setup_64_128
    return problem_4(grid_x.centers, grid_y.centers, phi.shape)

def test4(setup, test4_setup, grad_func_solve):
    grid_x, grid_y, phi, acceptable_error_abs, acceptable_error_rel = setup
    phi_ana, rhs, lam, D_xx, D_yy = test4_setup
    gsolve = grad_func_solve

    curr_time = time.time()
    phi_res, iterations = solve_2d_simple(grid_x=grid_x, grid_y=grid_y, phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy)
    print("solve took {} s".format(time.time() - curr_time))

    assert iterations == 468
    assert pytest.approx(2.179081815661214e-05, abs=acceptable_error_abs, rel=acceptable_error_rel) == rmse(phi_ana, phi_res)

    if check_autodiff_fwd:
        curr_time = time.time()
        check_grads(gsolve, (grid_x, grid_y, phi, rhs, lam, D_xx, D_yy), order=1, modes=["fwd"])
        print("check_grads(gsolve, ...) took {} s".format(time.time() - curr_time))

def test4_64(setup_64, test4_setup_64, grad_func_solve_64):
    grid_x, grid_y, phi, acceptable_error_abs, acceptable_error_rel = setup_64
    phi_ana, rhs, lam, D_xx, D_yy = test4_setup_64
    gsolve = grad_func_solve_64

    curr_time = time.time()
    phi_res, iterations = solve_2d_simple(grid_x=grid_x, grid_y=grid_y, phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy)
    print("solve took {} s".format(time.time() - curr_time))

    assert iterations == 228
    assert pytest.approx(8.596967777404521e-05, abs=acceptable_error_abs, rel=acceptable_error_rel) == rmse(phi_ana, phi_res)

    if check_autodiff_fwd:
        curr_time = time.time()
        check_grads(gsolve, (grid_x, grid_y, phi, rhs, lam, D_xx, D_yy), order=1, modes=["fwd"])
        print("check_grads(gsolve, ...) took {} s".format(time.time() - curr_time))

def test4_64_128(setup_64_128, test4_setup_64_128, grad_func_solve_64_128):
    grid_x, grid_y, phi, acceptable_error_abs, acceptable_error_rel = setup_64_128
    phi_ana, rhs, lam, D_xx, D_yy = test4_setup_64_128
    gsolve = grad_func_solve_64_128

    curr_time = time.time()
    phi_res, iterations = solve_2d_simple(grid_x=grid_x, grid_y=grid_y, phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy)
    print("solve took {} s".format(time.time() - curr_time))

    assert iterations == 796
    assert pytest.approx(5.979198934671856e-05, abs=acceptable_error_abs, rel=acceptable_error_rel) == rmse(phi_ana, phi_res)

    if check_autodiff_fwd:
        curr_time = time.time()
        check_grads(gsolve, (grid_x, grid_y, phi, rhs, lam, D_xx, D_yy), order=1, modes=["fwd"])
        print("check_grads(gsolve, ...) took {} s".format(time.time() - curr_time))

def problem_5(x_cen, y_cen, shape):
    D_xx = 1.0
    D_yy = 1e-2
    x = x_cen[np.newaxis]
    y = y_cen[:, np.newaxis]
    sx = np.sin(np.pi * x)
    sy = np.sin(np.pi * y)
    phi_ana = sx * sy
    lam = 0.2 * x * y ** 2
    rhs = -((np.pi ** 2) * (D_xx + D_yy) + lam) * phi_ana

    return phi_ana, rhs, lam, D_xx, D_yy

@pytest.fixture
def test5_setup(setup):
    grid_x, grid_y, phi, _, _ = setup
    return problem_5(grid_x.centers, grid_y.centers, phi.shape)

@pytest.fixture
def test5_setup_64(setup_64):
    grid_x, grid_y, phi, _, _ = setup_64
    return problem_5(grid_x.centers, grid_y.centers, phi.shape)

@pytest.fixture
def test5_setup_64_128(setup_64_128):
    grid_x, grid_y, phi, _, _ = setup_64_128
    return problem_5(grid_x.centers, grid_y.centers, phi.shape)

def test5(setup, test5_setup, grad_func_solve):
    grid_x, grid_y, phi, acceptable_error_abs, acceptable_error_rel = setup
    phi_ana, rhs, lam, D_xx, D_yy = test5_setup
    gsolve = grad_func_solve

    curr_time = time.time()
    phi_res, iterations = solve_2d_simple(grid_x=grid_x, grid_y=grid_y, phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy)
    print("solve took {} s".format(time.time() - curr_time))

    assert iterations == 208
    assert pytest.approx(2.483558318071642e-05, abs=acceptable_error_abs, rel=acceptable_error_rel) == rmse(phi_ana, phi_res)

    if check_autodiff_fwd:
        curr_time = time.time()
        check_grads(gsolve, (grid_x, grid_y, phi, rhs, lam, D_xx, D_yy), order=1, modes=["fwd"])
        print("check_grads(gsolve, ...) took {} s".format(time.time() - curr_time))

def test5_64(setup_64, test5_setup_64, grad_func_solve_64):
    grid_x, grid_y, phi, acceptable_error_abs, acceptable_error_rel = setup_64
    phi_ana, rhs, lam, D_xx, D_yy = test5_setup_64
    gsolve = grad_func_solve_64

    curr_time = time.time()
    phi_res, iterations = solve_2d_simple(grid_x=grid_x, grid_y=grid_y, phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy)
    print("solve took {} s".format(time.time() - curr_time))

    assert iterations == 205
    assert pytest.approx(9.858702741615382e-05, abs=acceptable_error_abs, rel=acceptable_error_rel) == rmse(phi_ana, phi_res)

    if check_autodiff_fwd:
        curr_time = time.time()
        check_grads(gsolve, (grid_x, grid_y, phi, rhs, lam, D_xx, D_yy), order=1, modes=["fwd"])
        print("check_grads(gsolve, ...) took {} s".format(time.time() - curr_time))

def test5_64_128(setup_64_128, test5_setup_64_128, grad_func_solve_64_128):
    grid_x, grid_y, phi, acceptable_error_abs, acceptable_error_rel = setup_64_128
    phi_ana, rhs, lam, D_xx, D_yy = test5_setup_64_128
    gsolve = grad_func_solve_64_128

    curr_time = time.time()
    phi_res, iterations = solve_2d_simple(grid_x=grid_x, grid_y=grid_y, phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy)
    print("solve took {} s".format(time.time() - curr_time))

    assert iterations == 107
    assert pytest.approx(9.823336470021998e-05, abs=acceptable_error_abs, rel=acceptable_error_rel) == rmse(phi_ana, phi_res)

    if check_autodiff_fwd:
        curr_time = time.time()
        check_grads(gsolve, (grid_x, grid_y, phi, rhs, lam, D_xx, D_yy), order=1, modes=["fwd"])
        print("check_grads(gsolve, ...) took {} s".format(time.time() - curr_time))

def problem_7(x_cen, y_cen, shape):
    c = 0.1
    y = y_cen[:, np.newaxis]
    sy = np.sin(np.pi * y)
    cy = np.cos(np.pi * y)
    x = x_cen[np.newaxis]
    sx = np.sin(np.pi * x)
    cx = np.cos(np.pi * x)
    D_xx = np.full(shape, x)
    D_yy = np.zeros(shape) + y
    D_xy = np.full(shape, c * x)
    phi_ana = sx * sy
    lam = 0.2 * x * y ** 2
    rhs = (1. + c) * np.pi * sx * cy + np.pi * cx * sy + c * 2. * x * np.pi ** 2 * cx * cy - ((x + y) * np.pi ** 2 + lam) * phi_ana

    return phi_ana, rhs, lam, D_xx, D_yy, D_xy

@pytest.fixture
def test7_setup(setup):
    grid_x, grid_y, phi, _, _ = setup
    return problem_7(grid_x.centers, grid_y.centers, phi.shape)

@pytest.fixture
def test7_setup_64(setup_64):
    grid_x, grid_y, phi, _, _ = setup_64
    return problem_7(grid_x.centers, grid_y.centers, phi.shape)

@pytest.fixture
def test7_setup_64_128(setup_64_128):
    grid_x, grid_y, phi, _, _ = setup_64_128
    return problem_7(grid_x.centers, grid_y.centers, phi.shape)

def test7(setup, test7_setup, grad_func_solve):
    grid_x, grid_y, phi, acceptable_error_abs, acceptable_error_rel = setup
    phi_ana, rhs, lam, D_xx, D_yy, D_xy = test7_setup
    gsolve = grad_func_solve

    curr_time = time.time()
    phi_res, iterations = solve_2d_simple(grid_x=grid_x, grid_y=grid_y, phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy, D_xy=D_xy)
    print("solve took {} s".format(time.time() - curr_time))

    assert iterations == 466
    assert pytest.approx(2.291672360185362e-05, abs=acceptable_error_abs, rel=acceptable_error_rel) == rmse(phi_ana, phi_res)

    if check_autodiff_fwd:
        curr_time = time.time()
        check_grads(gsolve, (grid_x, grid_y, phi, rhs, lam, D_xx, D_yy, D_xy), order=1, modes=["fwd"])
        print("check_grads(gsolve, ...) took {} s".format(time.time() - curr_time))

def test7_64(setup_64, test7_setup_64, grad_func_solve_64):
    grid_x, grid_y, phi, acceptable_error_abs, acceptable_error_rel = setup_64
    phi_ana, rhs, lam, D_xx, D_yy, D_xy = test7_setup_64
    gsolve = grad_func_solve_64

    curr_time = time.time()
    phi_res, iterations = solve_2d_simple(grid_x=grid_x, grid_y=grid_y, phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy, D_xy=D_xy)
    print("solve took {} s".format(time.time() - curr_time))

    assert iterations == 226
    assert pytest.approx(8.920084594712214e-05, abs=acceptable_error_abs, rel=acceptable_error_rel) == rmse(phi_ana, phi_res)

    if check_autodiff_fwd:
        curr_time = time.time()
        check_grads(gsolve, (grid_x, grid_y, phi, rhs, lam, D_xx, D_yy, D_xy), order=1, modes=["fwd"])
        print("check_grads(gsolve, ...) took {} s".format(time.time() - curr_time))

def test7_64_128(setup_64_128, test7_setup_64_128, grad_func_solve_64_128):
    grid_x, grid_y, phi, acceptable_error_abs, acceptable_error_rel = setup_64_128
    phi_ana, rhs, lam, D_xx, D_yy, D_xy = test7_setup_64_128
    gsolve = grad_func_solve_64_128

    curr_time = time.time()
    phi_res, iterations = solve_2d_simple(grid_x=grid_x, grid_y=grid_y, phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy, D_xy=D_xy)
    print("solve took {} s".format(time.time() - curr_time))

    assert iterations == 794
    assert pytest.approx(6.078926956740372e-05, abs=acceptable_error_abs, rel=acceptable_error_rel) == rmse(phi_ana, phi_res)

    if check_autodiff_fwd:
        curr_time = time.time()
        check_grads(gsolve, (grid_x, grid_y, phi, rhs, lam, D_xx, D_yy, D_xy), order=1, modes=["fwd"])
        print("check_grads(gsolve, ...) took {} s".format(time.time() - curr_time))

def problem_9(x_cen, y_cen, shape):
    c = 0.5
    y = y_cen[:, np.newaxis]
    sy = np.sin(np.pi * y)
    cy = np.cos(np.pi * y)
    x = x_cen[np.newaxis]
    sx = np.sin(np.pi * x)
    cx = np.cos(np.pi * x)
    D_xx = np.full(shape, x)
    D_yy = np.zeros(shape) + y
    D_xy = c * x * y
    phi_ana = sx * sy
    lam = 0.2 * x * y ** 2
    rhs = (1 + c * x) * np.pi * cx * sy + (1 + c * y) * np.pi * sx * cy + c * 2 * x * y * np.pi ** 2 * cx * cy - ((x + y) * np.pi ** 2 + lam) * phi_ana

    return phi_ana, rhs, lam, D_xx, D_yy, D_xy

@pytest.fixture
def test9_setup(setup):
    grid_x, grid_y, phi, _, _ = setup
    return problem_9(grid_x.centers, grid_y.centers, phi.shape)

@pytest.fixture
def test9_setup_64(setup_64):
    grid_x, grid_y, phi, _, _ = setup_64
    return problem_9(grid_x.centers, grid_y.centers, phi.shape)

@pytest.fixture
def test9_setup_64_128(setup_64_128):
    grid_x, grid_y, phi, _, _ = setup_64_128
    return problem_9(grid_x.centers, grid_y.centers, phi.shape)

def test9(setup, test9_setup, grad_func_solve):
    grid_x, grid_y, phi, acceptable_error_abs, acceptable_error_rel = setup
    phi_ana, rhs, lam, D_xx, D_yy, D_xy = test9_setup
    gsolve = grad_func_solve

    curr_time = time.time()
    phi_res, iterations = solve_2d_simple(grid_x=grid_x, grid_y=grid_y, phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy, D_xy=D_xy)
    print("solve took {} s".format(time.time() - curr_time))

    assert iterations == 462
    assert pytest.approx(2.719754537531388e-05, abs=acceptable_error_abs, rel=acceptable_error_rel) == rmse(phi_ana, phi_res)

    if check_autodiff_fwd:
        curr_time = time.time()
        check_grads(gsolve, (grid_x, grid_y, phi, rhs, lam, D_xx, D_yy, D_xy), order=1, modes=["fwd"])
        print("check_grads(gsolve, ...) took {} s".format(time.time() - curr_time))

def test9_64(setup_64, test9_setup_64, grad_func_solve_64):
    grid_x, grid_y, phi, acceptable_error_abs, acceptable_error_rel = setup_64
    phi_ana, rhs, lam, D_xx, D_yy, D_xy = test9_setup_64
    gsolve = grad_func_solve_64

    curr_time = time.time()
    phi_res, iterations = solve_2d_simple(grid_x=grid_x, grid_y=grid_y, phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy, D_xy=D_xy)
    print("solve took {} s".format(time.time() - curr_time))

    assert iterations == 224
    assert pytest.approx(1.073853922527805e-04, abs=acceptable_error_abs, rel=acceptable_error_rel) == rmse(phi_ana, phi_res)

    if check_autodiff_fwd:
        curr_time = time.time()
        check_grads(gsolve, (grid_x, grid_y, phi, rhs, lam, D_xx, D_yy, D_xy), order=1, modes=["fwd"])
        print("check_grads(gsolve, ...) took {} s".format(time.time() - curr_time))

def test9_64_128(setup_64_128, test9_setup_64_128, grad_func_solve_64_128):
    grid_x, grid_y, phi, acceptable_error_abs, acceptable_error_rel = setup_64_128
    phi_ana, rhs, lam, D_xx, D_yy, D_xy = test9_setup_64_128
    gsolve = grad_func_solve_64_128

    curr_time = time.time()
    phi_res, iterations = solve_2d_simple(grid_x=grid_x, grid_y=grid_y, phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy, D_xy=D_xy)
    print("solve took {} s".format(time.time() - curr_time))

    assert iterations == 785
    assert pytest.approx(7.061117869397386e-05, abs=acceptable_error_abs, rel=acceptable_error_rel) == rmse(phi_ana, phi_res)

    if check_autodiff_fwd:
        curr_time = time.time()
        check_grads(gsolve, (grid_x, grid_y, phi, rhs, lam, D_xx, D_yy, D_xy), order=1, modes=["fwd"])
        print("check_grads(gsolve, ...) took {} s".format(time.time() - curr_time))

def problem_10(x_cen, y_cen, shape):
    c = 0.1
    y = y_cen[:, np.newaxis]
    sy = np.sin(np.pi * y)
    cy = np.cos(np.pi * y)
    x = x_cen[np.newaxis]
    sx = np.sin(np.pi * x)
    cx = np.cos(np.pi * x)
    D_xx = np.zeros(shape) + y
    D_yy = np.zeros(shape) + y
    D_xy = c * x * y
    phi_ana = sx * sy
    lam = 0.2 * x * y ** 2
    rhs = np.pi * (1 + c * y) * sx * cy + c * x * np.pi * cx * sy + c * 2 * x * y * np.pi ** 2 * cx * cy - (2 * y * np.pi ** 2 + lam) * phi_ana

    return phi_ana, rhs, lam, D_xx, D_yy, D_xy

@pytest.fixture
def test10_setup(setup):
    grid_x, grid_y, phi, _, _ = setup
    return problem_10(grid_x.centers, grid_y.centers, phi.shape)

@pytest.fixture
def test10_setup_64(setup_64):
    grid_x, grid_y, phi, _, _ = setup_64
    return problem_10(grid_x.centers, grid_y.centers, phi.shape)

@pytest.fixture
def test10_setup_64_128(setup_64_128):
    grid_x, grid_y, phi, _, _ = setup_64_128
    return problem_10(grid_x.centers, grid_y.centers, phi.shape)

def test10(setup, test10_setup, grad_func_solve):
    grid_x, grid_y, phi, acceptable_error_abs, acceptable_error_rel = setup
    phi_ana, rhs, lam, D_xx, D_yy, D_xy = test10_setup
    gsolve = grad_func_solve

    curr_time = time.time()
    phi_res, iterations = solve_2d_simple(grid_x=grid_x, grid_y=grid_y, phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy, D_xy=D_xy)
    print("solve took {} s".format(time.time() - curr_time))

    assert iterations == 41
    assert pytest.approx(2.317202758890556e-05, abs=acceptable_error_abs, rel=acceptable_error_rel) == rmse(phi_ana, phi_res)

    if check_autodiff_fwd:
        curr_time = time.time()
        check_grads(gsolve, (grid_x, grid_y, phi, rhs, lam, D_xx, D_yy, D_xy), order=1, modes=["fwd"])
        print("check_grads(gsolve, ...) took {} s".format(time.time() - curr_time))

def test10_64(setup_64, test10_setup_64, grad_func_solve_64):
    grid_x, grid_y, phi, acceptable_error_abs, acceptable_error_rel = setup_64
    phi_ana, rhs, lam, D_xx, D_yy, D_xy = test10_setup_64
    gsolve = grad_func_solve_64

    curr_time = time.time()
    phi_res, iterations = solve_2d_simple(grid_x=grid_x, grid_y=grid_y, phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy, D_xy=D_xy)
    print("solve took {} s".format(time.time() - curr_time))

    assert iterations == 34
    assert pytest.approx(9.205560538691322e-05, abs=acceptable_error_abs, rel=acceptable_error_rel) == rmse(phi_ana, phi_res)

    if check_autodiff_fwd:
        curr_time = time.time()
        check_grads(gsolve, (grid_x, grid_y, phi, rhs, lam, D_xx, D_yy, D_xy), order=1, modes=["fwd"])
        print("check_grads(gsolve, ...) took {} s".format(time.time() - curr_time))

def test10_64_128(setup_64_128, test10_setup_64_128, grad_func_solve_64_128):
    grid_x, grid_y, phi, acceptable_error_abs, acceptable_error_rel = setup_64_128
    phi_ana, rhs, lam, D_xx, D_yy, D_xy = test10_setup_64_128
    gsolve = grad_func_solve_64_128

    curr_time = time.time()
    phi_res, iterations = solve_2d_simple(grid_x=grid_x, grid_y=grid_y, phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy, D_xy=D_xy)
    print("solve took {} s".format(time.time() - curr_time))

    assert iterations == 33
    assert pytest.approx(6.674092567109719e-05, abs=acceptable_error_abs, rel=acceptable_error_rel) == rmse(phi_ana, phi_res)

    if check_autodiff_fwd:
        curr_time = time.time()
        check_grads(gsolve, (grid_x, grid_y, phi, rhs, lam, D_xx, D_yy, D_xy), order=1, modes=["fwd"])
        print("check_grads(gsolve, ...) took {} s".format(time.time() - curr_time))

def problem_11(x_cen, y_cen, shape):
    dpar = 1.
    dperp = 0.1
    y = y_cen[:, np.newaxis]
    sy = np.sin(np.pi * y)
    cy = np.cos(np.pi * y)
    x = x_cen[np.newaxis]
    sx = np.sin(np.pi * x)
    cx = np.cos(np.pi * x)
    phi_ana = sx * sy
    lam = 0.2 * x * y ** 2
    angle = np.arctan2(y, x)
    sangle = np.sin(angle)
    cangle = np.cos(angle)
    scangle = sangle * cangle
    sqsangle = sangle ** 2
    sqcangle = cangle ** 2
    D_xx = dpar * sqsangle + dperp * sqcangle
    D_yy = dpar * sqcangle + dperp * sqsangle
    D_xy = (dperp - dpar) * scangle
    sqrad = x ** 2 + y ** 2
    dphi_dx = -y / sqrad
    dphi_dy = x / sqrad
    dD_xx_dx = 2 * (dpar - dperp) * scangle * dphi_dx
    dD_yy_dy = 2 * (dperp - dpar) * scangle * dphi_dy
    dD_xy_dx = (dperp - dpar) * (sqcangle - sqsangle) * dphi_dx
    dD_xy_dy = (dperp - dpar) * (sqcangle - sqsangle) * dphi_dy
    rhs = (dD_yy_dy + dD_xy_dx) * np.pi * sx * cy + (dD_xx_dx + dD_xy_dy) * np.pi * cx * sy + 2 * D_xy * np.pi ** 2 * cx * cy - ((D_xx + D_yy) * np.pi ** 2 + lam) * phi_ana

    return phi_ana, rhs, lam, D_xx, D_yy, D_xy

@pytest.fixture
def test11_setup(setup):
    grid_x, grid_y, phi, _, _ = setup
    return problem_11(grid_x.centers, grid_y.centers, phi.shape)

@pytest.fixture
def test11_setup_64(setup_64):
    grid_x, grid_y, phi, _, _ = setup_64
    return problem_11(grid_x.centers, grid_y.centers, phi.shape)

@pytest.fixture
def test11_setup_64_128(setup_64_128):
    grid_x, grid_y, phi, _, _ = setup_64_128
    return problem_11(grid_x.centers, grid_y.centers, phi.shape)

def test11(setup, test11_setup, grad_func_solve):
    grid_x, grid_y, phi, acceptable_error_abs, acceptable_error_rel = setup
    phi_ana, rhs, lam, D_xx, D_yy, D_xy = test11_setup
    gsolve = grad_func_solve

    curr_time = time.time()
    phi_res, iterations = solve_2d_simple(grid_x=grid_x, grid_y=grid_y, phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy, D_xy=D_xy)
    print("solve took {} s".format(time.time() - curr_time))

    assert iterations == 56
    assert pytest.approx(3.881907960699822e-05, abs=acceptable_error_abs, rel=acceptable_error_rel) == rmse(phi_ana, phi_res)

    if check_autodiff_fwd:
        curr_time = time.time()
        check_grads(gsolve, (grid_x, grid_y, phi, rhs, lam, D_xx, D_yy, D_xy), order=1, modes=["fwd"])
        print("check_grads(gsolve, ...) took {} s".format(time.time() - curr_time))

def test11_64(setup_64, test11_setup_64, grad_func_solve_64):
    grid_x, grid_y, phi, acceptable_error_abs, acceptable_error_rel = setup_64
    phi_ana, rhs, lam, D_xx, D_yy, D_xy = test11_setup_64
    gsolve = grad_func_solve_64

    curr_time = time.time()
    phi_res, iterations = solve_2d_simple(grid_x=grid_x, grid_y=grid_y, phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy, D_xy=D_xy)
    print("solve took {} s".format(time.time() - curr_time))

    assert iterations == 53
    assert pytest.approx(1.541913423109781e-04, abs=acceptable_error_abs, rel=acceptable_error_rel) == rmse(phi_ana, phi_res)

    if check_autodiff_fwd:
        curr_time = time.time()
        check_grads(gsolve, (grid_x, grid_y, phi, rhs, lam, D_xx, D_yy, D_xy), order=1, modes=["fwd"])
        print("check_grads(gsolve, ...) took {} s".format(time.time() - curr_time))

def test11_64_128(setup_64_128, test11_setup_64_128, grad_func_solve_64_128):
    grid_x, grid_y, phi, acceptable_error_abs, acceptable_error_rel = setup_64_128
    phi_ana, rhs, lam, D_xx, D_yy, D_xy = test11_setup_64_128
    gsolve = grad_func_solve_64_128

    curr_time = time.time()
    phi_res, iterations = solve_2d_simple(grid_x=grid_x, grid_y=grid_y, phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy, D_xy=D_xy)
    print("solve took {} s".format(time.time() - curr_time))

    assert iterations == 155
    assert pytest.approx(1.008028166210623e-04, abs=acceptable_error_abs, rel=acceptable_error_rel) == rmse(phi_ana, phi_res)

    if check_autodiff_fwd:
        curr_time = time.time()
        check_grads(gsolve, (grid_x, grid_y, phi, rhs, lam, D_xx, D_yy, D_xy), order=1, modes=["fwd"])
        print("check_grads(gsolve, ...) took {} s".format(time.time() - curr_time))

