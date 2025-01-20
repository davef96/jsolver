import jax
import jax.numpy as jnp
import numpy as np
#from jax.test_util import check_grads
from jsolver.grid_1d import grid_1D
from jsolver.solver_2d import solve_2d, solve_2d_simple, solve_2d_fixed_simple, solve_2d_fixed_fcycle_simple, solve_2d_fcycle_simple
import time
#import timeit
import argparse
import csv
import datetime
from functools import partial
#from memory_profiler import profile

#jax.config.update('jax_enable_x64', True) # use float64 as default (already done in 'solver_2d')
#jax.config.update("jax_debug_nans", True)
#jax.config.update("jax_enable_compilation_cache", False)
#jax.config.update('jax_compiler_enable_remat_pass', False)
#jax.config.update('jax_exec_time_optimization_effort', 1.0)
#jax.config.update('jax_memory_fitting_effort', 1.0)

# dictionary: (problem, resolution) --> (iterations, rmse)
reference = {
    (1, 8): (12, 0.0002316246921061767),
    (1, 16): (12, 6.133035189023003e-05),
    (1, 32): (13, 1.57976014514119e-05),
    (1, 64): (13, 4.010172727197344e-06),
    (1, 128): (13, 1.010315424011278e-06),
    (1, 256): (13, 2.535616941153208e-07),
    (1, 512): (13, 6.351400233907249e-08),
    (1, 1024): (13, 1.589399431109495e-08),
    (1, 2048): (13, 3.975474299592814e-09),
    (1, 4096): (13, 9.941673842082783e-10),
    (1, 8192): (13, 2.486839176535239e-10),
    (1, 16384): (13, 6.238011942809133e-11),
    (2, 8): (14, 0.005755887431937916),
    (2, 16): (14, 0.00151480679529332),
    (2, 32): (14, 0.0003896134202454288),
    (2, 64): (14, 9.886612163935591e-05),
    (2, 128): (14, 2.490588070275737e-05),
    (2, 256): (14, 6.250556427007311e-06),
    (2, 512): (14, 1.565676214226442e-06),
    (2, 1024): (15, 3.918003126171656e-07),
    (2, 2048): (15, 9.799762321654539e-08),
    (2, 4096): (16, 2.450534799661715e-08),
    (2, 8192): (17, 2.450534799661715e-08),
    (2, 16384): (18, 1.531246026997151e-09),
    (3, 8): (18, 0.005193228058673101),
    (3, 16): (23, 0.001348083634709043),
    (3, 32): (28, 0.0003446304408848213),
    (3, 64): (35, 8.725944838558505e-05),
    (3, 128): (41, 2.197736631958706e-05),
    (3, 256): (49, 5.51994606201339e-06),
    (3, 512): (57, 1.384419683655519e-06),
    (3, 1024): (66, 3.469509299140943e-07),
    (3, 2048): (75, 8.691253056633096e-08),
    (3, 4096): (86, 2.176612985358238e-08),
    (3, 8192): (96, 5.450236750721341e-09),
    (3, 16384): (108, 1.364670972822034e-09),
    (4, 8): (27, 0.005049861248634514),
    (4, 16): (53, 0.001312846780844815),
    (4, 32): (109, 0.0003373750023021041),
    (4, 64): (228, 0.0003373750023021041),
    (4, 128): (468, 2.179081815661214e-05),
    (4, 256): (945, 5.505458367096344e-06),
    (4, 512): (1866, 1.388133899887112e-06),
    (4, 1024): (3596, 3.495317379710846e-07),
    (4, 2048): (6750, 8.792993838917313e-08),
    (4, 4096): (12310, 2.210804333484445e-08),
    (4, 8192): (21735, 5.564187965275233e-09),
    (4, 16384): (36967, 1.426130423197298e-09),
    (5, 8): (49, 0.005739437566369961),
    (5, 16): (113, 0.001510517811999759),
    (5, 32): (177, 0.0003885129024460673),
    (5, 64): (205, 9.858702741615382e-05),
    (5, 128): (208, 2.483558318071642e-05),
    (5, 256): (208, 6.232914725194504e-06),
    (5, 512): (207, 1.561257178014406e-06),
    (5, 1024): (207, 3.906936721924083e-07),
    (5, 2048): (207, 9.771740020335442e-08),
    (5, 4096): (207, 2.44213476126886e-08),
    (5, 8192): (207, 6.051376737930316e-09),
    (5, 16384): (208, 1.295119749813762e-09),
    (7, 8): (27, 0.005005917496964761),
    (7, 16): (52, 0.001319879972701206),
    (7, 32): (108, 0.0003447378004132545),
    (7, 64): (226, 8.920084594712214e-05),
    (7, 128): (466, 2.291672360185362e-05),
    (7, 256): (942, 5.857772918705054e-06),
    (7, 512): (8717, 1.49195765181308e-06),
    (7, 1024): (52425, -np.nan),
    (7, 2048): (28023, -np.nan),
    (9, 8): (26, 0.00621838301543223),
    (9, 16): (51, 0.001634732335814828),
    (9, 32): (107, 0.0004213283875961785),
    (9, 64): (224, 0.0001073853922527805),
    (9, 128): (462, 2.719754537531388e-05),
    (9, 256): (933, 6.863482720569492e-06),
    (9, 512): (1842, 1.728338883644243e-06),
    (9, 1024): (3547, 4.346415270744447e-07),
    (9, 2048): (6650, 1.09206121436147e-07),
    (9, 4096): (12104, 2.742322009653933e-08),
    (9, 8192): (21317, 6.888761663168639e-09),
    (9, 16384): (36142, 1.747992590089799e-09),
    (10, 8): (18, 0.005459589664463751),
    (10, 16): (23, 0.001421873425127782),
    (10, 32): (28, 0.0003636870035134351),
    (10, 64): (34, 9.205560538691322e-05),
    (10, 128): (41, 2.317202758890556e-05),
    (10, 256): (48, 5.816297751795117e-06),
    (10, 512): (57, 1.457830953274442e-06),
    (10, 1024): (65, 3.651328433704817e-07),
    (10, 2048): (75, 9.141696491011185e-08),
    (10, 4096): (85, 2.288280396577327e-08),
    (10, 8192): (96, 5.727234089163642e-09),
    (10, 16384): (107, 1.434650008466684e-09),
    (11, 8): (25, 0.008356560823155142),
    (11, 16): (37, 0.002341340860301975),
    (11, 32): (47, 0.0006075975162552756),
    (11, 64): (53, 0.0001541913423108001),
    (11, 128): (56, 3.881907960678173e-05),
    (11, 256): (57, 9.739157998607151e-06),
    (11, 512): (57, 2.43921992579483e-06),
    (11, 1024): (58, 6.103735882786685e-07),
    (11, 2048): (58, 1.526658330561973e-07),
    (11, 4096): (58, 3.817561403959691e-08),
    (11, 8192): (58, 9.544985514841355e-09),
    (11, 16384): (60, 2.386464395180635e-09),
}

def bench(x_res, y_res, print_result=False, check_result=False, bench_solve=True, bench_jvp=True, bench_vjp=True, reps=10, reps_details=1, reps_util=1, reps_transfer=10, isolate=0, skip=0, cutoff=0, start=0, problem_type=None, export=False, save_result=False, file_prefix=None, file_suffix=None, scalar_lambda=True, fcycle=False, timestamp=False, jit_details=False, details_only=False, cpu_utilization=False, gpu_utilization=False, fixed=False, remat=False, force_iterations=0, smi=False, export_internals=False, bench_transfer=False, transfer_only=False, max_mem=False):
    def initialize_psutil():
        """Initializes the psutil library if available.

        Returns:
            error_msg: Error message if initialization fails, else None.
        """
        try:
            import psutil
            return None
        except ModuleNotFoundError:
            return "psutil is not installed."

    def initialize_threading():
        """Initializes the threading library if available.

        Returns:
            error_msg: Error message if initialization fails, else None.
        """
        try:
            import threading
            return None
        except ModuleNotFoundError:
            return "threading is not installed."

    def initialize_pynvml():
        """Initializes NVIDIA's pynvml library if available.

        Returns:
            handle: GPU handle if a GPU is detected, else None.
            error_msg: Error message if initialization fails, else None.
        """
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            return handle, None
        except ModuleNotFoundError:
            return None, "pynvml is not installed."
        except pynvml.NVMLError as e:
            return None, f"pynvml initialization failed: {str(e)}"

    def initialize_jax_smi():
        """Initializes the jax-smi library if available.

        Returns:
            error_msg: Error message if initialization fails, else None.
        """
        try:
            from jax_smi import initialise_tracking
            initialise_tracking()
            return None
        except ModuleNotFoundError:
            return "jax-smi is not installed."

    def monitor_cpu(interval, result):
        """Monitor CPU utilization."""
        usage = []
        i = 0
        while True:
            if i > 0:
                usage.append(psutil.cpu_percent(interval=None, percpu=False))
            i += 1
            time.sleep(interval)
            if not result["running"]:
                break
        result["usage"] = usage

    def monitor_gpu(handle, interval, result):
        """Monitor GPU utilization."""
        usage = []
        mem_usage = []
        mem = []
        i = 0
        while True:
            if i > 0:
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                mem_util = pynvml.nvmlDeviceGetUtilizationRates(handle).memory
                m = pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024 ** 3
                usage.append(gpu_util)
                mem_usage.append(mem_util)
                mem.append(m)
            i += 1
            time.sleep(interval)
            if not result["running"]:
                break
        result["usage"] = usage
        result["mem_usage"] = mem_usage
        result["mem"] = mem

    reference_usable = x_res == y_res and x_res >= 8 and x_res <= 16384
    str_fcycle = " (F-cycle)" if fcycle else ""

    # for export
    file_prefix = "" if file_prefix is None else file_prefix + "_"
    file_suffix = "" if file_suffix is None else "_" + file_suffix

    if reference_usable and fixed:
        print(f"mode: FIXED{str_fcycle}")
    elif force_iterations > 0:
        fixed = True
        print(f"mode: FIXED{str_fcycle}; enforcing {force_iterations} iterations")
    elif not reference_usable and fixed:
        raise ValueError("No reference value available. Cannot use mode FIXED.")
    else:
        print(f"mode: DYNAMIC{str_fcycle}")

    begin = time.time()

    if timestamp:
        print(f"benchmark start: {str(datetime.datetime.now())}")

    if not bench_solve and not bench_jvp and not bench_vjp:
        raise ValueError("No benchmark was selected. Nothing to do.")

    problem_ids = [ i for i in range(1, 12) ]
    problem_ids = filter(lambda i: not i == 6 and not i == 8, problem_ids)

    if start > 0:
        problem_ids = filter(lambda p: p >= start, problem_ids)
    if cutoff > 0:
        problem_ids = filter(lambda p: p <= cutoff, problem_ids)
    if isolate > 0:
        problem_ids = filter(lambda p: p == isolate, problem_ids)
    if skip > 0:
        problem_ids = filter(lambda p: not p == skip, problem_ids)
    if problem_type is not None:
        if problem_type == "c":
            problem_ids = filter(lambda p: p == 1 or p == 2 or p == 5, problem_ids)
        elif problem_type == "s":
            problem_ids = filter(lambda p: p == 3 or p == 4, problem_ids)
        elif problem_type == "o":
            problem_ids = filter(lambda p: p == 7 or p == 9 or p == 10 or p == 11, problem_ids)
        else:
            raise ValueError(f"Unknown problem type: {problem_type}")

    problem_ids = list(problem_ids)

    if not problem_ids:
        raise ValueError("No problems were selected. Nothing to do.")

    print(f"benchmarking problems {problem_ids} (filtering took {time.time() - begin} s)")

    curr_time = time.time()

    grid_x = grid_1D(mx=x_res, xb=0., xe=1., rim=1, grid_type=0, centered=True)
    grid_y = grid_1D(mx=y_res, xb=0., xe=1., rim=1, grid_type=0, centered=True)

    print("creating grid objects took {} s".format(time.time() - curr_time))

    curr_time = time.time()

    arr_shape = (y_res + 3, x_res + 3)
    phi = np.zeros(arr_shape)

    problems = []

    # test 1 (constant diffusion)
    if 1 in problem_ids:
        D_xx = 1.0
        D_yy = 1.0
        x = grid_x.centers[np.newaxis]
        y = grid_y.centers[:, np.newaxis]
        ox = 1. - x
        oy = 1. - y
        prod = y * oy
        phi_ana = x * ox ** 3 * prod
        lam = 0. if scalar_lambda else np.zeros(arr_shape)
        rhs = -2 * x * ox ** 3 - 6 * ox * (ox - x) * prod

        problems.append((1, phi_ana, phi, lam, rhs, D_xx, D_yy, None))

    # Expected (test 1):
    # Number of iterations: 13
    # rmse for 128 is 1.010315424011278e-06
    # result here: 1.010315424011275e-06

    # test 2 (constant diffusion)
    if 2 in problem_ids:
        D_xx = 1.0
        D_yy = 1.0
        sy = np.sin(np.pi * grid_y.centers)[:, np.newaxis]
        sx = np.sin(np.pi * grid_x.centers)[np.newaxis]
        phi_ana = sx * sy
        lam = 0. if scalar_lambda else np.zeros(arr_shape)
        rhs = -(np.pi ** 2) * (D_xx + D_yy) * phi_ana

        problems.append((2, phi_ana, phi, lam, rhs, D_xx, D_yy, None))

    # Expected (test 2):
    # Number of iterations: 14
    # rmse for 128 is 2.490588070275737e-05
    # result here: 2.490588070275727e-05

    # test 3 (spatial diffusion)
    if 3 in problem_ids:
        sy = np.sin(np.pi * grid_y.centers)[:, np.newaxis]
        x = grid_x.centers[np.newaxis]
        sx = np.sin(np.pi * x)
        cx = np.cos(np.pi * x)
        D_xx = np.full(arr_shape, x)
        D_yy = np.full(arr_shape, x)
        phi_ana = sx * sy
        lam = 0. if scalar_lambda else np.zeros(arr_shape)
        rhs = np.pi * cx * sy - 2 * np.pi ** 2 * x * phi_ana

        problems.append((3, phi_ana, phi, lam, rhs, D_xx, D_yy, None))

    # Expected (test 3):
    # Number of iterations: 41
    # rmse for 128 is 2.197736631958706e-05
    # result here: 2.197736631958708e-05

    # test 4 (spatial diffusion with lambda)
    if 4 in problem_ids:
        y = grid_y.centers[:, np.newaxis]
        sy = np.sin(np.pi * y)
        x = grid_x.centers[np.newaxis]
        sx = np.sin(np.pi * x)
        D_xx = np.full(arr_shape, x)
        D_yy = np.zeros(arr_shape) + y
        phi_ana = sx * sy
        lam = 0.2 * x * y ** 2
        xysum = x + y
        rhs = np.pi * np.sin(np.pi * xysum) - (xysum * np.pi ** 2 + lam) * phi_ana

        problems.append((4, phi_ana, phi, lam, rhs, D_xx, D_yy, None))

    # Expected (test 4):
    # Number of iterations: 468
    # rmse for 128 is 2.179081815661214e-05
    # result here: 2.179081815661213e-05

    # test 5 (constant diffusion with lambda)
    if 5 in problem_ids:
        D_xx = 1.0
        D_yy = 1e-2
        x = grid_x.centers[np.newaxis]
        y = grid_y.centers[:, np.newaxis]
        sx = np.sin(np.pi * x)
        sy = np.sin(np.pi * y)
        phi_ana = sx * sy
        lam = 0.2 * x * y ** 2
        rhs = -((np.pi ** 2) * (D_xx + D_yy) + lam) * phi_ana

        problems.append((5, phi_ana, phi, lam, rhs, D_xx, D_yy, None))

    # Expected (test 5):
    # Number of iterations: 208
    # rmse for 128 is 2.483558318071642e-05
    # result here: 2.483558318071646e-05

    # test 7 (spatial diffusion with off-diagonal component)
    if 7 in problem_ids:
        c = 0.1
        y = grid_y.centers[:, np.newaxis]
        sy = np.sin(np.pi * y)
        cy = np.cos(np.pi * y)
        x = grid_x.centers[np.newaxis]
        sx = np.sin(np.pi * x)
        cx = np.cos(np.pi * x)
        D_xx = np.full(arr_shape, x)
        D_yy = np.zeros(arr_shape) + y
        D_xy = np.full(arr_shape, c * x)
        phi_ana = sx * sy
        lam = 0.2 * x * y ** 2
        rhs = (1. + c) * np.pi * sx * cy + np.pi * cx * sy + c * 2. * x * np.pi ** 2 * cx * cy - ((x + y) * np.pi ** 2 + lam) * phi_ana

        problems.append((7, phi_ana, phi, lam, rhs, D_xx, D_yy, D_xy))

    # Expected (test 7):
    # Number of iterations: 466
    # rmse for 128 is 2.291672360185362e-05
    # result here: 2.291672360362869e-05 (+1.775069349320424e-15)

    # test 9 (spatial diffusion with off-diagonal component)
    if 9 in problem_ids:
        c = 0.5
        y = grid_y.centers[:, np.newaxis]
        sy = np.sin(np.pi * y)
        cy = np.cos(np.pi * y)
        x = grid_x.centers[np.newaxis]
        sx = np.sin(np.pi * x)
        cx = np.cos(np.pi * x)
        D_xx = np.full(arr_shape, x)
        D_yy = np.zeros(arr_shape) + y
        D_xy = c * x * y
        phi_ana = sx * sy
        lam = 0.2 * x * y ** 2
        rhs = (1 + c * x) * np.pi * cx * sy + (1 + c * y) * np.pi * sx * cy + c * 2 * x * y * np.pi ** 2 * cx * cy - ((x + y) * np.pi ** 2 + lam) * phi_ana

        problems.append((9, phi_ana, phi, lam, rhs, D_xx, D_yy, D_xy))

    # Expected (test 9):
    # Number of iterations: 462
    # rmse for 128 is 2.719754537531388e-05
    # result here: 2.719754537272108e-05

    # test 10 (spatial diffusion with off-diagonal component)
    if 10 in problem_ids:
        c = 0.1
        y = grid_y.centers[:, np.newaxis]
        sy = np.sin(np.pi * y)
        cy = np.cos(np.pi * y)
        x = grid_x.centers[np.newaxis]
        sx = np.sin(np.pi * x)
        cx = np.cos(np.pi * x)
        D_xx = np.zeros(arr_shape) + y
        D_yy = np.zeros(arr_shape) + y
        D_xy = c * x * y
        phi_ana = sx * sy
        lam = 0.2 * x * y ** 2
        rhs = np.pi * (1 + c * y) * sx * cy + c * x * np.pi * cx * sy + c * 2 * x * y * np.pi ** 2 * cx * cy - (2 * y * np.pi ** 2 + lam) * phi_ana

        problems.append((10, phi_ana, phi, lam, rhs, D_xx, D_yy, D_xy))

    # Expected (test 10):
    # Number of iterations: 41
    # rmse for 128 is 2.317202758890556e-05
    # result here: 2.317202759071838e-05

    # test 11 (spatial diffusion with off-diagonal component; cylinder)
    if 11 in problem_ids:
        dpar = 1.
        dperp = 0.1
        y = grid_y.centers[:, np.newaxis]
        sy = np.sin(np.pi * y)
        cy = np.cos(np.pi * y)
        x = grid_x.centers[np.newaxis]
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

        problems.append((11, phi_ana, phi, lam, rhs, D_xx, D_yy, D_xy))

    # Expected (test 11):
    # Number of iterations: 56
    # rmse for 128 is 3.881907960699822e-05
    # result here: 3.881907960671440e-05

    print("setting up problems took {} s".format(time.time() - curr_time))

    sep = "---------------------------------------------------------------------------------------"
    sep1 = "*************************************************************"

    data_runs = []
    data_warmups = []

    if smi:
        err = initialize_jax_smi()
        if err:
            print(f"Unable to initialize jax-smi: {err}")
        else:
            import jax_smi

    if cpu_utilization or gpu_utilization:
        err = initialize_threading()
        if err:
            print(f"Unable to initialize threading: skipping '<device>_utilization'. Reason: {err}")
            cpu_utilization = False
            gpu_utilization = False
        else:
            import threading

    if cpu_utilization:
        err = initialize_psutil()
        if err:
            print(f"Unable to initialize psutil: skipping 'cpu_utilization'. Reason: {err}")
            cpu_utilization = False
        else:
            import psutil

    if gpu_utilization:
        handle, error_msg = initialize_pynvml()
        if handle:
            import pynvml
            gpu_name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(gpu_name, bytes):
                gpu_name = gpu_name.decode("utf-8")
            print(f"GPU detected: {gpu_name}")
        else:
            print(f"No GPU available: skipping 'gpu_utilization'. Reason: {error_msg}")
            gpu_utilization = False

    for (testnr, phi_ana, phi, lam, rhs, D_xx, D_yy, D_xy) in problems:
        print(sep)
        print(f"problem: {testnr} ({str(datetime.datetime.now())})" if timestamp else f"problem: {testnr}")
        print(sep)

        (ref_iters, ref_rmse) = (force_iterations, None) if force_iterations > 0 else reference[(testnr, x_res)] if reference_usable else (None, None)

        if (bench_vjp or fixed) and ref_iters is None:
            raise ValueError("No reference value for number of iterations available for this setup.")

        @jax.jit
        def solve(primals):
            """
            Note: Local definition leads to recompilation across loop iterations (i.e. between different tests).
            However, since 'ref_iters' is a test-specific constant (corresponds to static arg w.r.t. jit on a global function) there is no way around it.
            Also for functions below!
            """
            if fixed and fcycle:
                return solve_2d_fixed_fcycle_simple(grid_x, grid_y, ref_iters, *primals)
            elif not fixed and fcycle:
                return solve_2d_fcycle_simple(grid_x, grid_y, *primals)
            elif fixed and not fcycle:
                return solve_2d_fixed_simple(grid_x, grid_y, ref_iters, *primals)
            else:
                return solve_2d_simple(grid_x, grid_y, *primals)

        @jax.jit
        def jvp(primals, tangents):
            """
            Needed since 'jax.jit' needs to be outermost call in order to use '.trace()'.
            """
            if fixed and fcycle:
                f = partial(solve_2d_fixed_fcycle_simple, grid_x, grid_y, ref_iters)
            elif not fixed and fcycle:
                f = partial(solve_2d_fcycle_simple, grid_x, grid_y)
            elif fixed and not fcycle:
                f = partial(solve_2d_fixed_simple, grid_x, grid_y, ref_iters)
            else:
                f = partial(solve_2d_simple, grid_x, grid_y)

            return jax.jvp(f, primals, tangents, True)

        @jax.jit
        def vjp(primals, cotangent):
            """
            Needed since 'jax.jit' needs to be outermost call in order to use '.trace()'.
            Also gives significant speedup.
            """
            if fcycle:
                f = partial(solve_2d_fixed_fcycle_simple, grid_x, grid_y, ref_iters)
            else:
                f = partial(solve_2d_fixed_simple, grid_x, grid_y, ref_iters)

            g = jax.checkpoint(f)

            primals_out, f_vjp, iterations = jax.vjp(g if remat else f, *primals, has_aux=True)
            cotangents_out = f_vjp(cotangent)

            return primals_out, cotangents_out, iterations

        # args for jvp/vjp
        primals = (phi, rhs, lam, D_xx, D_yy, D_xy) if D_xy is not None else (phi, rhs, lam, D_xx, D_yy)
        cotangent = np.ones(arr_shape)

        def compute_tuple_size_mib(data, print_individual_sizes=False):
            """Computes the size in MiB of a tuple containing NumPy arrays and scalars.

            Args:
                data (tuple): A tuple containing NumPy arrays and scalars.
                print_individual_sizes (bool): If True, prints the size of each tuple element in MiB.

            Returns:
                float: Total size in MiB.
            """
            total_bytes = 0
            for item in data:
                if isinstance(item, np.ndarray):
                    item_bytes = item.nbytes
                elif np.isscalar(item):
                    item_bytes = np.array(item).nbytes
                else:
                    raise TypeError(f"Unsupported type: {type(item)}")

                if print_individual_sizes:
                    print(f"Item size: {item_bytes / (1024 ** 2):.6f} MiB")
                total_bytes += item_bytes

            size_mib = total_bytes / (1024 ** 2)
            return size_mib

        def measure_transfer_times(data):
            gpu_device = jax.devices('gpu')[0]
            #gpu_device = jax.devices('cpu')[0]

            cpu_to_gpu_times = []
            gpu_to_cpu_times = []

            for i in range(1, reps_transfer + 1):
                # modify data between iterations
                data = jax.tree.map(lambda d: d + 1.42, data)

                # cpu -> gpu
                start_time = time.time()
                dev_data = jax.device_put(data, device=gpu_device)
                jax.block_until_ready(dev_data)
                cpu_to_gpu_times.append(time.time() - start_time)

                # execute kernel to manipulate device data
                @partial(jax.jit, static_argnames=('fac',))
                def func(x, fac):
                    return jax.tree.map(lambda d: (d + 0.42) * fac, x)

                dev_data = func(dev_data, i * 1.42)
                jax.block_until_ready(dev_data)

                # gpu -> cpu
                start_time = time.time()
                retrieved_data = jax.device_get(dev_data)
                jax.block_until_ready(retrieved_data)
                gpu_to_cpu_times.append(time.time() - start_time)

            print(f"cpu -> gpu: {np.median(cpu_to_gpu_times):.6f} s (min {np.min(cpu_to_gpu_times):.6f}, mean {np.mean(cpu_to_gpu_times):.6f}, max {np.max(cpu_to_gpu_times):.6f})")
            print(f"cpu <- gpu: {np.median(gpu_to_cpu_times):.6f} s (min {np.min(gpu_to_cpu_times):.6f}, mean {np.mean(gpu_to_cpu_times):.6f}, max {np.max(gpu_to_cpu_times):.6f})")

        if bench_transfer:
            data = primals
            print("data size: {} MiB".format(compute_tuple_size_mib(data, print_individual_sizes=True)))
            measure_transfer_times(data)

        if bench_transfer and transfer_only:
            continue

        #@profile # memory_profiler (mprof)
        def details(dsolve=False, djvp=False, dvjp=False):
            unit = 'solve' if dsolve else 'jvp' if djvp else 'vjp' if dvjp else 'none'
            print(f"JIT details for '{unit}'")
            print(sep1)
            trace_times = []
            lower_times = []
            compile_times = []
            for i in range(1, reps_details + 1):
                jax.clear_caches()
                curr_time = time.time()
                if dsolve:
                    traced = solve.trace(primals)
                elif djvp:
                    traced = jvp.trace(primals, primals)
                elif dvjp:
                    traced = vjp.trace(primals, cotangent)
                else:
                    raise ValueError("details() was called for unknown unit.")
                trace_time = time.time() - curr_time
                trace_times.append(trace_time)
                curr_time = time.time()
                lowered = traced.lower()
                lower_time = time.time() - curr_time
                lower_times.append(lower_time)
                curr_time = time.time()
                compiled = lowered.compile()
                compile_time = time.time() - curr_time
                compile_times.append(compile_time)
            print("tracing: took {} s".format(np.median(trace_times)))
            print("lowering: took {} s".format(np.median(lower_times)))
            print("compilation: took {} s".format(np.median(compile_times)))
            flops = int(compiled.cost_analysis()[0]['flops'])
            print(f"cost['flops']: {flops}")
            memory = compiled.memory_analysis()
            print("code size: {}\narg size: {}\noutput size: {}\ntmp size: {}\nsum size: {}".format(memory.generated_code_size_in_bytes, memory.argument_size_in_bytes, memory.output_size_in_bytes, memory.temp_size_in_bytes, memory.generated_code_size_in_bytes + memory.argument_size_in_bytes + memory.output_size_in_bytes + memory.temp_size_in_bytes))
            #print(f"complete memory analysis: {memory}")
            if export_internals:
                filename_jaxpr = file_prefix + unit + "_jaxpr" + file_suffix + ".txt"
                filename_lower = file_prefix + unit + "_lowered" + file_suffix + ".txt"
                filename_compile = file_prefix + unit + "_compiled" + file_suffix + ".txt"
                with open(filename_jaxpr, "w") as file_jaxpr:
                    if dsolve:
                        file_jaxpr.write(str(jax.make_jaxpr(solve)(primals)))
                    elif djvp:
                        file_jaxpr.write(str(jax.make_jaxpr(jvp)(primals, primals)))
                    elif dvjp:
                        file_jaxpr.write(str(jax.make_jaxpr(vjp)(primals, cotangent)))
                with open(filename_lower, "w") as file_lower:
                    file_lower.write(lowered.as_text())
                with open(filename_compile, "w") as file_compile:
                    file_compile.write(compiled.as_text())
            #print(f"num runs: {len(trace_times)}, {trace_times}, {lower_times}, {compile_times}")
            print(sep1)

        if jit_details and bench_solve:
            details(dsolve=True)
        if jit_details and bench_jvp:
            details(djvp=True)
        if jit_details and bench_vjp:
            details(dvjp=True)
        if jit_details and details_only:
            continue

        def cpu_util(usolve=False, ujvp=False, uvjp=False):
            results = []
            samples = []
            for _ in range(1, reps_util + 1):
                result = {"running": True, "usage": []}
                monitor_thread = threading.Thread(target=monitor_cpu, args=(0.1, result))
                monitor_thread.start()

                if usolve:
                    phi_res, iterations = solve(primals)
                    jax.block_until_ready(phi_res)
                elif ujvp:
                    primals_out, tangents_out, iterations = jvp(primals, primals)
                    jax.block_until_ready(primals_out)
                    jax.block_until_ready(tangents_out)
                elif uvjp:
                    primals_out, cotangents_out, iterations = vjp(primals, cotangent)
                    jax.block_until_ready(primals_out)
                    jax.block_until_ready(cotangents_out)
                else:
                    raise ValueError("cpu_util() called for unknown unit.")

                result["running"] = False
                monitor_thread.join()
                utilization = result["usage"]

                mean_result = np.mean(utilization)
                mean_result *= psutil.cpu_count(logical=False)
                results.append(mean_result)
                samples.append(len(utilization))

            print(f"CPU utilization ({np.median(samples):.0f} samples): {np.median(results):.0f} % (min {np.min(results):.0f}, mean {np.mean(results):.1f}, max {np.max(results):.0f})")

        def gpu_util(usolve=False, ujvp=False, uvjp=False):
            results = []
            mem_results = []
            mem_total_results = []
            samples = []
            for _ in range(1, reps_util + 1):
                result = {"running": True, "usage": [], "mem_usage": [], "mem": []}
                monitor_thread = threading.Thread(target=monitor_gpu, args=(handle, 0.1, result))
                monitor_thread.start()

                if usolve:
                    phi_res, iterations = solve(primals)
                    jax.block_until_ready(phi_res)
                elif ujvp:
                    primals_out, tangents_out, iterations = jvp(primals, primals)
                    jax.block_until_ready(primals_out)
                    jax.block_until_ready(tangents_out)
                elif uvjp:
                    primals_out, cotangents_out, iterations = vjp(primals, cotangent)
                    jax.block_until_ready(primals_out)
                    jax.block_until_ready(cotangents_out)
                else:
                    raise ValueError("gpu_util() called for unknown unit.")

                result["running"] = False
                monitor_thread.join()
                utilization = result["usage"]
                mem_utilization = result["mem_usage"]
                mem = result["mem"]

                results.append(np.mean(utilization))
                mem_results.append(np.mean(mem_utilization))
                mem_total_results.append(np.max(mem) if max_mem else np.mean(mem))
                samples.append(len(utilization))

            print(f"GPU utilization ({np.median(samples):.0f} samples): {np.median(results):.0f} % (min {np.min(results):.0f}, mean {np.mean(results):.1f}, max {np.max(results):.0f})")
            print(f"GPU memory utilization ({np.median(samples):.0f} samples): {np.median(mem_results):.0f} % (min {np.min(mem_results):.0f}, mean {np.mean(mem_results):.1f}, max {np.max(mem_results):.0f})")
            print(f"GPU memory ({np.median(samples):.0f} samples): {np.median(mem_total_results):.1f} GiB (min {np.min(mem_total_results):.1f}, mean {np.mean(mem_total_results):.2f}, max {np.max(mem_total_results):.1f})")

        if bench_solve:
            curr_time = time.time()
            phi_res, iterations = solve(primals)
            jax.block_until_ready(phi_res)
            solver_warmup = time.time() - curr_time
            print("initial call: solved problem in {} s".format(solver_warmup))

            error = (phi_ana - phi_res)[1:-1,1:-1]
            rmse = np.linalg.norm(error) / np.sqrt(error.size)

            if print_result:
                print("==> iterations: {}, rmse: {}".format(iterations, rmse))

            if save_result:
                np.savez_compressed(f"problem_{testnr}_{phi.shape[0]-3}x{phi.shape[1]-3}.npz", rmse=rmse, iterations=iterations, phi_res=phi_res, phi_ana=phi_ana, lam=lam, rhs=rhs, D_xx=D_xx, D_yy=D_yy, D_xy=D_xy)

            if reference_usable and force_iterations == 0 and check_result:
                if rmse > ref_rmse + 2e-15 or not iterations == ref_iters:
                    print("rmse deviation: {}".format(rmse - ref_rmse))
                    print("iterations: {} (goal: {})".format(iterations, ref_iters))
                    raise ValueError(f"Failed to meet goals for problem {testnr}")

            if cpu_utilization:
                cpu_util(usolve=True)

            if gpu_utilization:
                gpu_util(usolve=True)

            # repeated solver calls
            solver_runs = []
            for i in range(1, reps + 1):
                #with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
                #with jax.profiler.trace("/tmp/tensorboard"):
                curr_time = time.time()
                phi_res, iterations = solve(primals)
                jax.block_until_ready(phi_res)
                solver_runs.append(time.time() - curr_time)
                print("repeated call {}: solved problem in {} s".format(i, solver_runs[-1]))

        if bench_jvp:
            curr_time = time.time()
            primals_out, tangents_out, iterations = jvp(primals, primals)
            jax.block_until_ready(primals_out)
            jax.block_until_ready(tangents_out)
            jvp_warmup = time.time() - curr_time
            print("initial jvp call: took {} s".format(jvp_warmup))

            if cpu_utilization:
                cpu_util(ujvp=True)

            if gpu_utilization:
                gpu_util(ujvp=True)

            # repeated jvp calls
            jvp_runs = []
            for i in range(1, reps + 1):
                curr_time = time.time()
                primals_out, tangents_out, iterations = jvp(primals, primals)
                jax.block_until_ready(primals_out)
                jax.block_until_ready(tangents_out)
                jvp_runs.append(time.time() - curr_time)
                print("repeated jvp call {}: took {} s".format(i, jvp_runs[-1]))

        if bench_vjp:
            curr_time = time.time()
            primals_out, cotangents_out, iterations = vjp(primals, cotangent)
            jax.block_until_ready(primals_out)
            jax.block_until_ready(cotangents_out)
            vjp_warmup = time.time() - curr_time
            print("initial vjp call: took {} s".format(vjp_warmup))

            if cpu_utilization:
                cpu_util(uvjp=True)

            if gpu_utilization:
                gpu_util(uvjp=True)

            # repeated vjp calls
            vjp_runs = []
            for i in range(1, reps + 1):
                curr_time = time.time()
                primals_out, cotangents_out, iterations = vjp(primals, cotangent)
                jax.block_until_ready(primals_out)
                jax.block_until_ready(cotangents_out)
                vjp_runs.append(time.time() - curr_time)
                print("repeated vjp call {}: took {} s".format(i, vjp_runs[-1]))

        if bench_solve and bench_jvp and bench_vjp:
            data_warmups.append({
                "problem": testnr,
                "solver_warmup": solver_warmup,
                "jvp_warmup": jvp_warmup,
                "vjp_warmup": vjp_warmup
            })
            for i in range(reps):
                data_runs.append({
                    "problem": testnr,
                    "solver_run": solver_runs[i],
                    "jvp_run": jvp_runs[i],
                    "vjp_run": vjp_runs[i]
                })
        elif bench_solve and bench_jvp and not bench_vjp:
            data_warmups.append({
                "problem": testnr,
                "solver_warmup": solver_warmup,
                "jvp_warmup": jvp_warmup
            })
            for i in range(reps):
                data_runs.append({
                    "problem": testnr,
                    "solver_run": solver_runs[i],
                    "jvp_run": jvp_runs[i]
                })
        elif bench_solve and not bench_jvp and bench_vjp:
            data_warmups.append({
                "problem": testnr,
                "solver_warmup": solver_warmup,
                "vjp_warmup": vjp_warmup
            })
            for i in range(reps):
                data_runs.append({
                    "problem": testnr,
                    "solver_run": solver_runs[i],
                    "vjp_run": vjp_runs[i]
                })
        elif bench_solve and not bench_jvp and not bench_vjp:
            data_warmups.append({
                "problem": testnr,
                "solver_warmup": solver_warmup
            })
            for i in range(reps):
                data_runs.append({
                    "problem": testnr,
                    "solver_run": solver_runs[i]
                })
        elif not bench_solve and bench_jvp and bench_vjp:
            data_warmups.append({
                "problem": testnr,
                "jvp_warmup": jvp_warmup,
                "vjp_warmup": vjp_warmup
            })
            for i in range(reps):
                data_runs.append({
                    "problem": testnr,
                    "jvp_run": jvp_runs[i],
                    "vjp_run": vjp_runs[i]
                })
        elif not bench_solve and bench_jvp and not bench_vjp:
            data_warmups.append({
                "problem": testnr,
                "jvp_warmup": jvp_warmup
            })
            for i in range(reps):
                data_runs.append({
                    "problem": testnr,
                    "jvp_run": jvp_runs[i]
                })
        elif not bench_solve and not bench_jvp and bench_vjp:
            data_warmups.append({
                "problem": testnr,
                "vjp_warmup": vjp_warmup
            })
            for i in range(reps):
                data_runs.append({
                    "problem": testnr,
                    "vjp_run": vjp_runs[i]
                })
        else:
            raise ValueError("No data to append.")

    if export:
        filename_runs = file_prefix + "runs" + file_suffix + ".csv"
        filename_warmups = file_prefix + "warmups" + file_suffix + ".csv"

        if bench_solve and bench_jvp and bench_vjp:
            fieldnames_runs = ["problem", "solver_run", "jvp_run", "vjp_run"]
            fieldnames_warmups = ["problem", "solver_warmup", "jvp_warmup", "vjp_warmup"]
        elif bench_solve and bench_jvp and not bench_vjp:
            fieldnames_runs = ["problem", "solver_run", "jvp_run"]
            fieldnames_warmups = ["problem", "solver_warmup", "jvp_warmup"]
        elif bench_solve and not bench_jvp and bench_vjp:
            fieldnames_runs = ["problem", "solver_run", "vjp_run"]
            fieldnames_warmups = ["problem", "solver_warmup", "vjp_warmup"]
        elif bench_solve and not bench_jvp and not bench_vjp:
            fieldnames_runs = ["problem", "solver_run"]
            fieldnames_warmups = ["problem", "solver_warmup"]
        elif not bench_solve and bench_jvp and bench_vjp:
            fieldnames_runs = ["problem", "jvp_run", "vjp_run"]
            fieldnames_warmups = ["problem", "jvp_warmup", "vjp_warmup"]
        elif not bench_solve and bench_jvp and not bench_vjp:
            fieldnames_runs = ["problem", "jvp_run"]
            fieldnames_warmups = ["problem", "jvp_warmup"]
        elif not bench_solve and not bench_jvp and bench_vjp:
            fieldnames_runs = ["problem", "vjp_run"]
            fieldnames_warmups = ["problem", "vjp_warmup"]
        else:
            raise ValueError("Export failed: No valid benchmarks selected.")

        with open(filename_runs, mode="w", newline="") as file_runs, open(filename_warmups, mode="w", newline="") as file_warmups:
            writer_runs = csv.DictWriter(file_runs, fieldnames=fieldnames_runs)
            writer_runs.writeheader()
            writer_runs.writerows(data_runs)
            writer_warmups = csv.DictWriter(file_warmups, fieldnames=fieldnames_warmups)
            writer_warmups.writeheader()
            writer_warmups.writerows(data_warmups)

        print(sep)
        print(f"CSV data has been written to '{filename_runs}' and '{filename_warmups}'.")

    print(sep)

def main():
    parser = argparse.ArgumentParser(
        description="Script to benchmark solver and jvp calls.",
        epilog="Example: python benchmark.py --resolution 64 --type 'c' --reps 5"
    )

    parser.add_argument('-r', '--resolution', type=int, default=128,
                        help='Global resolution value (integer; power of 2) for both dimensions. Default is 128.')
    parser.add_argument('-x', '--x_resolution', type=int,
                        help='Resolution value for the x dimension (integer; power of 2).')
    parser.add_argument('-y', '--y_resolution', type=int,
                        help='Resolution value for the y dimension (integer; power of 2).')
    parser.add_argument('--reps', type=int, default=10,
                        help='Repetitions (runs) of compiled function for each problem (integer). Default is 10.')
    parser.add_argument('--solve', default=True, action=argparse.BooleanOptionalAction,
                        help='Benchmark solver calls. Default is True.')
    parser.add_argument('--jvp', default=True, action=argparse.BooleanOptionalAction,
                        help='Benchmark JVP calls. Default is True.')
    parser.add_argument('--vjp', default=True, action=argparse.BooleanOptionalAction,
                        help='Benchmark VJP calls. Default is True.')
    parser.add_argument('--check', default=True, action=argparse.BooleanOptionalAction,
                        help='Check result of solver call (only supported for resolution == 128). Default is True.')
    parser.add_argument('--print', default=False, action=argparse.BooleanOptionalAction,
                        help='Print result of solver call (RMSE and iteration count). Default is False.')
    parser.add_argument('-e', '--export', default=False, action=argparse.BooleanOptionalAction,
                        help='Export measurements as CSV. Default is False.')
    parser.add_argument('--save', default=False, action=argparse.BooleanOptionalAction,
                        help='Save problem and computed result. Default is False.')
    parser.add_argument('--scalar', default=True, action=argparse.BooleanOptionalAction,
                        help='Use scalar lambda for problems 1, 2, and 3. Leads to additional recompilation. Default is True.')
    parser.add_argument('--fcycle', default=False, action=argparse.BooleanOptionalAction,
                        help='Use F-cycle instead of V-cycle. Default is False.')
    parser.add_argument('-d', '--details', default=False, action=argparse.BooleanOptionalAction,
                        help='Enable JIT details. Default is False.')
    parser.add_argument('--details-only', default=False, action=argparse.BooleanOptionalAction,
                        help='Enable JIT details and disable the rest. Default is False.')
    parser.add_argument('--util', default=False, action=argparse.BooleanOptionalAction,
                        help='Single run where CPU utilization is obtained. Default is False.')
    parser.add_argument('--gutil', default=False, action=argparse.BooleanOptionalAction,
                        help='Single run where GPU utilization is obtained. Default is False.')
    parser.add_argument('--fixed', default=False, action=argparse.BooleanOptionalAction,
                        help='Use fixed number of iterations for solve/jvp (looked up from hardcoded dictionary). Default is False.')
    parser.add_argument('--remat', default=False, action=argparse.BooleanOptionalAction,
                        help='Use checkpointed function for vjp, i.e. use rematerialization strategy. Default is False.')
    parser.add_argument('--smi', default=False, action=argparse.BooleanOptionalAction,
                        help='Initialize tracking for jax-smi. Default is False.')
    parser.add_argument('--internals', default=False, action=argparse.BooleanOptionalAction,
                        help='Export internals (i.e. intermediate representations during JIT process) when details are investigated. Default is False.')
    parser.add_argument('--transfer', default=False, action=argparse.BooleanOptionalAction,
                        help='Benchmark transfer times. Default is False.')
    parser.add_argument('--transfer-only', default=False, action=argparse.BooleanOptionalAction,
                        help='Only benchmark transfer times. Default is False.')
    parser.add_argument('--max', default=False, action=argparse.BooleanOptionalAction,
                        help='Use maximum instead of mean to aggregate memory consumption samples. Default is False.')
    parser.add_argument('--dreps', type=int, default=1,
                        help='Repetitions (runs) of JIT details (integer). Default is 1.')
    parser.add_argument('--ureps', type=int, default=1,
                        help='Repetitions (runs) of calls for sampling CPU or GPU utilization (integer). Default is 1.')
    parser.add_argument('--treps', type=int, default=10,
                        help='Repetitions (runs) of transfer benchmark (integer). Default is 10.')
    parser.add_argument('--iters', type=int, default=0,
                        help='Force solver to do exactly this number of multigrid iterations (do not set --fixed too). Default is 0 (dynamic).')
    parser.add_argument('--timestamp', default=False, action=argparse.BooleanOptionalAction,
                        help='Print timestamp. Default is False.')
    parser.add_argument('-s', '--suffix', type=str, default=None,
                        help='Suffix for CSV file names. Default is None.')
    parser.add_argument('-p', '--prefix', type=str, default=None,
                        help='Prefix for CSV file names. Default is None.')
    parser.add_argument('-i', '--isolate', type=int, default=0,
                        help='Isolate a specific problem (integer; e.g. 2 to only run problem with id 2). Default is 0 (run all problems).')
    parser.add_argument('--skip', type=int, default=0,
                        help='Skip a specific problem (integer; e.g. 2 to skip problem with id 2). Default is 0 (run all problems).')
    parser.add_argument('-c', '--cutoff', type=int, default=0,
                        help='Benchmarks problems up to cutoff (integer; inclusive). Default is 0 (run all problems).')
    parser.add_argument('--start', type=int, default=0,
                        help='Benchmarks problems starting from start (integer; inclusive). Default is 0 (run all problems).')
    parser.add_argument('-t', '--type', type=str, default=None,
                        help='Benchmark only problems of the given type: "c" for spatially-constant diffusion, "s" for spatially-varying diffusion, or "o" for spatially-varying diffusion with off-diagonal component. Default is None (run all problems).')

    args = parser.parse_args()

    x_res = args.x_resolution if args.x_resolution is not None else args.resolution
    y_res = args.y_resolution if args.y_resolution is not None else args.resolution

    if x_res <= 0 or y_res <= 0 or args.reps < 0 or args.isolate < 0 or args.cutoff < 0:
        parser.error("All integer arguments must be sensible (e.g. non-negative)!")

    print(f"Resolution is set to: x={x_res}, y={y_res}")

    details = args.details or args.details_only
    transfer = args.transfer or args.transfer_only

    bench(x_res, y_res, print_result=args.print, check_result=args.check, bench_solve=args.solve, bench_jvp=args.jvp, bench_vjp=args.vjp, reps=args.reps, isolate=args.isolate, skip=args.skip, cutoff=args.cutoff, start=args.start, export=args.export, save_result=args.save, file_prefix=args.prefix, file_suffix=args.suffix, problem_type=args.type, scalar_lambda=args.scalar, fcycle=args.fcycle, timestamp=args.timestamp, jit_details=details, reps_details=args.dreps, details_only=args.details_only, cpu_utilization=args.util, gpu_utilization=args.gutil, reps_util=args.ureps, fixed=args.fixed, remat=args.remat, force_iterations=args.iters, smi=args.smi, export_internals=args.internals, bench_transfer=transfer, transfer_only=args.transfer_only, reps_transfer=args.treps, max_mem=args.max)

if __name__ == "__main__":
    main()

