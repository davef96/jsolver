import jax
import numpy as np
from jax.test_util import check_grads
from jsolver.grid_1d import grid_1D
from jsolver.solver_2d import solve_2d, solve_2d_simple
import time
import argparse
import csv
import datetime
from functools import partial
#from memory_profiler import profile

#jax.config.update('jax_enable_x64', True) # use float64 as default (already done in 'solver_2d')
#jax.config.update("jax_debug_nans", True)
#jax.config.update("jax_enable_compilation_cache", False)
#jax.config.update('jax_compiler_enable_remat_pass', False)

def bench(x_res, y_res, print_result=False, check_result=False, bench_solve=True, bench_jvp=True, reps=10, reps_details=1, isolate=0, skip=0, cutoff=0, start=0, problem_type=None, export=False, save_result=False, file_prefix=None, file_suffix=None, scalar_lambda=True, fcycle=False, timestamp=False, jit_details=False, details_only=False):
    begin = time.time()

    if timestamp:
        print(f"benchmark start: {str(datetime.datetime.now())}")

    if not bench_solve and not bench_jvp:
        raise ValueError("No benchmark was selected. Nothing to do.")

    problem_ids = [ i for i in range(1, 12) ]
    problem_ids = filter(lambda i: not i == 6 or not i == 8, problem_ids)

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

        rmse_goal = 1.010315424011278e-06
        iter_goal = 13
        problems.append((1, phi_ana, rmse_goal, iter_goal, phi, lam, rhs, D_xx, D_yy, None))

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

        rmse_goal = 2.490588070275737e-05
        iter_goal = 14
        problems.append((2, phi_ana, rmse_goal, iter_goal, phi, lam, rhs, D_xx, D_yy, None))

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

        rmse_goal = 2.197736631958706e-05
        iter_goal = 41
        problems.append((3, phi_ana, rmse_goal, iter_goal, phi, lam, rhs, D_xx, D_yy, None))

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

        rmse_goal = 2.179081815661214e-05
        iter_goal = 468
        problems.append((4, phi_ana, rmse_goal, iter_goal, phi, lam, rhs, D_xx, D_yy, None))

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

        rmse_goal = 2.483558318071642e-05
        iter_goal = 208
        problems.append((5, phi_ana, rmse_goal, iter_goal, phi, lam, rhs, D_xx, D_yy, None))

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

        rmse_goal = 2.291672360185362e-05
        iter_goal = 466
        problems.append((7, phi_ana, rmse_goal, iter_goal, phi, lam, rhs, D_xx, D_yy, D_xy))

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

        rmse_goal = 2.719754537531388e-05
        iter_goal = 462
        problems.append((9, phi_ana, rmse_goal, iter_goal, phi, lam, rhs, D_xx, D_yy, D_xy))

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

        rmse_goal = 2.317202758890556e-05
        iter_goal = 41
        problems.append((10, phi_ana, rmse_goal, iter_goal, phi, lam, rhs, D_xx, D_yy, D_xy))

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

        rmse_goal = 3.881907960699822e-05
        iter_goal = 56
        problems.append((11, phi_ana, rmse_goal, iter_goal, phi, lam, rhs, D_xx, D_yy, D_xy))

    # Expected (test 11):
    # Number of iterations: 56
    # rmse for 128 is 3.881907960699822e-05
    # result here: 3.881907960671440e-05

    print("setting up problems took {} s".format(time.time() - curr_time))

    sep = "---------------------------------------------------------------------------------------"
    sep1 = "*************************************************************"

    data_runs = []
    data_warmups = []

    for (testnr, phi_ana, rmse_goal, iter_goal, phi, lam, rhs, D_xx, D_yy, D_xy) in problems:
        print(sep)
        print(f"problem: {testnr} ({str(datetime.datetime.now())})" if timestamp else f"problem: {testnr}")
        print(sep)
        primals = (phi, rhs, lam, D_xx, D_yy, D_xy) if D_xy is not None else (phi, rhs, lam, D_xx, D_yy)

        #@profile
        def details():
            if bench_solve:
                print("JIT details for 'solve'")
                print(sep1)
                trace_times = []
                lower_times = []
                compile_times = []
                for i in range(1, reps_details + 1):
                    jax.clear_caches()
                    curr_time = time.time()
                    traced = solve_2d.trace(grid_x=grid_x, grid_y=grid_y, phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy, D_xy=D_xy, fcycle=fcycle)
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
                #print(lowered.as_text())
                #print(compiled.as_text())
                print(f"num runs: {len(trace_times)}, {trace_times}, {lower_times}, {compile_times}")
                print(sep1)

            if bench_jvp:
                @jax.jit
                def jvp(primals):
                    return jax.jvp(partial(solve_2d_simple, grid_x, grid_y), primals, primals, True)

                print("JIT details for 'jvp'")
                print(sep1)
                trace_times = []
                lower_times = []
                compile_times = []
                for i in range(1, reps_details + 1):
                    jax.clear_caches()
                    curr_time = time.time()
                    traced = jvp.trace(primals)
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
                #print(lowered.as_text())
                #print(compiled.as_text())
                print(sep1)

        if jit_details:
            details()
            if details_only:
                continue

        if bench_solve:
            curr_time = time.time()
            phi_res, iterations = solve_2d(grid_x=grid_x, grid_y=grid_y, phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy, D_xy=D_xy, fcycle=fcycle)
            jax.block_until_ready(phi_res)
            solver_warmup = time.time() - curr_time
            print("initial call: solved problem in {} s".format(solver_warmup))

            error = (phi_ana - phi_res)[1:-1,1:-1]
            rmse = np.linalg.norm(error) / np.sqrt(error.size)

            if print_result:
                print("==> iterations: {}, rmse: {}".format(iterations, rmse))

            if save_result:
                np.savez_compressed(f"problem_{testnr}_{phi.shape[0]-3}x{phi.shape[1]-3}.npz", rmse=rmse, iterations=iterations, phi_res=phi_res, phi_ana=phi_ana, lam=lam, rhs=rhs, D_xx=D_xx, D_yy=D_yy, D_xy=D_xy)

            if check_result:
                if rmse > rmse_goal + 2e-15 or not iterations == iter_goal:
                    print("rmse deviation: {}".format(rmse - rmse_goal))
                    print("iterations: {} (goal: {})".format(iterations, iter_goal))
                    raise ValueError(f"Failed to meet goals for problem {testnr}")

            # repeated solver calls
            solver_runs = []
            for i in range(1, reps + 1):
                #with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
                #with jax.profiler.trace("/tmp/tensorboard"):
                curr_time = time.time()
                phi_res, iterations = solve_2d(grid_x=grid_x, grid_y=grid_y, phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy, D_xy=D_xy, fcycle=fcycle)
                jax.block_until_ready(phi_res)
                solver_runs.append(time.time() - curr_time)
                print("repeated call {}: solved problem in {} s".format(i, solver_runs[-1]))

        if bench_jvp:
            curr_time = time.time()
            primals_out, tangents_out, iterations = jax.jvp(partial(solve_2d_simple, grid_x, grid_y), primals, primals, True)
            jax.block_until_ready(primals_out)
            jax.block_until_ready(tangents_out)
            jvp_warmup = time.time() - curr_time
            print("initial jvp call: took {} s".format(jvp_warmup))

            # repeated jvp calls
            jvp_runs = []
            for i in range(1, reps + 1):
                curr_time = time.time()
                primals_out, tangents_out, iterations = jax.jvp(partial(solve_2d_simple, grid_x, grid_y), primals, primals, True)
                jax.block_until_ready(primals_out)
                jax.block_until_ready(tangents_out)
                jvp_runs.append(time.time() - curr_time)
                print("repeated jvp call {}: took {} s".format(i, jvp_runs[-1]))

        if bench_solve and bench_jvp:
            data_warmups.append({"problem": testnr, "solver_warmup": solver_warmup, "jvp_warmup": jvp_warmup})
            for i in range(reps):
                data_runs.append({"problem": testnr, "solver_run": solver_runs[i], "jvp_run": jvp_runs[i]})
        elif bench_solve and not bench_jvp:
            data_warmups.append({"problem": testnr, "solver_warmup": solver_warmup})
            for i in range(reps):
                data_runs.append({"problem": testnr, "solver_run": solver_runs[i]})
        elif not bench_solve and bench_jvp:
            data_warmups.append({"problem": testnr, "jvp_warmup": jvp_warmup})
            for i in range(reps):
                data_runs.append({"problem": testnr, "jvp_run": jvp_runs[i]})
        else:
            raise ValueError("No data to append.")

    if export:
        file_prefix = "" if file_prefix is None else file_prefix + "_"
        file_suffix = "" if file_suffix is None else "_" + file_suffix
        filename_runs = file_prefix + "runs" + file_suffix + ".csv"
        filename_warmups = file_prefix + "warmups" + file_suffix + ".csv"
        fieldnames_runs = ["problem", "solver_run", "jvp_run"] if bench_jvp else ["problem", "solver_run"]
        fieldnames_warmups = ["problem", "solver_warmup", "jvp_warmup"] if bench_jvp else ["problem", "solver_warmup"]

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
    parser.add_argument('--dreps', type=int, default=1,
                        help='Repetitions (runs) of JIT details (integer). Default is 1.')
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

    check = args.check and x_res == 128 and y_res == 128

    print(f"Resolution is set to: x={x_res}, y={y_res}")

    details = args.details or args.details_only

    bench(x_res, y_res, print_result=args.print, check_result=check, bench_solve=args.solve, bench_jvp=args.jvp, reps=args.reps, isolate=args.isolate, skip=args.skip, cutoff=args.cutoff, start=args.start, export=args.export, save_result=args.save, file_prefix=args.prefix, file_suffix=args.suffix, problem_type=args.type, scalar_lambda=args.scalar, fcycle=args.fcycle, timestamp=args.timestamp, jit_details=details, reps_details=args.dreps, details_only=args.details_only)

if __name__ == "__main__":
    main()

