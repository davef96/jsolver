import numpy as np
from jsolver.grid_1d import grid_1D
from jsolver.solver_2d import solve_2d_simple
import time

# cache all jit-compiled functions in local directory to avoid recompilation across multiple runs of this script
# see https://docs.jax.dev/en/latest/persistent_compilation_cache.html
import jax
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

# create linear grid objects with desired resolution
x_res = 128 # column-dimension
y_res = 128 # row-dimension
grid_x = grid_1D(x_res) # centered linear grid for interval [0,1]
grid_y = grid_1D(y_res)
arr_shape = (y_res + 3, x_res + 3) # row-dimension comes first in numpy!

# set up problem using numpy ("broadcasting")
D_xx = 1.0
D_yy = 1.0
phi = np.zeros(arr_shape)
lam = 0.
sy = np.sin(np.pi * grid_y.centers)[:, np.newaxis]
sx = np.sin(np.pi * grid_x.centers)[np.newaxis]
phi_ana = sx * sy
rhs = -(np.pi ** 2) * (D_xx + D_yy) * phi_ana

# solve problem
curr_time = time.time()
phi_res, iterations = solve_2d_simple(grid_x, grid_y, phi, rhs, lam, D_xx, D_yy)
jax.block_until_ready(phi_res)
print(f"Warm-up run took {time.time() - curr_time} s")
err = (phi_ana - phi_res)[1:-1, 1:-1]
rmse = np.linalg.norm(err) / np.sqrt(err.size)
print(f"Type of 'phi_res': {type(phi_res)}, Iterations: {iterations}, RMSE: {rmse}")
for i in range(3):
    curr_time = time.time()
    phi_res, iterations = solve_2d_simple(grid_x, grid_y, phi, rhs, lam, D_xx, D_yy)
    jax.block_until_ready(phi_res)
    print(f"Repeated run {i} took {time.time() - curr_time} s")
