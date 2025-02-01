import numpy as np
from jsolver.grid_1d import grid_1D
from jsolver.solver_2d import solve_2d_simple

# cache all jit-compiled functions in local directory (note: does not work if compiled function contains host callbacks); this reduces (but does not avoid!) recompilation time across multiple runs of this script
import jax
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

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
x = grid_x.centers[np.newaxis]
y = grid_y.centers[:, np.newaxis]
ox = 1. - x
oy = 1. - y
prod = y * oy
phi_ana = x * ox ** 3 * prod
rhs = -2 * x * ox ** 3 - 6 * ox * (ox - x) * prod

# solve problem
phi_res, iterations = solve_2d_simple(grid_x, grid_y, phi, rhs, lam, D_xx, D_yy)
err = (phi_ana - phi_res)[1:-1, 1:-1]
rmse = np.linalg.norm(err) / np.sqrt(err.size)
print(f"Type of 'phi_res': {type(phi_res)}, Iterations: {iterations}, RMSE: {rmse}")
