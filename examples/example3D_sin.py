import numpy as np
from jsolver.grid_1d import grid_1D
from jsolver.solver_3d import solve_3d_simple
import sys

# cache all jit-compiled functions in local directory (note: does not work if compiled function contains host callbacks); this reduces (but does not avoid!) recompilation time across multiple runs of this script
import jax
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

# create linear grid objects with desired resolution
num_points = 64

x_res = num_points # first dimension
y_res = num_points # second dimension
z_res = num_points # second dimension
grid_x = grid_1D(x_res) # centered linear grid for interval [0,1]
grid_y = grid_1D(y_res)
grid_z = grid_1D(z_res)
arr_shape = (z_res + 3, y_res + 3, x_res + 3) # first dimension comes last

# set up problem using numpy ("broadcasting")
D_xx = 1.0
D_yy = 1.0
D_zz = 1.0
phi = np.zeros(arr_shape)
lam = 0.0

sinx = np.sin(np.pi * grid_x.centers)[np.newaxis, np.newaxis, :]
siny = np.sin(np.pi * grid_y.centers)[np.newaxis, :, np.newaxis]
sinz = np.sin(np.pi * grid_z.centers)[:, np.newaxis, np.newaxis]

phi_ana = sinx * siny * sinz
rhs = -(np.pi ** 2) * (D_xx + D_yy + D_zz) * phi_ana


#sys.exit()
# solve problem
phi_res, iterations = solve_3d_simple(grid_x, grid_y, grid_z, phi, rhs, lam, D_xx, D_yy, D_zz)
err = (phi_ana - phi_res)[1:-1, 1:-1, 1:-1]
rmse = np.linalg.norm(err) / np.sqrt(err.size)
print(f"Type of 'phi_res': {type(phi_res)}, Iterations: {iterations}, RMSE: {rmse}")
