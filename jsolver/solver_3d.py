import jax
import jax.numpy as jnp
import numpy as np
from jsolver.boundary_handler_3d import do_BCs
from functools import partial

# set default dtype to float64 (solver does not work properly otherwise!)
jax.config.update('jax_enable_x64', True)

def solve_3d_simple(grid_x, grid_y, grid_z, phi, rhs, lam, D_xx, D_yy, D_zz):
    """
    Simplified, functionally pure wrapper function with positional arguments.

    See `solve_3d`.
    """
    return solve_3d(grid_x=grid_x, grid_y=grid_y, grid_z=grid_z, phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy, D_zz=D_zz)

def solve_3d_fixed_simple(grid_x, grid_y, grid_z, iters, phi, rhs, lam, D_xx, D_yy, D_zz):
    """
    Simplified, functionally pure wrapper function with positional arguments and fixed number of solver iterations (smoothing iterations for coarsest level are set to `1`).

    The argument `iters` controls the number of multigrid cycles.

    See `solve_3d`.
    """
    return solve_3d_fixed(grid_x=grid_x, grid_y=grid_y, grid_z=grid_z, iters_outer=iters, iters_inner=1, phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy, D_zz=D_zz)

def solve_3d_fixed(grid_x, grid_y, grid_z, iters_outer, iters_inner, phi, rhs, lam, D_xx, D_yy, D_zz):
    """
    Simplified, functionally pure wrapper function with positional arguments and fixed number of solver iterations.

    The argument `iters_outer` controls the number of multigrid cycles, `iters_inner` controls the number of smoothing iterations for the coarsest grid level.

    See `solve_3d`.
    """
    return solve_3d(grid_x=grid_x, grid_y=grid_y, grid_z=grid_z, fixed=True, max_iters_outer=iters_outer, max_iters_inner=iters_inner, phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy, D_zz=D_zz)

def solve_3d_fcycle_simple(grid_x, grid_y, grid_z, phi, rhs, lam, D_xx, D_yy, D_zz):
    """
    Simplified, functionally pure wrapper function with positional arguments and F-cycle multigrid.

    See `solve_3d`.
    """
    return solve_3d(grid_x=grid_x, grid_y=grid_y, grid_z=grid_z, phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy, D_zz=D_zz, fcycle=True)

def solve_3d_fixed_fcycle_simple(grid_x, grid_y, grid_z, iters, phi, rhs, lam, D_xx, D_yy, D_zz):
    """
    Simplified, functionally pure wrapper function with positional arguments, fixed number of solver iterations (smoothing iterations for coarsest level are set to `1`) and F-cycle multigrid.

    The argument `iters` controls the number of multigrid cycles.

    See `solve_3d`.
    """
    return solve_3d_fixed_fcycle(grid_x=grid_x, grid_y=grid_y, grid_z=grid_z, iters_outer=iters, iters_inner=1, phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy, D_zz=D_zz)

def solve_3d_fixed_fcycle(grid_x, grid_y, grid_z, iters_outer, iters_inner, phi, rhs, lam, D_xx, D_yy, D_zz):
    """
    Simplified, functionally pure wrapper function with positional arguments, fixed number of solver iterations and F-cycle multigrid.

    The argument `iters_outer` controls the number of multigrid cycles, `iters_inner` controls the number of smoothing iterations for the coarsest grid level.

    See `solve_3d`.
    """
    return solve_3d(grid_x=grid_x, grid_y=grid_y, grid_z=grid_z, fixed=True, max_iters_outer=iters_outer, max_iters_inner=iters_inner, phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy, D_zz=D_zz, fcycle=True)

def solve_3d_wcycle_simple(grid_x, grid_y, phi, rhs, lam, D_xx, D_yy, D_zz):
    """
    Simplified, functionally pure wrapper function with positional arguments and W-cycle multigrid.

    See `solve_3d`.
    """
    return solve_3d(grid_x=grid_x, grid_y=grid_y, phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy, D_zz=D_zz, gamma=2)

def solve_3d_fixed_wcycle_simple(grid_x, grid_y, iters, phi, rhs, lam, D_xx, D_yy, D_zz):
    """
    Simplified, functionally pure wrapper function with positional arguments, fixed number of solver iterations (smoothing iterations for coarsest level are set to `1`) and W-cycle multigrid.

    The argument `iters` controls the number of multigrid cycles.

    See `solve_3d`.
    """
    return solve_3d_fixed_wcycle(grid_x=grid_x, grid_y=grid_y, iters_outer=iters, iters_inner=1, phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy, D_zz=D_zz)

def solve_3d_fixed_wcycle(grid_x, grid_y, iters_outer, iters_inner, phi, rhs, lam, D_xx, D_yy, D_zz):
    """
    Simplified, functionally pure wrapper function with positional arguments, fixed number of solver iterations and W-cycle multigrid.

    The argument `iters_outer` controls the number of multigrid cycles, `iters_inner` controls the number of smoothing iterations for the coarsest grid level.

    See `solve_3d`.
    """
    return solve_3d(grid_x=grid_x, grid_y=grid_y, fixed=True, max_iters_outer=iters_outer, max_iters_inner=iters_inner, phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy, D_zz=D_zz, gamma=2)

@partial(jax.jit, static_argnames=('grid_x', 'grid_y', 'grid_z', 'fixed', 'epsilon', 'gamma', 'fcycle', 'nu1', 'nu2', 'max_iters_outer', 'max_iters_inner'))
def solve_3d(grid_x, grid_y, grid_z, *, fixed=False, epsilon=1e-12, gamma=1, fcycle=False, nu1=1, nu2=1, max_iters_outer=np.iinfo(np.int32).max, max_iters_inner=np.iinfo(np.int32).max, phi, rhs, lam, D_xx, D_yy, D_zz):
    """
    Wrapper function which is functionally pure and thus compatible with `jax` transformations.

    Initializes `Solver_3D` object and calls `solve` method.

    Arguments after grid objects are set to keyword-only (`*`) for full support of all optional arguments.

    Note that `grid_x` `grid_y`, and `grid_y` must be static arguments for `jax.jit` since array shapes and number of multigrid levels are computed from them.

    All arrays must be three-dimensional and of compatible shape: `lam`, `D_xx`, `D_yy`, `D_zz` can be scalars, but `D_xx`, `D_yy`, and `D_zz` must either all be scalar or all of the same shape as the other arrays; all non-scalars must be of the same shape.

    Args:
        grid_x (grid_1D): Linear grid for x-dimension (third index).
        grid_y (grid_1D): Linear grid for y-dimension (second index).
        grid_z (grid_1D): Linear grid for z-dimension (first index).
        fixed (bool, optional): Perform exactly `max_iters_outer` solver iterations and `max_iters_inner` smoothing iterations for the coarsest grid level instead of relying on a dynamic termination threshold. The function is compatible with reverse-mode automatic differentiation only if set to `True`. Defaults to `False`.
        epsilon (float, optional): Solver termination threshold. Defaults to `1e-12`.
        gamma (int, optional): Number of recursive calls between `defect` and `prolongate` in `multigrid_cycle`. Defaults to `1` (V-cycle).
        fcycle (bool, optional): Use F-cycle in `multigrid_cycle`. Can be combined with `gamma` into an "F-gamma-cycle". Defaults to `False`.
        skip_pre (bool, optional): Skip pre-smoothing for repeated cycles for finest/highest resolution (such that no consecutive smoothing calls happen). Only makes sense if `nu1 > 0` and `nu2 > 0`. Defaults to `False`.
        nu1 (int, optional): Number of pre-smoothing iterations (i.e., before `defect`, which relies on `nu1 > 0`). Defaults to `1`.
        nu2 (int, optional): Number of post-smoothing iterations (i.e., after `prolongate`). Defaults to `1`.
        max_iters_outer (int, optional): The maximum number of solver iterations (i.e., multigrid cycles). Defaults to `int32.max`.
        max_iters_inner (int, optional): The maximum number of smoothing iterations for coarsest grid level. Defaults to `int32.max`.
        phi (jax.Array): Initial value for the unknown `phi` (usually filled with zeros).
        rhs (jax.Array): Right-hand side of the equation.
        lam (float or jax.Array): `lambda` (either constant or spatially varying decay rate).
        D_xx_fine (float or jax.Array): Either `xx` diffusion constant (scalar) or `xx` diffusion tensor component for finest/highest resolution (if spatially varying diffusion is desired).
        D_yy_fine (float or jax.Array): Either `yy` diffusion constant (scalar) or `yy` diffusion tensor component for finest/highest resolution (if spatially varying diffusion is desired).
        D_zz_fine (float or jax.Array): Either `zz` diffusion constant (scalar) or `zz` diffusion tensor component for finest/highest resolution (if spatially varying diffusion is desired).
    Returns:
        phi (jax.Array): Computed solution for `phi`.
        count (int): Number of iterations performed.

    Raises:
        ValueError: If either grid is not linear, or the grid resolution is not a power of 2 or not efficient for multigrid configuration, or array shapes are incompatible.
    """
    solver = Solver_3D(grid_x=grid_x, grid_y=grid_y, grid_z=grid_z, fixed=fixed, epsilon=epsilon, gamma=gamma, fcycle=fcycle, nu1=nu1, nu2=nu2, max_iters_outer=max_iters_outer, max_iters_inner=max_iters_inner)
    return solver.solve(phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy, D_zz=D_zz)

class Solver_3D:
    """
    A class containing methods implementing an iterative geometric multigrid solver for the 3D steady-state diffusion–absorption problem.

    Members are used for convenience, as explicit parameter passing would be quite the hassle (for some functions).
    Note that all methods are impure due to the implicit `self` parameter and thus not (directly) compatible with `jax` transformations.
    However, all invariants are established after calling `compute_discretization` (i.e., no members are modified after this point); hence, `jax` transformations can be applied to subsequent methods by making `self` a static parameter.
    """

    def __init__(self, grid_x, grid_y, grid_z, fixed=False, epsilon=1e-12, gamma=1, fcycle=False, nu1=1, nu2=1, max_iters_outer=np.iinfo(np.int32).max, max_iters_inner=np.iinfo(np.int32).max):
        """
        Initialize members for solver configuration and compute important compile-time constants.

        Args:
            grid_x (grid_1D): Linear grid for x-dimension (third index).
            grid_y (grid_1D): Linear grid for y-dimension (second index).
            grid_z (grid_1D): Linear grid for z-dimension (first index).
            fixed (bool, optional): Perform exactly `max_iters_outer` solver iterations and `max_iters_inner` smoothing iterations for the coarsest grid level instead of relying on a dynamic termination threshold. The solver is compatible with reverse-mode automatic differentiation only if set to `True`. Defaults to `False`.
            epsilon (float, optional): Solver termination threshold. Defaults to `1e-12`.
            gamma (int, optional): Number of recursive calls between `defect` and `prolongate` in `multigrid_cycle`. Defaults to `1` (V-cycle).
            fcycle (bool, optional): Use F-cycle in `multigrid_cycle`. Can be combined with `gamma`. Defaults to `False`.
            skip_pre (bool, optional): Skip pre-smoothing for repeated cycles for finest/highest resolution (such that no consecutive smoothing calls happen). Defaults to `False`.
            nu1 (int, optional): Number of pre-smoothing iterations (i.e., before `defect`, which relies on `nu1 > 0`). Defaults to `1`.
            nu2 (int, optional): Number of post-smoothing iterations (i.e., after `prolongate`). Defaults to `1`.
            max_iters_outer (int, optional): The maximum number of solver iterations (i.e., multigrid cycles). Defaults to `int32.max`.
            max_iters_inner (int, optional): The maximum number of smoothing iterations for coarsest grid level. Defaults to `int32.max`.

        Raises:
            ValueError: If either grid is not linear, or the grid resolution is not a power of 2 or not efficient for multigrid configuration.
        """
        self.fixed = fixed
        self.max_iters_outer = max_iters_outer
        self.max_iters_inner = max_iters_inner
        self.epsilon = epsilon
        self.gamma = gamma
        self.fcycle = fcycle
        self.nu1 = nu1
        self.nu2 = nu2
        self.compute_shapes(grid_x.mx, grid_y.mx, grid_z.mx)
        self.compute_idel(grid_x, grid_y, grid_z)

    def compute_shapes(self, mx, my, mz):
        """
        Compute number of levels and shapes for each level statically (i.e., use `numpy` instead of `jax.numpy`).

        All array shapes (and types) must be compile-time constants w.r.t. `jax.jit`!

        Args:
            mx (int): Resolution of grid for x-dimension (third index). Must be a power of 2.
            my (int): Resolution of grid for y-dimension (second index). Must be a power of 2.
            mz (int): Resolution of grid for z-dimension (first index). Must be a power of 2.

        Raises:
            ValueError: If resolution of either grid is not a power of 2 or not efficient for multigrid configuration.
        """
        if np.bitwise_and(mx, mx - 1) > 0:
            raise ValueError(f"Resolution of x-grid is not a power of 2. Value: {mx}")
        if np.bitwise_and(my, my - 1) > 0:
            raise ValueError(f"Resolution of y-grid is not a power of 2. Value: {my}")
        if np.bitwise_and(mz, mz - 1) > 0:
            raise ValueError(f"Resolution of z-grid is not a power of 2. Value: {mz}")

        x_levels = (np.ceil(np.log2(mx)) + 1).astype(int)
        y_levels = (np.ceil(np.log2(my)) + 1).astype(int)
        z_levels = (np.ceil(np.log2(mz)) + 1).astype(int)
        self.levels = x_levels if x_levels < y_levels else y_levels
        self.levels = self.levels if self.levels < z_levels else z_levels
        self.mx_level = 2 ** np.arange(x_levels - 1, x_levels - self.levels - 1, -1, dtype=int)
        self.my_level = 2 ** np.arange(y_levels - 1, y_levels - self.levels - 1, -1, dtype=int)
        self.mz_level = 2 ** np.arange(z_levels - 1, z_levels - self.levels - 1, -1, dtype=int)
        self.arr_shape = [ (self.mz_level[level] + 3, self.my_level[level] + 3, self.mx_level[level] + 3) for level in range(self.levels) ]

        if self.mx_level[-1] > 5:
            raise ValueError(f"Resolution of x-grid not efficient for multigrid configuration. Resolution of coarsest level: {self.mx_level[-1]}")
        if self.my_level[-1] > 5:
            raise ValueError(f"Resolution of y-grid not efficient for multigrid configuration. Resolution of coarsest level: {self.my_level[-1]}")
        if self.mz_level[-1] > 5:
            raise ValueError(f"Resolution of z-grid not efficient for multigrid configuration. Resolution of coarsest level: {self.mz_level[-1]}")

    def compute_idel(self, grid_x, grid_y, grid_z):
        """
        Compute reciprocals of the linear grid's cell width for every level and set `self.idel_x`, `self.idel_y`, and `self.idel_z` accordingly.

        Args:
            grid_x (grid_1D): Linear grid for x-dimension (third index).
            grid_y (grid_1D): Linear grid for y-dimension (second index).
            grid_z (grid_1D): Linear grid for z-dimension (first index).

        Raises:
            ValueError: If either grid is not linear.
        """
        if not grid_x.is_linear or not grid_y.is_linear or not grid_z.is_linear:
            raise ValueError("Only linear grids are supported!")

        self.idel_x = grid_x.idel * 0.5 ** np.arange(self.levels)
        self.idel_y = grid_y.idel * 0.5 ** np.arange(self.levels)
        self.idel_z = grid_z.idel * 0.5 ** np.arange(self.levels)

    def compute_diffusion_tensor(self, D_xx_fine, D_yy_fine, D_zz_fine):
        """
        Compute diffusion tensor components for every level of the multigrid structure (if spatially varying diffusion is desired).

        Interpolated values between the gridpoints are used for the diffusion.

        Args:
            D_xx_fine (jax.Array): `xx` diffusion tensor component for finest/highest resolution.
            D_yy_fine (jax.Array): `yy` diffusion tensor component for finest/highest resolution.
            D_zz_fine (jax.Array): `zz` diffusion tensor component for finest/highest resolution.

        Returns:
            List of length `self.levels` for each component.
        """
        # relies on zeroing since not all values get set here
        D_xx = [ jnp.zeros(self.arr_shape[level]) for level in range(self.levels) ]
        D_yy = [ jnp.zeros(self.arr_shape[level]) for level in range(self.levels) ]
        D_zz = [ jnp.zeros(self.arr_shape[level]) for level in range(self.levels) ]
        D_xx[0] = D_xx_fine
        D_yy[0] = D_yy_fine
        D_zz[0] = D_zz_fine

        # set the values for the lower resolution directly from the higher one
        # Remark: gridpoint 0 represent the ghostcell, which in Picard is located at -1
        # -> therefore point 1 in this case represents point 0 for Picard
        for level in range(1, self.levels):
            D_xx[level] = D_xx[level].at[1:-1, 1:-1, 1:-1].set(D_xx[level - 1][1::2, 1::2, 1::2])
            D_yy[level] = D_yy[level].at[1:-1, 1:-1, 1:-1].set(D_yy[level - 1][1::2, 1::2, 1::2])
            D_zz[level] = D_zz[level].at[1:-1, 1:-1, 1:-1].set(D_zz[level - 1][1::2, 1::2, 1::2])
            # Transform D_xx to D_xxR, i.e., value between the two points - see also D_xxL and D_xxR in C++ solver
            D_xx[level] = D_xx[level].at[1:-1,1:-1, :-1].set(0.5 * (D_xx[level][1:-1,1:-1, 1:] + D_xx[level][1:-1,1:-1, :-1])) # average in x direction
            D_yy[level] = D_yy[level].at[1:-1,:-1, 1:-1].set(0.5 * (D_yy[level][1:-1,1:, 1:-1] + D_yy[level][1:-1,:-1, 1:-1])) # average in y direction
            D_zz[level] = D_zz[level].at[:-1,1:-1, 1:-1].set(0.5 * (D_zz[level][1:, 1:-1,1:-1] + D_zz[level][:-1, 1:-1,1:-1])) # average in z direction

        return D_xx, D_yy, D_zz

    def compute_lambda(self, lambda_fine):
        """
        Compute `lambda` (spatially varying decay rate) for every level of the multigrid structure.

        Args:
            lambda_fine (jax.Array): `lambda` for finest/highest resolution.

        Returns:
            List of length `self.levels`.
        """
        # relies on zeroing since not all values get set here
        lambda_level = [ jnp.zeros(self.arr_shape[level]) for level in range(self.levels) ]
        lambda_level[0] = lambda_fine

        # set the values for the lower resolution directly from the higher one:
        for level in range(1, self.levels):
            lambda_level[level] = lambda_level[level].at[1:-1, 1:-1, 1:-1].set(lambda_level[level - 1][1::2, 1::2, 1::2])

        return lambda_level

    def compute_discretization(self, lambda_fine, D_xx, D_yy, D_zz):
        """
        Compute discretization arrays for every level of the multigrid structure and store them as class members (lists of length `self.levels`).

        Args:
            lambda_fine (float or jax.Array): Either constant or `lambda` for finest/highest resolution.
            D_xx_fine (float or jax.Array): Either `xx` diffusion constant (scalar) or `xx` diffusion tensor component for finest/highest resolution (if spatially varying diffusion is desired).
            D_yy_fine (float or jax.Array): Either `yy` diffusion constant (scalar) or `yy` diffusion tensor component for finest/highest resolution (if spatially varying diffusion is desired).
            D_zz_fine (float or jax.Array): Either `zz` diffusion constant (scalar) or `zz` diffusion tensor component for finest/highest resolution (if spatially varying diffusion is desired).
        """
        sq_idel_x = self.idel_x ** 2
        sq_idel_y = self.idel_y ** 2
        sq_idel_z = self.idel_z ** 2

        lambda_level = None if self.scalar_lambda else self.compute_lambda(lambda_fine)

        if self.spatial_diffusion:
            D_xx, D_yy, D_zz = self.compute_diffusion_tensor(D_xx, D_yy, D_zz)
            # x-direction
            self.AMat_lcc = [ D_xx[level][1:-1, 1:-1, 0:-2] * sq_idel_x[level] for level in range(self.levels) ] # get D_xxL via shifting right by 1 position
            self.AMat_rcc = [ D_xx[level][1:-1, 1:-1, 1:-1] * sq_idel_x[level] for level in range(self.levels) ] # just ignore ghost cells for D_xxR
            # y-direction
            self.AMat_clc = [ D_yy[level][1:-1, 0:-2, 1:-1] * sq_idel_y[level] for level in range(self.levels) ]
            self.AMat_crc = [ D_yy[level][1:-1, 1:-1, 1:-1] * sq_idel_y[level] for level in range(self.levels) ]
            # z-direction
            self.AMat_ccl = [ D_zz[level][0:-2, 1:-1, 1:-1] * sq_idel_z[level] for level in range(self.levels) ]
            self.AMat_ccr = [ D_zz[level][1:-1, 1:-1, 1:-1] * sq_idel_z[level] for level in range(self.levels) ]
            # centered
            self.AMat_ccc = [ self.AMat_lcc[level] + self.AMat_rcc[level] + self.AMat_clc[level] + self.AMat_crc[level] + self.AMat_ccl[level] + self.AMat_ccr[level] + (lambda_fine if self.scalar_lambda else lambda_level[level][1:-1, 1:-1, 1:-1]) for level in range(self.levels) ]

        else:
            self.AMat_x = D_xx * sq_idel_x
            self.AMat_y = D_yy * sq_idel_y
            self.AMat_z = D_zz * sq_idel_z
            add = 2 * (self.AMat_x + self.AMat_y + self.AMat_z)
            self.AMat_ccc = add + lambda_fine if self.scalar_lambda else [ add[level] + lambda_level[level][1:-1, 1:-1, 1:-1] for level in range(self.levels) ]

    def solve(self, phi, rhs, lam, D_xx, D_yy, D_zz):
        """
        Multigrid solver for 3D steady-state diffusion–absorption problem.

        Iterate (i.e., call `multigrid_cycle`) until `distance(phi_old, phi) <= self.epsilon`.

        Args:
            phi (jax.Array): Initial value for the unknown `phi` (usually filled with zeros).
            rhs (jax.Array): Right-hand side of the equation.
            lam (float or jax.Array): `lambda` (either constant or spatially varying decay rate).
            D_xx_fine (float or jax.Array): Either `xx` diffusion constant (scalar) or `xx` diffusion tensor component for finest/highest resolution (if spatially varying diffusion is desired).
            D_yy_fine (float or jax.Array): Either `yy` diffusion constant (scalar) or `yy` diffusion tensor component for finest/highest resolution (if spatially varying diffusion is desired).
            D_zz_fine (float or jax.Array): Either `zz` diffusion constant (scalar) or `zz` diffusion tensor component for finest/highest resolution (if spatially varying diffusion is desired).

        Returns:
            phi (jax.Array): Computed solution for `phi`.
            count (int): Number of iterations performed.

        Raises:
            ValueError: If argument shapes or types are incompatible.
        """
        self.scalar_lambda = jnp.isscalar(lam)
        self.scalar_diffusion = jnp.isscalar(D_xx) and jnp.isscalar(D_yy) and jnp.isscalar(D_zz)
        self.spatial_diffusion = not self.scalar_diffusion

        if not phi.dtype == rhs.dtype == lam.dtype == D_xx.dtype == D_yy.dtype == D_zz.dtype == jnp.float64:
            raise ValueError("Incorrect configuration: dtype must be 'jnp.float64' for all arguments.")

        if not D_xx.shape == D_yy.shape == D_zz.shape or (self.scalar_lambda and not phi.shape == rhs.shape) or (not self.scalar_lambda and not phi.shape == rhs.shape == lam.shape) or (self.spatial_diffusion and not phi.shape == rhs.shape == D_xx.shape == D_yy.shape == D_zz.shape):
            raise ValueError(f"Shape mismatch: phi={phi.shape}, rhs={rhs.shape}, lam={lam.shape}, D_xx={D_xx.shape}, D_yy={D_yy.shape}, D_zz={D_zz.shape}.")

        self.compute_discretization(lam, D_xx, D_yy, D_zz)

        if self.fixed:
            phi = jax.lax.fori_loop(0, self.max_iters_outer, lambda i, phi: self.multigrid_cycle(do_BCs(phi), rhs, self.fcycle), phi)
            count = self.max_iters_outer
        else:
            def cond(arg):
                phi, count, dist = arg
                return jnp.logical_and(dist > self.epsilon, count < self.max_iters_outer)

            def body(arg):
                phi, count, dist = arg
                distance = partial(self.distance, phi)
                phi = self.multigrid_cycle(do_BCs(phi), rhs, self.fcycle)
                dist = distance(phi)
                #jax.debug.print("count: {}, dist = {}", count, dist)
                return (phi, count + 1, dist)

            phi, count, dist = jax.lax.while_loop(cond, body, (phi, 0, jnp.inf))

        return phi, count

    def distance(self, phi_old, phi):
        """
        Compute distance (here: infinity norm of vectors) to previous step.

        A step (iteration) corresponds to a call of `multigrid_cycle` within the while loop in `solve`.

        Args:
           phi_old (jax.Array): Previous value (i.e., before current call to `multigrid_cycle`).
            phi (jax.Array): Return value of `multigrid_cycle`.

        Returns:
            float: Maximum absolute difference between `phi` and `phi_old`.
        """
        return jnp.max(jnp.abs(phi_old - phi))

    def compute_residual(self, phi, rhs): # pragma: no cover
        """
        UNUSED.

        Compute residual (3D array) by testing the difference of LHS and RHS of the equation.

        Args:
            phi (jax.Array): Current value for the unknown `phi`.
            rhs (jax.Array): Right-hand side of the equation.

        Returns:
            jax.Array: residual array
        """
        if self.spatial_diffusion:
            residual = self.AMat_ccc[0][1:-1, 1:-1, 1:-1] * phi[2:-2, 2:-2, 2:-2] - self.AMat_lcc[0][1:-1, 1:-1, 1:-1] * phi[2:-2, 2:-2, 1:-3] - self.AMat_rcc[0][1:-1, 1:-1, 1:-1] * phi[2:-2, 2:-2, 3:-1] - self.AMat_clc[0][1:-1, 1:-1, 1:-1] * phi[2:-2, 1:-3, 2:-2] - self.AMat_crc[0][1:-1, 1:-1, 1:-1] * phi[2:-2, 3:-1, 2:-2] - self.AMat_clc[0][1:-1, 1:-1, 1:-1] * phi[1:-3, 2:-2, 2:-2] - self.AMat_crc[0][1:-1, 1:-1, 1:-1] * phi[3:-1, 2:-2, 2:-2] + rhs[2:-2, 2:-2, 2:-2]
        else:
            residual = (self.AMat_ccc[0] if self.scalar_lambda else self.AMat_ccc[0][1:-1, 1:-1, 1:-1]) * phi[2:-2, 2:-2, 2:-2] - self.AMat_x[0] * phi[2:-2, 2:-2, 1:-3] - self.AMat_x[0] * phi[2:-2, 2:-2, 3:-1] - self.AMat_y[0] * phi[2:-2, 1:-3, 2:-2] - self.AMat_y[0] * phi[2:-2, 3:-1, 2:-2] - self.AMat_z[0] * phi[1:-3, 2:-2, 2:-2] - self.AMat_z[0] * phi[3:-1, 2:-2, 2:-2] + rhs[2:-2, 2:-2, 2:-2]
        return residual

    def compute_error(self, phi, rhs): # pragma: no cover
        """
        UNUSED.

        Compute discrete error (maximum absolute value in residual array).

        Args:
            phi (jax.Array): Current value for the unknown `phi`.
            rhs (jax.Array): Right-hand side of the equation.

        Returns:
            float: discrete error
        """
        return jnp.max(jnp.abs(self.compute_residual(phi, rhs)))

    def get_l2_residual(self, phi, rhs): # pragma: no cover
        """
        UNUSED.

        Compute L2 norm of residual array.

        Args:
            phi (jax.Array): Current value for the unknown `phi`.
            rhs (jax.Array): Right-hand side of the equation.

        Returns:
            float: L2 norm
        """
        return jnp.linalg.norm(self.compute_residual(phi, rhs))

    # same in 2D and 3D
    def multigrid_cycle(self, phi, rhs, fcycle=False, level=0):
        """
        Recursively defined multigrid cycle.

        Behavior is controlled by members:
            self.fixed (bool): Perform fixed number (`self.max_iters_inner`) of smoothing iterations for coarsest grid level.
            self.nu1 (int): Smoothing iterations before `defect` / restriction operation (which relies on `self.nu1 > 0`).
            self.nu2 (int): Smoothing iterations after `prolongate`.
            self.gamma (int): Number of recursive calls between `defect` and `prolongate` (1: V-cycle, 2: W-cycle, ...).

        Args:
            phi (jax.Array): Current value for the unknown `phi`.
            rhs (jax.Array): Current right-hand side of the equation.
            level (int, optional): Current multigrid level (`0`: finest/highest resolution, ..., `self.levels-1`: coarsest/lowest resolution). Defaults to `0`.
            fcycle (bool, optional): Use F-cycle (i.e., do two recursive calls where the first one uses the F-gamma-cycle recursively and the second one performs a regular gamma-cycle). Defaults to `False`.

        Returns:
            phi (jax.Array): New value for the unknown `phi`.
        """
        if level == self.levels - 1:
            if self.fixed:
                phi = jax.lax.fori_loop(0, self.max_iters_inner, lambda i, phi: self.red_black_gauss_seidel(phi, rhs, level), phi)
            else:
                def cond(arg):
                    phi, count, dist = arg
                    return jnp.logical_and(dist > self.epsilon, count < self.max_iters_inner)

                def body(arg):
                    phi, count, dist = arg
                    distance = partial(self.distance, phi)
                    phi = self.red_black_gauss_seidel(phi, rhs, level)
                    dist = distance(phi)
                    #jax.debug.print("coarsest level -- count: {}, dist = {}", count, dist)
                    return (phi, count + 1, dist)

                phi, _, _ = jax.lax.while_loop(cond, body, (phi, 0, jnp.inf))
        else:
            phi = jax.lax.fori_loop(0, self.nu1, lambda i, phi: self.red_black_gauss_seidel(phi, rhs, level), phi)
            rhs_c = self.defect(phi, rhs, level)
            phi_c = jnp.full(self.arr_shape[level + 1], 0.)
            if fcycle:
                phi_c = jax.lax.fori_loop(0, self.gamma, lambda i, phi_c: self.multigrid_cycle(phi_c, rhs_c, True, level + 1), phi_c)
            phi_c = jax.lax.fori_loop(0, self.gamma, lambda i, phi_c: self.multigrid_cycle(phi_c, rhs_c, False, level + 1), phi_c)
            phi = jax.lax.fori_loop(0, self.nu2, lambda i, phi: self.red_black_gauss_seidel(phi, rhs, level), phi + self.prolongate(phi_c, level))

        return phi

    def defect(self, phi, rhs, level):
        """
        Compute defect for current solution and assign to coarser grid.

        Corresponds to residual computation and full-weighting restriction.

        Assumes that at least one RBGS pre-smoothing step was performed (i.e., that `self.nu1 > 0`).

        Args:
            phi (jax.Array): Current value for the unknown `phi`.
            rhs (jax.Array): Current right-hand side of the equation.
            level (int): Current multigrid level.

        Returns:
            rhs_c (jax.Array): Residual error downsampled to coarser grid.
        """
        del_fine = jnp.empty(rhs.shape)
        rhs_c = jnp.empty(self.arr_shape[level + 1])

        if self.spatial_diffusion:
            del_fine = del_fine.at[1:-1, 1:-1, 1:-1].set(self.AMat_ccc[level] * phi[1:-1, 1:-1, 1:-1] - self.AMat_lcc[level] * phi[1:-1, 1:-1, :-2] - self.AMat_rcc[level] * phi[1:-1, 1:-1, 2:] - self.AMat_clc[level] * phi[1:-1, :-2, 1:-1] - self.AMat_crc[level] * phi[1:-1, 2:, 1:-1]  - self.AMat_ccl[level] * phi[:-2, 1:-1, 1:-1] - self.AMat_ccr[level] * phi[2:, 1:-1, 1:-1] + rhs[1:-1, 1:-1, 1:-1])

        else:
            del_fine = del_fine.at[1:-1, 1:-1, 1:-1].set(self.AMat_ccc[level] * phi[1:-1, 1:-1, 1:-1] - self.AMat_x[level] * phi[1:-1, 1:-1, :-2] - self.AMat_x[level] * phi[1:-1, 1:-1, 2:] - self.AMat_y[level] * phi[1:-1, :-2, 1:-1] - self.AMat_y[level] * phi[1:-1, 2:, 1:-1] - self.AMat_z[level] * phi[:-2, 1:-1, 1:-1] - self.AMat_z[level] * phi[2:, 1:-1, 1:-1] + rhs[1:-1, 1:-1, 1:-1])

        del_fine = do_BCs(del_fine)
        # full-weighting restriction, exploiting that black points have a defect of zero after RBGS pre-smoothing
        rhs_c = rhs_c.at[1:-1, 1:-1, 1:-1].set(0.03125 * (del_fine[1:-1:2, 2::2, 2::2] + del_fine[1:-1:2, 2::2, :-2:2] + del_fine[1:-1:2, :-2:2, 2::2] + del_fine[1:-1:2, :-2:2, :-2:2] + del_fine[2::2, 1:-1:2, 2::2] + del_fine[2::2, 1:-1:2, :-2:2] + del_fine[:-2:2, 1:-1:2, 2::2] + del_fine[:-2:2, 1:-1:2, :-2:2] + del_fine[2::2, 2::2, 1:-1:2] + del_fine[2::2, :-2:2, 1:-1:2] + del_fine[:-2:2, 2::2, 1:-1:2] + del_fine[:-2:2, :-2:2, 1:-1:2]) + 0.125 * del_fine[1:-1:2, 1:-1:2, 1:-1:2])
        rhs_c = do_BCs(rhs_c)

        return rhs_c

    def prolongate(self, phi_c, level):
        """
        Interpolate the computed coarse-grid correction into a finer grid.

        Args:
            phi_c (jax.Array): Computed coarse-grid correction (from `level + 1`)
            level (int): Current multigrid level, corresponding to finer grid.

        Returns:
            corr (jax.Array): Interpolated correction (to be added to the current value of `phi` on the finer grid).
        """
        corr = jnp.empty(self.arr_shape[level])

        corr = corr.at[1:-1:2, 1:-1:2, 1:-1:2].set(phi_c[1:-1, 1:-1, 1:-1]) # direct copy
        corr = corr.at[1:-1:2, 1:-1:2, 2:-1:2].set(0.5 * (corr[1:-1:2, 1:-1:2, 3::2] + corr[1:-1:2, 1:-1:2, 1:-2:2])) # linear interpolation in x
        corr = corr.at[1:-1:2, 2:-1:2, 1:-1:2].set(0.5 * (corr[1:-1:2, 3::2, 1:-1:2] + corr[1:-1:2, 1:-2:2, 1:-1:2])) # linear interpolation in y
        corr = corr.at[2:-1:2, 1:-1:2, 1:-1:2].set(0.5 * (corr[3::2, 1:-1:2, 1:-1:2] + corr[1:-2:2, 1:-1:2, 1:-1:2])) # linear interpolation in z
        if self.nu2 <= 0: # skip points overridden in red RBGS sweep; could skip linear & trilinear parts instead if post-smoothing RBGS sweep order is changed to black-first
            corr = corr.at[1:-1:2, 2:-1:2, 2:-1:2].set(0.25 * (corr[1:-1:2, 3::2, 3::2] + corr[1:-1:2, 1:-2:2, 3::2] + corr[1:-1:2, 3::2, 1:-2:2] + corr[1:-1:2, 1:-2:2, 1:-2:2])) # bilinear interpolation in x and y
            corr = corr.at[2:-1:2, 1:-1:2, 2:-1:2].set(0.25 * (corr[3::2, 1:-1:2, 3::2] + corr[1:-2:2, 1:-1:2, 3::2] + corr[3::2, 1:-1:2, 1:-2:2] + corr[1:-2:2, 1:-1:2, 1:-2:2])) # bilinear interpolation in x and z
            corr = corr.at[2:-1:2, 2:-1:2, 1:-1:2].set(0.25 * (corr[3::2, 3::2, 1:-1:2] + corr[1:-2:2, 3::2, 1:-1:2] + corr[3::2, 1:-2:2, 1:-1:2] + corr[1:-2:2, 1:-2:2, 1:-1:2])) # bilinear interpolation in y and z
        corr = corr.at[2:-1:2, 2:-1:2, 2:-1:2].set(0.125 * (corr[3::2, 3::2, 3::2] + corr[1:-2:2, 3::2, 3::2] + corr[3::2, 1:-2:2, 3::2] + corr[1:-2:2, 1:-2:2, 3::2] + corr[3::2, 3::2, 1:-2:2] + corr[1:-2:2, 3::2, 1:-2:2] + corr[3::2, 1:-2:2, 1:-2:2] + corr[1:-2:2, 1:-2:2, 1:-2:2])) # trilinear interpolation 

        return corr

    def red_black_gauss_seidel(self, phi, rhs, level):
        """
        Red-Black Gauss-Seidel (RBGS) routine for smoothing (i.e., reducing high-frequency errors).

        Args:
            phi (jax.Array): Current value for the unknown `phi`.
            rhs (jax.Array): Current right-hand side of the equation.
            level (int): Current multigrid level.

        Returns:
            phi (jax.Array): New value for the unknown `phi`.
        """
        phi = do_BCs(phi)

        for (lx, ly, lz) in [ (0,0,0), (0,1,1), (1,1,0), (1,0,1), (1,0,0), (0,1,0), (0,0,1), (1,1,1) ]:
            if self.spatial_diffusion:
                phi = phi.at[lz+1:-1:2, ly+1:-1:2, lx+1:-1:2].set((self.AMat_ccr[level][lz::2, ly::2, lx::2] * phi[lz+2::2, ly+1:-1:2, lx+1:-1:2] + self.AMat_ccl[level][lz::2, ly::2, lx::2] * phi[lz:-2:2, ly+1:-1:2, lx+1:-1:2] +
                                                                   self.AMat_crc[level][lz::2, ly::2, lx::2] * phi[lz+1:-1:2, ly+2::2, lx+1:-1:2] + self.AMat_clc[level][lz::2, ly::2, lx::2] * phi[lz+1:-1:2, ly:-2:2, lx+1:-1:2] +
                                                                   self.AMat_rcc[level][lz::2, ly::2, lx::2] * phi[lz+1:-1:2, ly+1:-1:2, lx+2::2] + self.AMat_lcc[level][lz::2, ly::2, lx::2] * phi[lz+1:-1:2, ly+1:-1:2, lx:-2:2] -
                                                                   rhs[lz+1:-1:2, ly+1:-1:2, lx+1:-1:2]) / self.AMat_ccc[level][lz::2, ly::2, lx::2])
            else:
                phi = phi.at[lz+1:-1:2, ly+1:-1:2, lx+1:-1:2].set((self.AMat_z[level] * phi[lz+2::2, ly+1:-1:2, lx+1:-1:2] + self.AMat_z[level] * phi[lz:-2:2, ly+1:-1:2, lx+1:-1:2] +
                                                                   self.AMat_y[level] * phi[lz+1:-1:2, ly+2::2, lx+1:-1:2] + self.AMat_y[level] * phi[lz+1:-1:2, ly:-2:2, lx+1:-1:2] +
                                                                   self.AMat_x[level] * phi[lz+1:-1:2, ly+1:-1:2, lx+2::2] + self.AMat_x[level] * phi[lz+1:-1:2, ly+1:-1:2, lx:-2:2] -
                                                                   rhs[lz+1:-1:2, ly+1:-1:2, lx+1:-1:2]) / (self.AMat_ccc[level] if self.scalar_lambda else self.AMat_ccc[level][lz::2, ly::2, lx::2]))
            phi = do_BCs(phi)

        return phi

