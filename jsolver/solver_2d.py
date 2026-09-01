import jax
import jax.numpy as jnp
import numpy as np
from jsolver.boundary_handler_2d import do_BCs
from functools import partial

# set default dtype to float64 (solver does not work properly otherwise!)
jax.config.update('jax_enable_x64', True)

def solve_2d_simple(grid_x, grid_y, phi, rhs, lam, D_xx, D_yy, D_xy=None):
    """
    Simplified, functionally pure wrapper function with positional arguments.

    See `solve_2d`.
    """
    return solve_2d(grid_x=grid_x, grid_y=grid_y, phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy, D_xy=D_xy)

def solve_2d_fixed_simple(grid_x, grid_y, iters, phi, rhs, lam, D_xx, D_yy, D_xy=None):
    """
    Simplified, functionally pure wrapper function with positional arguments and fixed number of solver iterations (smoothing iterations for coarsest level are set to `1`).

    The argument `iters` controls the number of multigrid cycles.

    See `solve_2d`.
    """
    return solve_2d_fixed(grid_x=grid_x, grid_y=grid_y, iters_outer=iters, iters_inner=1, phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy, D_xy=D_xy)

def solve_2d_fixed(grid_x, grid_y, iters_outer, iters_inner, phi, rhs, lam, D_xx, D_yy, D_xy=None):
    """
    Simplified, functionally pure wrapper function with positional arguments and fixed number of solver iterations.

    The argument `iters_outer` controls the number of multigrid cycles, `iters_inner` controls the number of smoothing iterations for the coarsest grid level.

    See `solve_2d`.
    """
    return solve_2d(grid_x=grid_x, grid_y=grid_y, fixed=True, max_iters_outer=iters_outer, max_iters_inner=iters_inner, phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy, D_xy=D_xy)

def solve_2d_skip_pre_simple(grid_x, grid_y, phi, rhs, lam, D_xx, D_yy, D_xy=None):
    """
    Simplified, functionally pure wrapper function with positional arguments.

    Pre-Smoothing is skipped for repeated V-cycles such that no consecutive smoothing operations happen for the finest/highest resolution. Otherwise, post-smoothing from previous cycle would be immediately followed by pre-smoothing.

    See `solve_2d`.
    """
    return solve_2d(grid_x=grid_x, grid_y=grid_y, phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy, D_xy=D_xy, skip_pre=True)

def solve_2d_fixed_skip_pre_simple(grid_x, grid_y, iters, phi, rhs, lam, D_xx, D_yy, D_xy=None):
    """
    Simplified, functionally pure wrapper function with positional arguments and fixed number of solver iterations (smoothing iterations for coarsest level are set to `1`).

    The argument `iters` controls the number of multigrid cycles.

    Pre-Smoothing is skipped for repeated V-cycles such that no consecutive smoothing operations happen for the finest/highest resolution. Otherwise, post-smoothing from previous cycle would be immediately followed by pre-smoothing.

    See `solve_2d`.
    """
    return solve_2d_fixed_skip_pre(grid_x=grid_x, grid_y=grid_y, iters_outer=iters, iters_inner=1, phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy, D_xy=D_xy)

def solve_2d_fixed_skip_pre(grid_x, grid_y, iters_outer, iters_inner, phi, rhs, lam, D_xx, D_yy, D_xy=None):
    """
    Simplified, functionally pure wrapper function with positional arguments and fixed number of solver iterations.

    The argument `iters_outer` controls the number of multigrid cycles, `iters_inner` controls the number of smoothing iterations for the coarsest grid level.

    Pre-Smoothing is skipped for repeated V-cycles such that no consecutive smoothing operations happen for the finest/highest resolution. Otherwise, post-smoothing from previous cycle would be immediately followed by pre-smoothing.

    See `solve_2d`.
    """
    return solve_2d(grid_x=grid_x, grid_y=grid_y, fixed=True, max_iters_outer=iters_outer, max_iters_inner=iters_inner, phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy, D_xy=D_xy, skip_pre=True)

def solve_2d_fcycle_simple(grid_x, grid_y, phi, rhs, lam, D_xx, D_yy, D_xy=None):
    """
    Simplified, functionally pure wrapper function with positional arguments and F-cycle multigrid.

    See `solve_2d`.
    """
    return solve_2d(grid_x=grid_x, grid_y=grid_y, phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy, D_xy=D_xy, fcycle=True)

def solve_2d_fixed_fcycle_simple(grid_x, grid_y, iters, phi, rhs, lam, D_xx, D_yy, D_xy=None):
    """
    Simplified, functionally pure wrapper function with positional arguments, fixed number of solver iterations (smoothing iterations for coarsest level are set to `1`) and F-cycle multigrid.

    The argument `iters` controls the number of multigrid cycles.

    See `solve_2d`.
    """
    return solve_2d_fixed_fcycle(grid_x=grid_x, grid_y=grid_y, iters_outer=iters, iters_inner=1, phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy, D_xy=D_xy)

def solve_2d_fixed_fcycle(grid_x, grid_y, iters_outer, iters_inner, phi, rhs, lam, D_xx, D_yy, D_xy=None):
    """
    Simplified, functionally pure wrapper function with positional arguments, fixed number of solver iterations and F-cycle multigrid.

    The argument `iters_outer` controls the number of multigrid cycles, `iters_inner` controls the number of smoothing iterations for the coarsest grid level.

    See `solve_2d`.
    """
    return solve_2d(grid_x=grid_x, grid_y=grid_y, fixed=True, max_iters_outer=iters_outer, max_iters_inner=iters_inner, phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy, D_xy=D_xy, fcycle=True)

def solve_2d_fcycle_skip_pre_simple(grid_x, grid_y, phi, rhs, lam, D_xx, D_yy, D_xy=None):
    """
    Simplified, functionally pure wrapper function with positional arguments and F-cycle multigrid.

    Pre-Smoothing is skipped for repeated F-cycles such that no consecutive smoothing operations happen for the finest/highest resolution. Otherwise, post-smoothing from previous cycle would be immediately followed by pre-smoothing.

    See `solve_2d`.
    """
    return solve_2d(grid_x=grid_x, grid_y=grid_y, phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy, D_xy=D_xy, fcycle=True, skip_pre=True)

def solve_2d_fixed_fcycle_skip_pre_simple(grid_x, grid_y, iters, phi, rhs, lam, D_xx, D_yy, D_xy=None):
    """
    Simplified, functionally pure wrapper function with positional arguments, fixed number of solver iterations and F-cycle multigrid (smoothing iterations for coarsest level are set to `1`).

    The argument `iters` controls the number of multigrid cycles.

    Pre-Smoothing is skipped for repeated F-cycles such that no consecutive smoothing operations happen for the finest/highest resolution. Otherwise, post-smoothing from previous cycle would be immediately followed by pre-smoothing.

    See `solve_2d`.
    """
    return solve_2d_fixed_fcycle_skip_pre(grid_x=grid_x, grid_y=grid_y, iters_outer=iters, iters_inner=1, phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy, D_xy=D_xy)

def solve_2d_fixed_fcycle_skip_pre(grid_x, grid_y, iters_outer, iters_inner, phi, rhs, lam, D_xx, D_yy, D_xy=None):
    """
    Simplified, functionally pure wrapper function with positional arguments, fixed number of solver iterations and F-cycle multigrid.

    The argument `iters_outer` controls the number of multigrid cycles, `iters_inner` controls the number of smoothing iterations for the coarsest grid level.

    Pre-Smoothing is skipped for repeated F-cycles such that no consecutive smoothing operations happen for the finest/highest resolution. Otherwise, post-smoothing from previous cycle would be immediately followed by pre-smoothing.

    See `solve_2d`.
    """
    return solve_2d(grid_x=grid_x, grid_y=grid_y, fixed=True, max_iters_outer=iters_outer, max_iters_inner=iters_inner, phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy, D_xy=D_xy, skip_pre=True, fcycle=True)

def solve_2d_wcycle_simple(grid_x, grid_y, phi, rhs, lam, D_xx, D_yy, D_xy=None):
    """
    Simplified, functionally pure wrapper function with positional arguments and W-cycle multigrid.

    See `solve_2d`.
    """
    return solve_2d(grid_x=grid_x, grid_y=grid_y, phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy, D_xy=D_xy, gamma=2)

def solve_2d_fixed_wcycle_simple(grid_x, grid_y, iters, phi, rhs, lam, D_xx, D_yy, D_xy=None):
    """
    Simplified, functionally pure wrapper function with positional arguments, fixed number of solver iterations (smoothing iterations for coarsest level are set to `1`) and W-cycle multigrid.

    The argument `iters` controls the number of multigrid cycles.

    See `solve_2d`.
    """
    return solve_2d_fixed_wcycle(grid_x=grid_x, grid_y=grid_y, iters_outer=iters, iters_inner=1, phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy, D_xy=D_xy)

def solve_2d_fixed_wcycle(grid_x, grid_y, iters_outer, iters_inner, phi, rhs, lam, D_xx, D_yy, D_xy=None):
    """
    Simplified, functionally pure wrapper function with positional arguments, fixed number of solver iterations and W-cycle multigrid.

    The argument `iters_outer` controls the number of multigrid cycles, `iters_inner` controls the number of smoothing iterations for the coarsest grid level.

    See `solve_2d`.
    """
    return solve_2d(grid_x=grid_x, grid_y=grid_y, fixed=True, max_iters_outer=iters_outer, max_iters_inner=iters_inner, phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy, D_xy=D_xy, gamma=2)

def solve_2d_wcycle_skip_pre_simple(grid_x, grid_y, phi, rhs, lam, D_xx, D_yy, D_xy=None):
    """
    Simplified, functionally pure wrapper function with positional arguments and W-cycle multigrid.

    Pre-Smoothing is skipped for repeated W-cycles such that no consecutive smoothing operations happen for the finest/highest resolution. Otherwise, post-smoothing from previous cycle would be immediately followed by pre-smoothing.

    See `solve_2d`.
    """
    return solve_2d(grid_x=grid_x, grid_y=grid_y, phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy, D_xy=D_xy, gamma=2, skip_pre=True)

def solve_2d_fixed_wcycle_skip_pre_simple(grid_x, grid_y, iters, phi, rhs, lam, D_xx, D_yy, D_xy=None):
    """
    Simplified, functionally pure wrapper function with positional arguments, fixed number of solver iterations and W-cycle multigrid (smoothing iterations for coarsest level are set to `1`).

    The argument `iters` controls the number of multigrid cycles.

    Pre-Smoothing is skipped for repeated W-cycles such that no consecutive smoothing operations happen for the finest/highest resolution. Otherwise, post-smoothing from previous cycle would be immediately followed by pre-smoothing.

    See `solve_2d`.
    """
    return solve_2d_fixed_wcycle_skip_pre(grid_x=grid_x, grid_y=grid_y, iters_outer=iters, iters_inner=1, phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy, D_xy=D_xy)

def solve_2d_fixed_wcycle_skip_pre(grid_x, grid_y, iters_outer, iters_inner, phi, rhs, lam, D_xx, D_yy, D_xy=None):
    """
    Simplified, functionally pure wrapper function with positional arguments, fixed number of solver iterations and W-cycle multigrid.

    The argument `iters_outer` controls the number of multigrid cycles, `iters_inner` controls the number of smoothing iterations for the coarsest grid level.

    Pre-Smoothing is skipped for repeated W-cycles such that no consecutive smoothing operations happen for the finest/highest resolution. Otherwise, post-smoothing from previous cycle would be immediately followed by pre-smoothing.

    See `solve_2d`.
    """
    return solve_2d(grid_x=grid_x, grid_y=grid_y, fixed=True, max_iters_outer=iters_outer, max_iters_inner=iters_inner, phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy, D_xy=D_xy, skip_pre=True, gamma=2)

@partial(jax.jit, static_argnames=('grid_x', 'grid_y', 'fixed', 'epsilon', 'gamma', 'fcycle', 'skip_pre', 'nu1', 'nu2', 'max_iters_outer', 'max_iters_inner'))
def solve_2d(grid_x, grid_y, *, fixed=False, epsilon=1e-12, gamma=1, fcycle=False, skip_pre=False, nu1=1, nu2=1, max_iters_outer=np.iinfo(np.int32).max, max_iters_inner=np.iinfo(np.int32).max, phi, rhs, lam, D_xx, D_yy, D_xy=None):
    """
    Wrapper function which is functionally pure and thus compatible with `jax` transformations.

    Initializes `Solver_2D` object and calls `solve` method.

    Arguments after grid objects are set to keyword-only (`*`) for full support of all optional arguments.

    Note that `grid_x` and `grid_y` must be static arguments for `jax.jit` since array shapes and number of multigrid levels are computed from them.

    All arrays must be two-dimensional and of compatible shape: `lam`, `D_xx`, and `D_yy` can be scalars, but `D_xx` and `D_yy` must either both be scalar or both of the same shape as the other arrays; all non-scalars must be of the same shape.

    Args:
        grid_x (grid_1D): Linear grid for x-dimension (column-dimension).
        grid_y (grid_1D): Linear grid for y-dimension (row-dimension).
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
        D_xy_fine (jax.Array, optional): `xy` diffusion tensor component for finest/highest resolution (if spatially varying diffusion with off-diagonal component is desired). Defaults to `None`.

    Returns:
        phi (jax.Array): Computed solution for `phi`.
        count (int): Number of iterations performed.

    Raises:
        ValueError: If either grid is not linear, or the grid resolution is not a power of 2 or not efficient for multigrid configuration, or array shapes are incompatible.
    """
    solver = Solver_2D(grid_x=grid_x, grid_y=grid_y, fixed=fixed, epsilon=epsilon, gamma=gamma, fcycle=fcycle, skip_pre=skip_pre, nu1=nu1, nu2=nu2, max_iters_outer=max_iters_outer, max_iters_inner=max_iters_inner)
    return solver.solve(phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy, D_xy=D_xy)

class Solver_2D:
    """
    A class containing methods implementing an iterative geometric multigrid solver for the 2D steady-state diffusion–absorption problem.

    Members are used for convenience, as explicit parameter passing would be quite the hassle (for some functions).
    Note that all methods are impure due to the implicit `self` parameter and thus not (directly) compatible with `jax` transformations.
    However, all invariants are established after calling `compute_discretization` (i.e., no members are modified after this point); hence, `jax` transformations can be applied to subsequent methods by making `self` a static parameter.
    """

    def __init__(self, grid_x, grid_y, fixed=False, epsilon=1e-12, gamma=1, fcycle=False, skip_pre=False, nu1=1, nu2=1, max_iters_outer=np.iinfo(np.int32).max, max_iters_inner=np.iinfo(np.int32).max):
        """
        Initialize members for solver configuration and compute important compile-time constants.

        Args:
            grid_x (grid_1D): Linear grid for x-dimension (column-dimension).
            grid_y (grid_1D): Linear grid for y-dimension (row-dimension).
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
            ValueError: If either grid is not linear, or the grid resolution is not a power of 2 or not efficient for multigrid configuration, or array shapes are incompatible.
        """
        self.fixed = fixed
        self.max_iters_outer = max_iters_outer
        self.max_iters_inner = max_iters_inner
        self.epsilon = epsilon
        self.gamma = gamma
        self.fcycle = fcycle
        self.skip_pre = skip_pre
        self.nu1 = nu1
        self.nu2 = nu2
        self.compute_shapes(grid_x.mx, grid_y.mx)
        self.compute_idel(grid_x, grid_y)

    def compute_shapes(self, mx, my):
        """
        Compute number of levels and shapes for each level statically (i.e., use `numpy` instead of `jax.numpy`).

        All array shapes (and types) must be compile-time constants w.r.t. `jax.jit`!

        Args:
            mx (int): Resolution of grid for x-dimension (column-dimension). Must be a power of 2.
            my (int): Resolution of grid for y-dimension (row-dimension). Must be a power of 2.

        Raises:
            ValueError: If resolution of either grid is not a power of 2 or not efficient for multigrid configuration.
        """
        if np.bitwise_and(mx, mx - 1) > 0:
            raise ValueError(f"Resolution of x-grid is not a power of 2. Value: {mx}")
        if np.bitwise_and(my, my - 1) > 0:
            raise ValueError(f"Resolution of y-grid is not a power of 2. Value: {my}")

        x_levels = (np.ceil(np.log2(mx)) + 1).astype(int)
        y_levels = (np.ceil(np.log2(my)) + 1).astype(int)
        self.levels = x_levels if x_levels < y_levels else y_levels
        self.mx_level = 2 ** np.arange(x_levels - 1, x_levels - self.levels - 1, -1, dtype=int)
        self.my_level = 2 ** np.arange(y_levels - 1, y_levels - self.levels - 1, -1, dtype=int)
        self.arr_shape = [ (self.my_level[level] + 3, self.mx_level[level] + 3) for level in range(self.levels) ]

        if self.mx_level[-1] > 5:
            raise ValueError(f"Resolution of x-grid not efficient for multigrid configuration. Resolution of coarsest level: {self.mx_level[-1]}")
        if self.my_level[-1] > 5:
            raise ValueError(f"Resolution of y-grid not efficient for multigrid configuration. Resolution of coarsest level: {self.my_level[-1]}")

    def compute_idel(self, grid_x, grid_y):
        """
        Compute reciprocals of the linear grid's cell width for every level and set `self.idel_x` and `self.idel_y` accordingly.

        Args:
            grid_x (grid_1D): Linear grid for x-dimension (column-dimension).
            grid_y (grid_1D): Linear grid for y-dimension (row-dimension).

        Raises:
            ValueError: If either grid is not linear.
        """
        if not grid_x.is_linear or not grid_y.is_linear:
            raise ValueError("Only linear grids are supported!")

        self.idel_x = grid_x.idel * 0.5 ** np.arange(self.levels)
        self.idel_y = grid_y.idel * 0.5 ** np.arange(self.levels)

    def compute_diffusion_tensor(self, D_xx_fine, D_yy_fine, D_xy_fine=None):
        """
        Compute diffusion tensor components for every level of the multigrid structure (if spatially varying diffusion is desired).

        Interpolated values between the gridpoints are used for the diffusion.

        Args:
            D_xx_fine (jax.Array): `xx` diffusion tensor component for finest/highest resolution.
            D_yy_fine (jax.Array): `yy` diffusion tensor component for finest/highest resolution.
            D_xy_fine (jax.Array, optional): `xy` (i.e., off-diagonal) diffusion tensor component for finest/highest resolution. Defaults to `None`.

        Returns:
            List of length `self.levels` for each component.
        """
        # relies on zeroing since not all values get set here
        D_xx = [ jnp.zeros(self.arr_shape[level]) for level in range(self.levels) ]
        D_yy = [ jnp.zeros(self.arr_shape[level]) for level in range(self.levels) ]
        D_xx[0] = D_xx_fine
        D_yy[0] = D_yy_fine
        if self.off_diag_diffusion:
            D_xy = [ jnp.zeros(self.arr_shape[level]) for level in range(self.levels) ]
            D_xy[0] = D_xy_fine
        else:
            D_xy = None

        # set the values for the lower resolution directly from the higher one
        for level in range(1, self.levels):
            D_xx[level] = D_xx[level].at[1:-1, 1:-1].set(D_xx[level - 1][1::2, 1::2])
            D_yy[level] = D_yy[level].at[1:-1, 1:-1].set(D_yy[level - 1][1::2, 1::2])
            if self.off_diag_diffusion:
                D_xy[level] = D_xy[level].at[1:-1, 1:-1].set(D_xy[level - 1][1::2, 1::2])

        # use interpolated values
        for level in range(self.levels):
            D_xx[level] = D_xx[level].at[1:-1, :-1].set(0.5 * (D_xx[level][1:-1, 1:] + D_xx[level][1:-1, :-1]))
            D_yy[level] = D_yy[level].at[:-1, 1:-1].set(0.5 * (D_yy[level][1:, 1:-1] + D_yy[level][:-1, 1:-1]))

        return D_xx, D_yy, D_xy

    def compute_lambda(self, lambda_fine):
        """
        Compute `lambda` (spatially varying decay rate) for every level of the multigrid structure.

        Args:
            lambda_fine (jax.Array): `lambda` for finest/highest resolution.
10.2
        Returns:
            List of length `self.levels`.
        """
        # relies on zeroing since not all values get set here
        lambda_level = [ jnp.zeros(self.arr_shape[level]) for level in range(self.levels) ]
        lambda_level[0] = lambda_fine

        # set the values for the lower resolution directly from the higher one:
        for level in range(1, self.levels):
            lambda_level[level] = lambda_level[level].at[1:-1, 1:-1].set(lambda_level[level - 1][1::2, 1::2])

        return lambda_level

    def compute_discretization(self, lambda_fine, D_xx, D_yy, D_xy=None):
        """
        Compute discretization arrays for every level of the multigrid structure and store them as class members (lists of length `self.levels`).

        Args:
            lambda_fine (float or jax.Array): Either constant or `lambda` for finest/highest resolution.
            D_xx_fine (float or jax.Array): Either `xx` diffusion constant (scalar) or `xx` diffusion tensor component for finest/highest resolution (if spatially varying diffusion is desired).
            D_yy_fine (float or jax.Array): Either `yy` diffusion constant (scalar) or `yy` diffusion tensor component for finest/highest resolution (if spatially varying diffusion is desired).
            D_xy_fine (jax.Array, optional): `xy` diffusion tensor component for finest/highest resolution (if spatially varying diffusion with off-diagonal component is desired). Defaults to `None`.
        """
        sq_idel_x = self.idel_x ** 2
        sq_idel_y = self.idel_y ** 2

        lambda_level = None if self.scalar_lambda else self.compute_lambda(lambda_fine)

        if self.spatial_diffusion:
            D_xx, D_yy, D_xy = self.compute_diffusion_tensor(D_xx, D_yy, D_xy)
            # x-direction
            self.AMat_lc = [ D_xx[level][1:-1, 0:-2] * sq_idel_x[level] for level in range(self.levels) ]
            self.AMat_rc = [ D_xx[level][1:-1, 1:-1] * sq_idel_x[level] for level in range(self.levels) ]
            # y-direction
            self.AMat_cl = [ D_yy[level][0:-2, 1:-1] * sq_idel_y[level] for level in range(self.levels) ]
            self.AMat_cr = [ D_yy[level][1:-1, 1:-1] * sq_idel_y[level] for level in range(self.levels) ]
            # centered
            self.AMat_cc = [ self.AMat_lc[level] + self.AMat_rc[level] + self.AMat_cl[level] + self.AMat_cr[level] + (lambda_fine if self.scalar_lambda else lambda_level[level][1:-1, 1:-1]) for level in range(self.levels) ]

            if self.off_diag_diffusion:
                coeff_xy = self.idel_x * self.idel_y * 0.25
                self.AMat_ll = [ (D_xy[level][1:-1, 0:-2] + D_xy[level][0:-2, 1:-1]) * coeff_xy[level] for level in range(self.levels) ]
                self.AMat_lr = [-(D_xy[level][1:-1, 0:-2] + D_xy[level][2:, 1:-1]) * coeff_xy[level] for level in range(self.levels) ]
                self.AMat_rl = [-(D_xy[level][1:-1, 2:] + D_xy[level][0:-2, 1:-1]) * coeff_xy[level] for level in range(self.levels) ]
                self.AMat_rr = [ (D_xy[level][1:-1, 2:] + D_xy[level][2:, 1:-1]) * coeff_xy[level] for level in range(self.levels) ]
        else:
            self.AMat_x = D_xx * sq_idel_x
            self.AMat_y = D_yy * sq_idel_y
            add = 2 * (self.AMat_x + self.AMat_y)
            self.AMat_cc = add + lambda_fine if self.scalar_lambda else [ add[level] + lambda_level[level][1:-1, 1:-1] for level in range(self.levels) ]

    def solve(self, phi, rhs, lam, D_xx, D_yy, D_xy=None):
        """
        Multigrid solver for 2D steady-state diffusion–absorption problem.

        Iterate (i.e., call `multigrid_cycle`) until `distance(phi_old, phi) <= self.epsilon`.

        Args:
            phi (jax.Array): Initial value for the unknown `phi` (usually filled with zeros).
            rhs (jax.Array): Right-hand side of the equation.
            lam (float or jax.Array): `lambda` (either constant or spatially varying decay rate).
            D_xx_fine (float or jax.Array): Either `xx` diffusion constant (scalar) or `xx` diffusion tensor component for finest/highest resolution (if spatially varying diffusion is desired).
            D_yy_fine (float or jax.Array): Either `yy` diffusion constant (scalar) or `yy` diffusion tensor component for finest/highest resolution (if spatially varying diffusion is desired).
            D_xy_fine (jax.Array, optional): `xy` diffusion tensor component for finest/highest resolution (if spatially varying diffusion with off-diagonal component is desired). Defaults to `None`.

        Returns:
            phi (jax.Array): Computed solution for `phi`.
            count (int): Number of iterations performed.

        Raises:
            ValueError: If argument shapes or types are incompatible.
        """
        self.scalar_lambda = jnp.isscalar(lam)
        self.scalar_diffusion = jnp.isscalar(D_xx) and jnp.isscalar(D_yy)
        self.spatial_diffusion = not self.scalar_diffusion
        self.off_diag_diffusion = self.spatial_diffusion and D_xy is not None

        if not phi.dtype == rhs.dtype == lam.dtype == D_xx.dtype == D_yy.dtype == jnp.float64 == (jnp.float64 if D_xy is None else D_xy.dtype):
            raise ValueError("Incorrect configuration: dtype must be 'jnp.float64' for all arguments.")

        if not D_xx.shape == D_yy.shape or (self.scalar_lambda and not phi.shape == rhs.shape) or (not self.scalar_lambda and not phi.shape == rhs.shape == lam.shape) or (self.spatial_diffusion and not phi.shape == rhs.shape == D_xx.shape == D_yy.shape) or (self.off_diag_diffusion and not phi.shape == rhs.shape == D_xx.shape == D_yy.shape == D_xy.shape):
            raise ValueError(f"Shape mismatch: phi={phi.shape}, rhs={rhs.shape}, lam={lam.shape}, D_xx={D_xx.shape}, D_yy={D_yy.shape}, D_xy={'None' if D_xy is None else D_xy.shape}.")

        self.compute_discretization(lam, D_xx, D_yy, D_xy)

        if self.skip_pre:
            if not self.fixed:
                distance = partial(self.distance, phi)
            phi = self.multigrid_cycle(do_BCs(phi), rhs, self.fcycle)

        if self.fixed:
            phi = jax.lax.fori_loop(1 if self.skip_pre else 0, self.max_iters_outer, lambda i, phi: self.multigrid_cycle(do_BCs(phi), rhs, fcycle=self.fcycle, skip_pre=self.skip_pre), phi)
            count = self.max_iters_outer
        else:
            def cond(arg):
                phi, count, dist = arg
                return jnp.logical_and(dist > self.epsilon, count < self.max_iters_outer)

            def body(arg):
                phi, count, dist = arg
                distance = partial(self.distance, phi)
                phi = self.multigrid_cycle(do_BCs(phi), rhs, fcycle=self.fcycle, skip_pre=self.skip_pre)
                dist = distance(phi)
                #jax.debug.print("count: {}, dist = {}", count, dist)
                return (phi, count + 1, dist)

            phi, count, dist = jax.lax.while_loop(cond, body, (phi, 1, distance(phi)) if self.skip_pre else (phi, 0, jnp.inf))

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

        Compute residual (2D array) by testing the difference of LHS and RHS of the equation.

        Args:
            phi (jax.Array): Current value for the unknown `phi`.
            rhs (jax.Array): Right-hand side of the equation.

        Returns:
            jax.Array: residual array
        """
        if self.spatial_diffusion:
            residual = self.AMat_cc[0][1:-1, 1:-1] * phi[2:-2, 2:-2] - self.AMat_lc[0][1:-1, 1:-1] * phi[2:-2, 1:-3] - self.AMat_rc[0][1:-1, 1:-1] * phi[2:-2, 3:-1] - self.AMat_cl[0][1:-1, 1:-1] * phi[1:-3, 2:-2] - self.AMat_cr[0][1:-1, 1:-1] * phi[3:-1, 2:-2] + rhs[2:-2, 2:-2]
            if self.off_diag_diffusion:
                residual -= self.AMat_ll[0][1:-1, 1:-1] * phi[1:-3, 1:-3] + self.AMat_rl[0][1:-1, 1:-1] * phi[1:-3, 3:-1] + self.AMat_lr[0][1:-1, 1:-1] * phi[3:-1, 1:-3] + self.AMat_rr[0][1:-1, 1:-1] * phi[3:-1, 3:-1]
        else:
            residual = (self.AMat_cc[0] if self.scalar_lambda else self.AMat_cc[0][1:-1, 1:-1]) * phi[2:-2, 2:-2] - self.AMat_x[0] * phi[2:-2, 1:-3] - self.AMat_x[0] * phi[2:-2, 3:-1] - self.AMat_y[0] * phi[1:-3, 2:-2] - self.AMat_y[0] * phi[3:-1, 2:-2] + rhs[2:-2, 2:-2]
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

    def multigrid_cycle(self, phi, rhs, fcycle=False, level=0, skip_pre=False):
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
            skip_pre (bool, optional): Skip pre-smoothing for this level (not applied recursively). Defaults to `False`.

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
            if not skip_pre:
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
            off_diag_term = self.AMat_ll[level] * phi[:-2, :-2] + self.AMat_rl[level] * phi[:-2, 2:] + self.AMat_lr[level] * phi[2:, :-2] + self.AMat_rr[level] * phi[2:, 2:] if self.off_diag_diffusion else 0
            del_fine = del_fine.at[1:-1, 1:-1].set(self.AMat_cc[level] * phi[1:-1, 1:-1] - self.AMat_lc[level] * phi[1:-1, :-2] - self.AMat_rc[level] * phi[1:-1, 2:] - self.AMat_cl[level] * phi[:-2, 1:-1] - self.AMat_cr[level] * phi[2:, 1:-1] + rhs[1:-1, 1:-1] - off_diag_term)

        else:
            del_fine = del_fine.at[1:-1, 1:-1].set(self.AMat_cc[level] * phi[1:-1, 1:-1] - self.AMat_x[level] * phi[1:-1, :-2] - self.AMat_x[level] * phi[1:-1, 2:] - self.AMat_y[level] * phi[:-2, 1:-1] - self.AMat_y[level] * phi[2:, 1:-1] + rhs[1:-1, 1:-1])

        del_fine = do_BCs(del_fine)
        # full-weighting restriction, exploiting that black points have a defect of zero after RBGS pre-smoothing
        rhs_c = rhs_c.at[1:-1, 1:-1].set(0.0625 * (del_fine[2::2, 2::2] + del_fine[2::2, :-2:2] + del_fine[:-2:2, 2::2] + del_fine[:-2:2, :-2:2]) + 0.25 * del_fine[1:-1:2, 1:-1:2])
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

        corr = corr.at[1:-1:2, 1:-1:2].set(phi_c[1:-1, 1:-1])
        corr = corr.at[1:-1:2, 2:-1:2].set(0.5 * (corr[1:-1:2, 3::2] + corr[1:-1:2, 1:-2:2]))
        corr = corr.at[2:-1:2, 1:-1:2].set(0.5 * (corr[3::2, 1:-1:2] + corr[1:-2:2, 1:-1:2]))
        if self.nu2 <= 0 or self.off_diag_diffusion: # interpolation of diagonal values can be skipped if nu2 > 0 since RBGS will override those values anyway in its red sweep; however, red–black partitioning does not suffice (i.e., does not properly decouple the color sweeps) for the 9-point stencil (i.e., all test cases that include D_xy) — a cell update requires the diagonal neighbors as well, turning the smoother into a combination of lexicographic GS and RBGS
            corr = corr.at[2:-1:2, 2:-1:2].set(0.25 * (corr[3::2, 3::2] + corr[1:-2:2, 3::2] + corr[3::2, 1:-2:2] + corr[1:-2:2, 1:-2:2]))

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

        for (lx, ly) in [ (0,0), (1,1), (0,1), (1,0) ]:
            if self.spatial_diffusion:
                off_diag_term = self.AMat_ll[level][ly::2, lx::2] * phi[ly:-2:2, lx:-2:2] + self.AMat_rl[level][ly::2, lx::2] * phi[ly:-2:2, lx+2::2] + self.AMat_lr[level][ly::2, lx::2] * phi[ly+2::2, lx:-2:2] + self.AMat_rr[level][ly::2, lx::2] * phi[ly+2::2, lx+2::2] if self.off_diag_diffusion else 0
                phi = phi.at[ly+1:-1:2, lx+1:-1:2].set((self.AMat_cr[level][ly::2, lx::2] * phi[ly+2::2, lx+1:-1:2] + self.AMat_cl[level][ly::2, lx::2] * phi[ly:-2:2, lx+1:-1:2] + self.AMat_rc[level][ly::2, lx::2] * phi[ly+1:-1:2, lx+2::2] + self.AMat_lc[level][ly::2, lx::2] * phi[ly+1:-1:2, lx:-2:2] - rhs[ly+1:-1:2, lx+1:-1:2] + off_diag_term) / self.AMat_cc[level][ly::2, lx::2])
            else:
                phi = phi.at[ly+1:-1:2, lx+1:-1:2].set((self.AMat_y[level] * phi[ly+2::2, lx+1:-1:2] + self.AMat_y[level] * phi[ly:-2:2, lx+1:-1:2] + self.AMat_x[level] * phi[ly+1:-1:2, lx+2::2] + self.AMat_x[level] * phi[ly+1:-1:2, lx:-2:2] - rhs[ly+1:-1:2, lx+1:-1:2]) / (self.AMat_cc[level] if self.scalar_lambda else self.AMat_cc[level][ly::2, lx::2]))
            phi = do_BCs(phi)

        return phi

