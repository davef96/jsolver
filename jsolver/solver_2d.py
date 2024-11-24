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

@partial(jax.jit, static_argnames=('grid_x', 'grid_y', 'epsilon', 'gamma', 'fcycle', 'nu1', 'nu2', 'max_iters'))
def solve_2d(grid_x, grid_y, *, epsilon=1e-12, gamma=1, fcycle=False, nu1=1, nu2=1, max_iters=np.iinfo(np.int32).max, phi, rhs, lam, D_xx, D_yy, D_xy=None):
    """
    Wrapper function which is functionally pure and thus compatible with `jax` transformations.

    Initializes `Solver_2D` object and calls `solve` method.

    Arguments after grid objects are set to keyword-only (`*`) for full support of all optional arguments.

    Note that `grid_x` and `grid_y` must be static arguments for `jax.jit` since array shapes and number of multigrid levels are computed from them.

    All arrays must be two-dimensional and of the same shape.

    Args:
        grid_x (grid_1D): Linear grid for x-dimension (column-dimension).
        grid_y (grid_1D): Linear grid for y-dimension (row-dimension).
        epsilon (float, optional): Solver termination threshold. Defaults to `1e-12`.
        gamma (int, optional): Number of recursive calls between `defect` and `prolongate` in `multigrid_routine`. Defaults to `1`.
        fcycle (bool, optional): Use F-cycle in `multigrid_routine`. Can be combined with `gamma`. Defaults to `False`.
        nu1 (int, optional): Number of smoothing iterations before `defect` operation. Defaults to `1`.
        nu2 (int, optional): Number of smoothing iterations after `prolongate`. Defaults to `1`.
        max_iters (int, optional): The maximum number of solver iterations. Defaults to `int32.max`.
        phi (jax.Array): Initial value for the unknown `phi` (usually filled with zeros).
        rhs (jax.Array): Right-hand side of the equation.
        lam (float or jax.Array): `lambda` (either constant or spatially-varying decay rate).
        D_xx_fine (float or jax.Array): Either `xx` diffusion constant (scalar) or `xx` diffusion tensor component for finest/highest resolution (if spatially-varying diffusion is desired).
        D_yy_fine (float or jax.Array): Either `yy` diffusion constant (scalar) or `yy` diffusion tensor component for finest/highest resolution (if spatially-varying diffusion is desired).
        D_xy_fine (jax.Array, optional): `xy` diffusion tensor component for finest/highest resolution (if spatially-varying diffusion with off-diagonal component is desired). Defaults to `None`.

    Returns:
        phi (jax.Array): Computed solution for `phi`.
        count (int): Number of iterations performed.

    Raises:
        ValueError: If either grid is not linear, or the grid resolution is not a power of 2 or not efficient for multigrid configuration.
    """
    solver = Solver_2D(grid_x=grid_x, grid_y=grid_y, epsilon=epsilon, gamma=gamma, fcycle=fcycle, nu1=nu1, nu2=nu2, max_iters=max_iters)
    return solver.solve(phi=phi, rhs=rhs, lam=lam, D_xx=D_xx, D_yy=D_yy, D_xy=D_xy)

class Solver_2D:
    """
    A class containing methods for the implementation of a multigrid solver for a 2D steady-state diffusion problem.

    Members are used for convenience, as explicit parameter passing would be quite the hassle for some functions.
    Note that all methods are impure due to the implicit `self` parameter and thus not (directly) compatible with JAX's transformations.
    """

    def __init__(self, grid_x, grid_y, epsilon=1e-12, gamma=1, fcycle=False, nu1=1, nu2=1, max_iters=np.iinfo(np.int32).max):
        """
        Initialize members for solver configuration and compute important constants.

        Args:
            grid_x (grid_1D): Linear grid for x-dimension (column-dimension).
            grid_y (grid_1D): Linear grid for y-dimension (row-dimension).
            epsilon (float, optional): Solver termination threshold. Defaults to `1e-12`.
            gamma (int, optional): Number of recursive calls between `defect` and `prolongate` in `multigrid_routine`. Defaults to `1`.
            fcycle (bool, optional): Use F-cycle in `multigrid_routine`. Can be combined with `gamma`. Defaults to `False`.
            nu1 (int, optional): Number of smoothing iterations before `defect` operation. Defaults to `1`.
            nu2 (int, optional): Number of smoothing iterations after `prolongate`. Defaults to `1`.
            max_iters (int, optional): The maximum number of solver iterations. Defaults to `int32.max`.

        Raises:
            ValueError: If either grid is not linear, or the grid resolution is not a power of 2 or not efficient for multigrid configuration.
        """
        self.max_iters = max_iters
        self.epsilon = epsilon
        self.gamma = gamma
        self.fcycle = fcycle
        self.nu1 = nu1
        self.nu2 = nu2
        self.compute_shapes(grid_x.mx, grid_y.mx)
        self.compute_idel(grid_x, grid_y)

    def compute_shapes(self, mx, my):
        """
        Compute number of levels and shapes for each level statically (i.e. use `numpy` instead of `jax.numpy`).

        All array shapes must be compile-time constants w.r.t. `jax.jit`!

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
        Compute diffusion tensor components for every level of the multigrid structure (if spatially-varying diffusion is desired).

        Interpolated values between the gridpoints are used for the diffusion.

        Args:
            D_xx_fine (jax.Array): `xx` diffusion tensor component for finest/highest resolution.
            D_yy_fine (jax.Array): `yy` diffusion tensor component for finest/highest resolution.
            D_xy_fine (jax.Array, optional): `xy` (i.e. off-diagonal) diffusion tensor component for finest/highest resolution. Defaults to `None`.

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
        Compute `lambda` (spatially-varying decay rate) for every level of the multigrid structure.

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
            lambda_level[level] = lambda_level[level].at[1:-1, 1:-1].set(lambda_level[level - 1][1::2, 1::2])

        return lambda_level

    def compute_discretization(self, lambda_fine, D_xx, D_yy, D_xy=None):
        """
        Compute discretization arrays for every level of the multigrid structure and store them as class members (lists of length `self.levels`).

        Args:
            lambda_fine (float or jax.Array): Either constant or `lambda` for finest/highest resolution.
            D_xx_fine (float or jax.Array): Either `xx` diffusion constant (scalar) or `xx` diffusion tensor component for finest/highest resolution (if spatially-varying diffusion is desired).
            D_yy_fine (float or jax.Array): Either `yy` diffusion constant (scalar) or `yy` diffusion tensor component for finest/highest resolution (if spatially-varying diffusion is desired).
            D_xy_fine (jax.Array, optional): `xy` diffusion tensor component for finest/highest resolution (if spatially-varying diffusion with off-diagonal component is desired). Defaults to `None`.
        """
        sq_idel_x = self.idel_x ** 2
        sq_idel_y = self.idel_y ** 2

        lambda_level = None if self.scalar_lambda else self.compute_lambda(lambda_fine)

        if self.spatial_diffusion:
            D_xx, D_yy, D_xy = self.compute_diffusion_tensor(D_xx, D_yy, D_xy)
            # x-direction
            self.AMat_lc = [ jnp.roll(D_xx[level], -1, axis=0) * sq_idel_x[level] for level in range(self.levels) ]
            self.AMat_rc = [ jnp.roll(D_xx[level], (-1,-1), axis=(0,1)) * sq_idel_x[level] for level in range(self.levels) ]
            # y-direction
            self.AMat_cl = [ jnp.roll(D_yy[level], -1, axis=1) * sq_idel_y[level] for level in range(self.levels) ]
            self.AMat_cr = [ jnp.roll(D_yy[level], (-1,-1), axis=(0,1)) * sq_idel_y[level] for level in range(self.levels) ]
            # centered
            self.AMat_cc = [ self.AMat_lc[level] + self.AMat_rc[level] + self.AMat_cl[level] + self.AMat_cr[level] + (lambda_fine if self.scalar_lambda else jnp.roll(lambda_level[level], (-1,-1), axis=(0,1))) for level in range(self.levels) ]

            if self.off_diag_diffusion:
                coeff_xy = self.idel_x * self.idel_y * 0.25
                self.AMat_ll = [ (jnp.roll(D_xy[level], -1, axis=0) + jnp.roll(D_xy[level], -1, axis=1)) * coeff_xy[level] for level in range(self.levels) ]
                self.AMat_lr = [-(jnp.roll(D_xy[level], -1, axis=0) + jnp.roll(D_xy[level], (-2, -1), axis=(0,1))) * coeff_xy[level] for level in range(self.levels) ]
                self.AMat_rl = [-(jnp.roll(D_xy[level], (-1,-2), axis=(0,1)) + jnp.roll(D_xy[level], -1, axis=1)) * coeff_xy[level] for level in range(self.levels) ]
                self.AMat_rr = [ (jnp.roll(D_xy[level], (-1,-2), axis=(0,1)) + jnp.roll(D_xy[level], (-2,-1), axis=(0,1))) * coeff_xy[level] for level in range(self.levels) ]
        else:
            self.AMat_x = D_xx * sq_idel_x
            self.AMat_y = D_yy * sq_idel_y
            add = 2 * (self.AMat_x + self.AMat_y)
            self.AMat_cc = add + lambda_fine if self.scalar_lambda else [ add[level] + jnp.roll(lambda_level[level], (-1,-1), axis=(0,1)) for level in range(self.levels) ]

    def solve(self, phi, rhs, lam, D_xx, D_yy, D_xy=None):
        """
        Multigrid solver for 2D steady-state diffusion problem.

        Iterate until `distance(phi_old, phi) <= self.epsilon`.

        Args:
            phi (jax.Array): Initial value for the unknown `phi` (usually filled with zeros).
            rhs (jax.Array): Right-hand side of the equation.
            lam (float or jax.Array): `lambda` (either constant or spatially-varying decay rate).
            D_xx_fine (float or jax.Array): Either `xx` diffusion constant (scalar) or `xx` diffusion tensor component for finest/highest resolution (if spatially-varying diffusion is desired).
            D_yy_fine (float or jax.Array): Either `yy` diffusion constant (scalar) or `yy` diffusion tensor component for finest/highest resolution (if spatially-varying diffusion is desired).
            D_xy_fine (jax.Array, optional): `xy` diffusion tensor component for finest/highest resolution (if spatially-varying diffusion with off-diagonal component is desired). Defaults to `None`.

        Returns:
            phi (jax.Array): Computed solution for `phi`.
            count (int): Number of iterations performed.
        """
        self.spatial_diffusion = phi.shape == D_xx.shape == D_yy.shape
        self.off_diag_diffusion = D_xy is not None and phi.shape == D_xy.shape
        self.scalar_lambda = not lam.shape == phi.shape

        self.compute_discretization(lam, D_xx, D_yy, D_xy)

        def cond(arg):
            phi, count, dist = arg
            return jnp.logical_and(dist > self.epsilon, count < self.max_iters)

        def body(arg):
            phi, count, dist = arg
            distance = partial(self.distance, phi)
            phi = self.multigrid_routine(do_BCs(phi), rhs, self.fcycle)
            dist = distance(phi)
            #jax.debug.print("count: {}, dist = {}", count, dist)
            return (phi, count + 1, dist)

        phi, count, dist = jax.lax.while_loop(cond, body, (phi, 0, jnp.inf))

        return phi, count

    def distance(self, phi_old, phi):
        """
        Compute distance to previous step.

        A step/iteration corresponds to a call to `multigrid_routine` within the while loop in `solve`.

        Args:
            phi_old (jax.Array): Value before current call to `multigrid_routine`.
            phi (jax.Array): Return value of `multigrid_routine`.

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
            residual = self.AMat_cc[0][1:-3, 1:-3] * phi[2:-2, 2:-2] - self.AMat_lc[0][1:-3, 1:-3] * phi[2:-2, 1:-3] - self.AMat_rc[0][1:-3, 1:-3] * phi[2:-2, 3:-1] - self.AMat_cl[0][1:-3, 1:-3] * phi[1:-3, 2:-2] - self.AMat_cr[0][1:-3, 1:-3] * phi[3:-1, 2:-2] + rhs[2:-2, 2:-2]
            if self.off_diag_diffusion:
                residual -= self.AMat_ll[0][1:-3, 1:-3] * phi[1:-3, 1:-3] + self.AMat_rl[0][1:-3, 1:-3] * phi[1:-3, 3:-1] + self.AMat_lr[0][1:-3, 1:-3] * phi[3:-1, 1:-3] + self.AMat_rr[0][1:-3, 1:-3] * phi[3:-1, 3:-1]
        else:
            residual = (self.AMat_cc[0] if self.scalar_lambda else self.AMat_cc[0][1:-3, 1:-3]) * phi[2:-2, 2:-2] - self.AMat_x[0] * phi[2:-2, 1:-3] - self.AMat_x[0] * phi[2:-2, 3:-1] - self.AMat_y[0] * phi[1:-3, 2:-2] - self.AMat_y[0] * phi[3:-1, 2:-2] + rhs[2:-2, 2:-2]
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

    def multigrid_routine(self, phi, rhs, fcycle=False, level=0):
        """
        Recursive multigrid routine.

        Behavior is controlled by members:
            self.nu1 (int): Smoothing iterations before `defect` / restriction operation.
            self.nu2 (int): Smoothing iterations after `prolongate`.
            self.gamma (int): Number of recursive calls between `defect` and `prolongate` (1: V-cycle, 2: W-cycle, ...).

        Args:
            phi (jax.Array): Current value for the unknown `phi`.
            rhs (jax.Array): Right-hand side of the equation.
            level (int, optional): Current multigrid level (`0`: finest/highest resolution, ..., `self.levels-1`: coarsest/lowest resolution). Defaults to `0`.
            fcycle (bool, optional): Use F-cycle (i.e. do two recursive calls where the first one uses the F-gamma-cycle recursively and the second one performs a regular gamma-cycle). Defaults to `False`.

        Returns:
            phi (jax.Array): New value for the unknown `phi`.
        """
        if level == self.levels - 1:
            def cond(arg):
                phi, dist = arg
                return dist > self.epsilon

            def body(arg):
                phi, dist = arg
                distance = partial(self.distance, phi)
                phi = self.RedBlackGaussSeidel(phi, rhs, level)
                return (phi, distance(phi))

            phi, _ = jax.lax.while_loop(cond, body, (phi, jnp.inf))
        else:
            phi = jax.lax.fori_loop(0, self.nu1, lambda i, phi: self.RedBlackGaussSeidel(phi, rhs, level), phi)
            rhs_c = self.defect(phi, rhs, level)
            if fcycle:
                phi_c = jax.lax.fori_loop(0, self.gamma, lambda i, phi_c: self.multigrid_routine(phi_c, rhs_c, True, level + 1), jnp.full(self.arr_shape[level + 1], 0.))
            phi_c = jax.lax.fori_loop(0, self.gamma, lambda i, phi_c: self.multigrid_routine(phi_c, rhs_c, False, level + 1), jnp.full(self.arr_shape[level + 1], 0.))
            phi = jax.lax.fori_loop(0, self.nu2, lambda i, phi: self.RedBlackGaussSeidel(phi, rhs, level), phi + self.prolongate(phi_c, level))

        return phi

    def defect(self, phi, rhs, level):
        """
        Compute defect for current solution and assign to coarser grid.

        Corresponds to residual computation and restriction.

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
            off_diag_term = self.AMat_ll[level][:-2, :-2] * phi[:-2, :-2] + self.AMat_rl[level][:-2, :-2] * phi[:-2, 2:] + self.AMat_lr[level][:-2, :-2] * phi[2:, :-2] + self.AMat_rr[level][:-2, :-2] * phi[2:, 2:] if self.off_diag_diffusion else 0
            del_fine = del_fine.at[1:-1, 1:-1].set(self.AMat_cc[level][:-2, :-2] * phi[1:-1, 1:-1] - self.AMat_lc[level][:-2, :-2] * phi[1:-1, :-2] - self.AMat_rc[level][:-2, :-2] * phi[1:-1, 2:] - self.AMat_cl[level][:-2, :-2] * phi[:-2, 1:-1] - self.AMat_cr[level][:-2, :-2] * phi[2:, 1:-1] + rhs[1:-1, 1:-1] - off_diag_term)

        else:
            del_fine = del_fine.at[1:-1, 1:-1].set((self.AMat_cc[level] if self.scalar_lambda else self.AMat_cc[level][:-2, :-2]) * phi[1:-1, 1:-1] - self.AMat_x[level] * phi[1:-1, :-2] - self.AMat_x[level] * phi[1:-1, 2:] - self.AMat_y[level] * phi[:-2, 1:-1] - self.AMat_y[level] * phi[2:, 1:-1] + rhs[1:-1, 1:-1])

        del_fine = do_BCs(del_fine)
        rhs_c = rhs_c.at[1:-1, 1:-1].set(0.0625 * (del_fine[2::2, 2::2] + del_fine[2::2, :-2:2] + del_fine[:-2:2, 2::2] + del_fine[:-2:2, :-2:2]) + 0.25 * del_fine[1:-1:2, 1:-1:2])
        rhs_c = do_BCs(rhs_c)

        return rhs_c

    def prolongate(self, phi_c, level):
        """
        Interpolate the correction computed on a coarser grid into a finer grid.

        Args:
            phi_c (jax.Array): Correction computed on a coarser grid (i.e. on `level + 1`)
            level (int): Current multigrid level, corresponding to finer grid.

        Returns:
            corr (jax.Array): Interpolated correction (to be added to the current value of `phi` on the finer grid).
        """
        corr = jnp.empty(self.arr_shape[level])

        corr = corr.at[1:-1:2, 1:-1:2].set(phi_c[1:-1, 1:-1])
        corr = corr.at[1:-1:2, 2:-1:2].set(0.5 * (corr[1:-1:2, 3::2] + corr[1:-1:2, 1:-2:2]))
        corr = corr.at[2:-1:2, 1:-1:2].set(0.5 * (corr[3::2, 1:-1:2] + corr[1:-2:2, 1:-1:2]))
        corr = corr.at[2:-1:2, 2:-1:2].set(0.25 * (corr[3::2, 3::2] + corr[1:-2:2, 3::2] + corr[3::2, 1:-2:2] + corr[1:-2:2, 1:-2:2]))

        return corr

    def RedBlackGaussSeidel(self, phi, rhs, level):
        """
        Red-Black Gauss-Seidel routine for smoothing, i.e. reducing high-frequency errors.

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
                off_diag_term = self.AMat_ll[level][ly:-2:2, lx:-2:2] * phi[ly:-2:2, lx:-2:2] + self.AMat_rl[level][ly:-2:2, lx:-2:2] * phi[ly:-2:2, lx+2::2] + self.AMat_lr[level][ly:-2:2, lx:-2:2] * phi[ly+2::2, lx:-2:2] + self.AMat_rr[level][ly:-2:2, lx:-2:2] * phi[ly+2::2, lx+2::2] if self.off_diag_diffusion else 0
                phi = phi.at[ly+1:-1:2, lx+1:-1:2].set((self.AMat_cr[level][ly:-2:2, lx:-2:2] * phi[ly+2::2, lx+1:-1:2] + self.AMat_cl[level][ly:-2:2, lx:-2:2] * phi[ly:-2:2, lx+1:-1:2] + self.AMat_rc[level][ly:-2:2, lx:-2:2] * phi[ly+1:-1:2, lx+2::2] + self.AMat_lc[level][ly:-2:2, lx:-2:2] * phi[ly+1:-1:2, lx:-2:2] - rhs[ly+1:-1:2, lx+1:-1:2] + off_diag_term) / self.AMat_cc[level][ly:-2:2, lx:-2:2])
            else:
                phi = phi.at[ly+1:-1:2, lx+1:-1:2].set((self.AMat_y[level] * phi[ly+2::2, lx+1:-1:2] + self.AMat_y[level] * phi[ly:-2:2, lx+1:-1:2] + self.AMat_x[level] * phi[ly+1:-1:2, lx+2::2] + self.AMat_x[level] * phi[ly+1:-1:2, lx:-2:2] - rhs[ly+1:-1:2, lx+1:-1:2]) / (self.AMat_cc[level] if self.scalar_lambda else self.AMat_cc[level][ly:-2:2, lx:-2:2]))
            phi = do_BCs(phi)

        return phi

