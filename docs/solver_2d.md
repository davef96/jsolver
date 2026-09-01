<a id="solver_2d"></a>

# solver\_2d

<a id="solver_2d.solve_2d_simple"></a>

#### solve\_2d\_simple

```python
def solve_2d_simple(grid_x, grid_y, phi, rhs, lam, D_xx, D_yy, D_xy=None)
```

Simplified, functionally pure wrapper function with positional arguments.

See `solve_2d`.

<a id="solver_2d.solve_2d_fixed_simple"></a>

#### solve\_2d\_fixed\_simple

```python
def solve_2d_fixed_simple(grid_x,
                          grid_y,
                          iters,
                          phi,
                          rhs,
                          lam,
                          D_xx,
                          D_yy,
                          D_xy=None)
```

Simplified, functionally pure wrapper function with positional arguments and fixed number of solver iterations (smoothing iterations for coarsest level are set to `1`).

The argument `iters` controls the number of multigrid cycles.

See `solve_2d`.

<a id="solver_2d.solve_2d_fixed"></a>

#### solve\_2d\_fixed

```python
def solve_2d_fixed(grid_x,
                   grid_y,
                   iters_outer,
                   iters_inner,
                   phi,
                   rhs,
                   lam,
                   D_xx,
                   D_yy,
                   D_xy=None)
```

Simplified, functionally pure wrapper function with positional arguments and fixed number of solver iterations.

The argument `iters_outer` controls the number of multigrid cycles, `iters_inner` controls the number of smoothing iterations for the coarsest grid level.

See `solve_2d`.

<a id="solver_2d.solve_2d_skip_pre_simple"></a>

#### solve\_2d\_skip\_pre\_simple

```python
def solve_2d_skip_pre_simple(grid_x,
                             grid_y,
                             phi,
                             rhs,
                             lam,
                             D_xx,
                             D_yy,
                             D_xy=None)
```

Simplified, functionally pure wrapper function with positional arguments.

Pre-Smoothing is skipped for repeated V-cycles such that no consecutive smoothing operations happen for the finest/highest resolution. Otherwise, post-smoothing from previous cycle would be immediately followed by pre-smoothing.

See `solve_2d`.

<a id="solver_2d.solve_2d_fixed_skip_pre_simple"></a>

#### solve\_2d\_fixed\_skip\_pre\_simple

```python
def solve_2d_fixed_skip_pre_simple(grid_x,
                                   grid_y,
                                   iters,
                                   phi,
                                   rhs,
                                   lam,
                                   D_xx,
                                   D_yy,
                                   D_xy=None)
```

Simplified, functionally pure wrapper function with positional arguments and fixed number of solver iterations (smoothing iterations for coarsest level are set to `1`).

The argument `iters` controls the number of multigrid cycles.

Pre-Smoothing is skipped for repeated V-cycles such that no consecutive smoothing operations happen for the finest/highest resolution. Otherwise, post-smoothing from previous cycle would be immediately followed by pre-smoothing.

See `solve_2d`.

<a id="solver_2d.solve_2d_fixed_skip_pre"></a>

#### solve\_2d\_fixed\_skip\_pre

```python
def solve_2d_fixed_skip_pre(grid_x,
                            grid_y,
                            iters_outer,
                            iters_inner,
                            phi,
                            rhs,
                            lam,
                            D_xx,
                            D_yy,
                            D_xy=None)
```

Simplified, functionally pure wrapper function with positional arguments and fixed number of solver iterations.

The argument `iters_outer` controls the number of multigrid cycles, `iters_inner` controls the number of smoothing iterations for the coarsest grid level.

Pre-Smoothing is skipped for repeated V-cycles such that no consecutive smoothing operations happen for the finest/highest resolution. Otherwise, post-smoothing from previous cycle would be immediately followed by pre-smoothing.

See `solve_2d`.

<a id="solver_2d.solve_2d_fcycle_simple"></a>

#### solve\_2d\_fcycle\_simple

```python
def solve_2d_fcycle_simple(grid_x,
                           grid_y,
                           phi,
                           rhs,
                           lam,
                           D_xx,
                           D_yy,
                           D_xy=None)
```

Simplified, functionally pure wrapper function with positional arguments and F-cycle multigrid.

See `solve_2d`.

<a id="solver_2d.solve_2d_fixed_fcycle_simple"></a>

#### solve\_2d\_fixed\_fcycle\_simple

```python
def solve_2d_fixed_fcycle_simple(grid_x,
                                 grid_y,
                                 iters,
                                 phi,
                                 rhs,
                                 lam,
                                 D_xx,
                                 D_yy,
                                 D_xy=None)
```

Simplified, functionally pure wrapper function with positional arguments, fixed number of solver iterations (smoothing iterations for coarsest level are set to `1`) and F-cycle multigrid.

The argument `iters` controls the number of multigrid cycles.

See `solve_2d`.

<a id="solver_2d.solve_2d_fixed_fcycle"></a>

#### solve\_2d\_fixed\_fcycle

```python
def solve_2d_fixed_fcycle(grid_x,
                          grid_y,
                          iters_outer,
                          iters_inner,
                          phi,
                          rhs,
                          lam,
                          D_xx,
                          D_yy,
                          D_xy=None)
```

Simplified, functionally pure wrapper function with positional arguments, fixed number of solver iterations and F-cycle multigrid.

The argument `iters_outer` controls the number of multigrid cycles, `iters_inner` controls the number of smoothing iterations for the coarsest grid level.

See `solve_2d`.

<a id="solver_2d.solve_2d_fcycle_skip_pre_simple"></a>

#### solve\_2d\_fcycle\_skip\_pre\_simple

```python
def solve_2d_fcycle_skip_pre_simple(grid_x,
                                    grid_y,
                                    phi,
                                    rhs,
                                    lam,
                                    D_xx,
                                    D_yy,
                                    D_xy=None)
```

Simplified, functionally pure wrapper function with positional arguments and F-cycle multigrid.

Pre-Smoothing is skipped for repeated F-cycles such that no consecutive smoothing operations happen for the finest/highest resolution. Otherwise, post-smoothing from previous cycle would be immediately followed by pre-smoothing.

See `solve_2d`.

<a id="solver_2d.solve_2d_fixed_fcycle_skip_pre_simple"></a>

#### solve\_2d\_fixed\_fcycle\_skip\_pre\_simple

```python
def solve_2d_fixed_fcycle_skip_pre_simple(grid_x,
                                          grid_y,
                                          iters,
                                          phi,
                                          rhs,
                                          lam,
                                          D_xx,
                                          D_yy,
                                          D_xy=None)
```

Simplified, functionally pure wrapper function with positional arguments, fixed number of solver iterations and F-cycle multigrid (smoothing iterations for coarsest level are set to `1`).

The argument `iters` controls the number of multigrid cycles.

Pre-Smoothing is skipped for repeated F-cycles such that no consecutive smoothing operations happen for the finest/highest resolution. Otherwise, post-smoothing from previous cycle would be immediately followed by pre-smoothing.

See `solve_2d`.

<a id="solver_2d.solve_2d_fixed_fcycle_skip_pre"></a>

#### solve\_2d\_fixed\_fcycle\_skip\_pre

```python
def solve_2d_fixed_fcycle_skip_pre(grid_x,
                                   grid_y,
                                   iters_outer,
                                   iters_inner,
                                   phi,
                                   rhs,
                                   lam,
                                   D_xx,
                                   D_yy,
                                   D_xy=None)
```

Simplified, functionally pure wrapper function with positional arguments, fixed number of solver iterations and F-cycle multigrid.

The argument `iters_outer` controls the number of multigrid cycles, `iters_inner` controls the number of smoothing iterations for the coarsest grid level.

Pre-Smoothing is skipped for repeated F-cycles such that no consecutive smoothing operations happen for the finest/highest resolution. Otherwise, post-smoothing from previous cycle would be immediately followed by pre-smoothing.

See `solve_2d`.

<a id="solver_2d.solve_2d_wcycle_simple"></a>

#### solve\_2d\_wcycle\_simple

```python
def solve_2d_wcycle_simple(grid_x,
                           grid_y,
                           phi,
                           rhs,
                           lam,
                           D_xx,
                           D_yy,
                           D_xy=None)
```

Simplified, functionally pure wrapper function with positional arguments and W-cycle multigrid.

See `solve_2d`.

<a id="solver_2d.solve_2d_fixed_wcycle_simple"></a>

#### solve\_2d\_fixed\_wcycle\_simple

```python
def solve_2d_fixed_wcycle_simple(grid_x,
                                 grid_y,
                                 iters,
                                 phi,
                                 rhs,
                                 lam,
                                 D_xx,
                                 D_yy,
                                 D_xy=None)
```

Simplified, functionally pure wrapper function with positional arguments, fixed number of solver iterations (smoothing iterations for coarsest level are set to `1`) and W-cycle multigrid.

The argument `iters` controls the number of multigrid cycles.

See `solve_2d`.

<a id="solver_2d.solve_2d_fixed_wcycle"></a>

#### solve\_2d\_fixed\_wcycle

```python
def solve_2d_fixed_wcycle(grid_x,
                          grid_y,
                          iters_outer,
                          iters_inner,
                          phi,
                          rhs,
                          lam,
                          D_xx,
                          D_yy,
                          D_xy=None)
```

Simplified, functionally pure wrapper function with positional arguments, fixed number of solver iterations and W-cycle multigrid.

The argument `iters_outer` controls the number of multigrid cycles, `iters_inner` controls the number of smoothing iterations for the coarsest grid level.

See `solve_2d`.

<a id="solver_2d.solve_2d_wcycle_skip_pre_simple"></a>

#### solve\_2d\_wcycle\_skip\_pre\_simple

```python
def solve_2d_wcycle_skip_pre_simple(grid_x,
                                    grid_y,
                                    phi,
                                    rhs,
                                    lam,
                                    D_xx,
                                    D_yy,
                                    D_xy=None)
```

Simplified, functionally pure wrapper function with positional arguments and W-cycle multigrid.

Pre-Smoothing is skipped for repeated W-cycles such that no consecutive smoothing operations happen for the finest/highest resolution. Otherwise, post-smoothing from previous cycle would be immediately followed by pre-smoothing.

See `solve_2d`.

<a id="solver_2d.solve_2d_fixed_wcycle_skip_pre_simple"></a>

#### solve\_2d\_fixed\_wcycle\_skip\_pre\_simple

```python
def solve_2d_fixed_wcycle_skip_pre_simple(grid_x,
                                          grid_y,
                                          iters,
                                          phi,
                                          rhs,
                                          lam,
                                          D_xx,
                                          D_yy,
                                          D_xy=None)
```

Simplified, functionally pure wrapper function with positional arguments, fixed number of solver iterations and W-cycle multigrid (smoothing iterations for coarsest level are set to `1`).

The argument `iters` controls the number of multigrid cycles.

Pre-Smoothing is skipped for repeated W-cycles such that no consecutive smoothing operations happen for the finest/highest resolution. Otherwise, post-smoothing from previous cycle would be immediately followed by pre-smoothing.

See `solve_2d`.

<a id="solver_2d.solve_2d_fixed_wcycle_skip_pre"></a>

#### solve\_2d\_fixed\_wcycle\_skip\_pre

```python
def solve_2d_fixed_wcycle_skip_pre(grid_x,
                                   grid_y,
                                   iters_outer,
                                   iters_inner,
                                   phi,
                                   rhs,
                                   lam,
                                   D_xx,
                                   D_yy,
                                   D_xy=None)
```

Simplified, functionally pure wrapper function with positional arguments, fixed number of solver iterations and W-cycle multigrid.

The argument `iters_outer` controls the number of multigrid cycles, `iters_inner` controls the number of smoothing iterations for the coarsest grid level.

Pre-Smoothing is skipped for repeated W-cycles such that no consecutive smoothing operations happen for the finest/highest resolution. Otherwise, post-smoothing from previous cycle would be immediately followed by pre-smoothing.

See `solve_2d`.

<a id="solver_2d.solve_2d"></a>

#### solve\_2d

```python
@partial(jax.jit,
         static_argnames=('grid_x', 'grid_y', 'fixed', 'epsilon', 'gamma',
                          'fcycle', 'skip_pre', 'nu1', 'nu2',
                          'max_iters_outer', 'max_iters_inner'))
def solve_2d(grid_x,
             grid_y,
             *,
             fixed=False,
             epsilon=1e-12,
             gamma=1,
             fcycle=False,
             skip_pre=False,
             nu1=1,
             nu2=1,
             max_iters_outer=np.iinfo(np.int32).max,
             max_iters_inner=np.iinfo(np.int32).max,
             phi,
             rhs,
             lam,
             D_xx,
             D_yy,
             D_xy=None)
```

Wrapper function which is functionally pure and thus compatible with `jax` transformations.

Initializes `Solver_2D` object and calls `solve` method.

Arguments after grid objects are set to keyword-only (`*`) for full support of all optional arguments.

Note that `grid_x` and `grid_y` must be static arguments for `jax.jit` since array shapes and number of multigrid levels are computed from them.

All arrays must be two-dimensional and of compatible shape: `lam`, `D_xx`, and `D_yy` can be scalars, but `D_xx` and `D_yy` must either both be scalar or both of the same shape as the other arrays; all non-scalars must be of the same shape.

**Arguments**:

- `grid_x` _grid_1D_ - Linear grid for x-dimension (column-dimension).
- `grid_y` _grid_1D_ - Linear grid for y-dimension (row-dimension).
- `fixed` _bool, optional_ - Perform exactly `max_iters_outer` solver iterations and `max_iters_inner` smoothing iterations for the coarsest grid level instead of relying on a dynamic termination threshold. The function is compatible with reverse-mode automatic differentiation only if set to `True`. Defaults to `False`.
- `epsilon` _float, optional_ - Solver termination threshold. Defaults to `1e-12`.
- `gamma` _int, optional_ - Number of recursive calls between `defect` and `prolongate` in `multigrid_cycle`. Defaults to `1` (V-cycle).
- `fcycle` _bool, optional_ - Use F-cycle in `multigrid_cycle`. Can be combined with `gamma` into an "F-gamma-cycle". Defaults to `False`.
- `skip_pre` _bool, optional_ - Skip pre-smoothing for repeated cycles for finest/highest resolution (such that no consecutive smoothing calls happen). Only makes sense if `nu1 > 0` and `nu2 > 0`. Defaults to `False`.
- `nu1` _int, optional_ - Number of pre-smoothing iterations (i.e., before `defect`, which relies on `nu1 > 0`). Defaults to `1`.
- `nu2` _int, optional_ - Number of post-smoothing iterations (i.e., after `prolongate`). Defaults to `1`.
- `max_iters_outer` _int, optional_ - The maximum number of solver iterations (i.e., multigrid cycles). Defaults to `int32.max`.
- `max_iters_inner` _int, optional_ - The maximum number of smoothing iterations for coarsest grid level. Defaults to `int32.max`.
- `phi` _jax.Array_ - Initial value for the unknown `phi` (usually filled with zeros).
- `rhs` _jax.Array_ - Right-hand side of the equation.
- `lam` _float or jax.Array_ - `lambda` (either constant or spatially varying decay rate).
- `D_xx_fine` _float or jax.Array_ - Either `xx` diffusion constant (scalar) or `xx` diffusion tensor component for finest/highest resolution (if spatially varying diffusion is desired).
- `D_yy_fine` _float or jax.Array_ - Either `yy` diffusion constant (scalar) or `yy` diffusion tensor component for finest/highest resolution (if spatially varying diffusion is desired).
- `D_xy_fine` _jax.Array, optional_ - `xy` diffusion tensor component for finest/highest resolution (if spatially varying diffusion with off-diagonal component is desired). Defaults to `None`.
  

**Returns**:

- `phi` _jax.Array_ - Computed solution for `phi`.
- `count` _int_ - Number of iterations performed.
  

**Raises**:

- `ValueError` - If either grid is not linear, or the grid resolution is not a power of 2 or not efficient for multigrid configuration, or array shapes are incompatible.

<a id="solver_2d.Solver_2D"></a>

## Solver\_2D Objects

```python
class Solver_2D()
```

A class containing methods implementing an iterative geometric multigrid solver for the 2D steady-state diffusion–absorption problem.

Members are used for convenience, as explicit parameter passing would be quite the hassle (for some functions).
Note that all methods are impure due to the implicit `self` parameter and thus not (directly) compatible with `jax` transformations.
However, all invariants are established after calling `compute_discretization` (i.e., no members are modified after this point); hence, `jax` transformations can be applied to subsequent methods by making `self` a static parameter.

<a id="solver_2d.Solver_2D.__init__"></a>

#### \_\_init\_\_

```python
def __init__(grid_x,
             grid_y,
             fixed=False,
             epsilon=1e-12,
             gamma=1,
             fcycle=False,
             skip_pre=False,
             nu1=1,
             nu2=1,
             max_iters_outer=np.iinfo(np.int32).max,
             max_iters_inner=np.iinfo(np.int32).max)
```

Initialize members for solver configuration and compute important compile-time constants.

**Arguments**:

- `grid_x` _grid_1D_ - Linear grid for x-dimension (column-dimension).
- `grid_y` _grid_1D_ - Linear grid for y-dimension (row-dimension).
- `fixed` _bool, optional_ - Perform exactly `max_iters_outer` solver iterations and `max_iters_inner` smoothing iterations for the coarsest grid level instead of relying on a dynamic termination threshold. The solver is compatible with reverse-mode automatic differentiation only if set to `True`. Defaults to `False`.
- `epsilon` _float, optional_ - Solver termination threshold. Defaults to `1e-12`.
- `gamma` _int, optional_ - Number of recursive calls between `defect` and `prolongate` in `multigrid_cycle`. Defaults to `1` (V-cycle).
- `fcycle` _bool, optional_ - Use F-cycle in `multigrid_cycle`. Can be combined with `gamma`. Defaults to `False`.
- `skip_pre` _bool, optional_ - Skip pre-smoothing for repeated cycles for finest/highest resolution (such that no consecutive smoothing calls happen). Defaults to `False`.
- `nu1` _int, optional_ - Number of pre-smoothing iterations (i.e., before `defect`, which relies on `nu1 > 0`). Defaults to `1`.
- `nu2` _int, optional_ - Number of post-smoothing iterations (i.e., after `prolongate`). Defaults to `1`.
- `max_iters_outer` _int, optional_ - The maximum number of solver iterations (i.e., multigrid cycles). Defaults to `int32.max`.
- `max_iters_inner` _int, optional_ - The maximum number of smoothing iterations for coarsest grid level. Defaults to `int32.max`.
  

**Raises**:

- `ValueError` - If either grid is not linear, or the grid resolution is not a power of 2 or not efficient for multigrid configuration, or array shapes are incompatible.

<a id="solver_2d.Solver_2D.compute_shapes"></a>

#### compute\_shapes

```python
def compute_shapes(mx, my)
```

Compute number of levels and shapes for each level statically (i.e., use `numpy` instead of `jax.numpy`).

All array shapes (and types) must be compile-time constants w.r.t. `jax.jit`!

**Arguments**:

- `mx` _int_ - Resolution of grid for x-dimension (column-dimension). Must be a power of 2.
- `my` _int_ - Resolution of grid for y-dimension (row-dimension). Must be a power of 2.
  

**Raises**:

- `ValueError` - If resolution of either grid is not a power of 2 or not efficient for multigrid configuration.

<a id="solver_2d.Solver_2D.compute_idel"></a>

#### compute\_idel

```python
def compute_idel(grid_x, grid_y)
```

Compute reciprocals of the linear grid's cell width for every level and set `self.idel_x` and `self.idel_y` accordingly.

**Arguments**:

- `grid_x` _grid_1D_ - Linear grid for x-dimension (column-dimension).
- `grid_y` _grid_1D_ - Linear grid for y-dimension (row-dimension).
  

**Raises**:

- `ValueError` - If either grid is not linear.

<a id="solver_2d.Solver_2D.compute_diffusion_tensor"></a>

#### compute\_diffusion\_tensor

```python
def compute_diffusion_tensor(D_xx_fine, D_yy_fine, D_xy_fine=None)
```

Compute diffusion tensor components for every level of the multigrid structure (if spatially varying diffusion is desired).

Interpolated values between the gridpoints are used for the diffusion.

**Arguments**:

- `D_xx_fine` _jax.Array_ - `xx` diffusion tensor component for finest/highest resolution.
- `D_yy_fine` _jax.Array_ - `yy` diffusion tensor component for finest/highest resolution.
- `D_xy_fine` _jax.Array, optional_ - `xy` (i.e., off-diagonal) diffusion tensor component for finest/highest resolution. Defaults to `None`.
  

**Returns**:

  List of length `self.levels` for each component.

<a id="solver_2d.Solver_2D.compute_lambda"></a>

#### compute\_lambda

```python
def compute_lambda(lambda_fine)
```

Compute `lambda` (spatially varying decay rate) for every level of the multigrid structure.

**Arguments**:

- `lambda_fine` _jax.Array_ - `lambda` for finest/highest resolution.
  10.2

**Returns**:

  List of length `self.levels`.

<a id="solver_2d.Solver_2D.compute_discretization"></a>

#### compute\_discretization

```python
def compute_discretization(lambda_fine, D_xx, D_yy, D_xy=None)
```

Compute discretization arrays for every level of the multigrid structure and store them as class members (lists of length `self.levels`).

**Arguments**:

- `lambda_fine` _float or jax.Array_ - Either constant or `lambda` for finest/highest resolution.
- `D_xx_fine` _float or jax.Array_ - Either `xx` diffusion constant (scalar) or `xx` diffusion tensor component for finest/highest resolution (if spatially varying diffusion is desired).
- `D_yy_fine` _float or jax.Array_ - Either `yy` diffusion constant (scalar) or `yy` diffusion tensor component for finest/highest resolution (if spatially varying diffusion is desired).
- `D_xy_fine` _jax.Array, optional_ - `xy` diffusion tensor component for finest/highest resolution (if spatially varying diffusion with off-diagonal component is desired). Defaults to `None`.

<a id="solver_2d.Solver_2D.solve"></a>

#### solve

```python
def solve(phi, rhs, lam, D_xx, D_yy, D_xy=None)
```

Multigrid solver for 2D steady-state diffusion–absorption problem.

Iterate (i.e., call `multigrid_cycle`) until `distance(phi_old, phi) <= self.epsilon`.

**Arguments**:

- `phi` _jax.Array_ - Initial value for the unknown `phi` (usually filled with zeros).
- `rhs` _jax.Array_ - Right-hand side of the equation.
- `lam` _float or jax.Array_ - `lambda` (either constant or spatially varying decay rate).
- `D_xx_fine` _float or jax.Array_ - Either `xx` diffusion constant (scalar) or `xx` diffusion tensor component for finest/highest resolution (if spatially varying diffusion is desired).
- `D_yy_fine` _float or jax.Array_ - Either `yy` diffusion constant (scalar) or `yy` diffusion tensor component for finest/highest resolution (if spatially varying diffusion is desired).
- `D_xy_fine` _jax.Array, optional_ - `xy` diffusion tensor component for finest/highest resolution (if spatially varying diffusion with off-diagonal component is desired). Defaults to `None`.
  

**Returns**:

- `phi` _jax.Array_ - Computed solution for `phi`.
- `count` _int_ - Number of iterations performed.
  

**Raises**:

- `ValueError` - If argument shapes or types are incompatible.

<a id="solver_2d.Solver_2D.distance"></a>

#### distance

```python
def distance(phi_old, phi)
```

Compute distance (here: infinity norm of vectors) to previous step.

A step (iteration) corresponds to a call of `multigrid_cycle` within the while loop in `solve`.

**Arguments**:

- `phi_old` _jax.Array_ - Previous value (i.e., before current call to `multigrid_cycle`).
- `phi` _jax.Array_ - Return value of `multigrid_cycle`.
  

**Returns**:

- `float` - Maximum absolute difference between `phi` and `phi_old`.

<a id="solver_2d.Solver_2D.compute_residual"></a>

#### compute\_residual

```python
def compute_residual(phi, rhs)
```

UNUSED.

Compute residual (2D array) by testing the difference of LHS and RHS of the equation.

**Arguments**:

- `phi` _jax.Array_ - Current value for the unknown `phi`.
- `rhs` _jax.Array_ - Right-hand side of the equation.
  

**Returns**:

- `jax.Array` - residual array

<a id="solver_2d.Solver_2D.compute_error"></a>

#### compute\_error

```python
def compute_error(phi, rhs)
```

UNUSED.

Compute discrete error (maximum absolute value in residual array).

**Arguments**:

- `phi` _jax.Array_ - Current value for the unknown `phi`.
- `rhs` _jax.Array_ - Right-hand side of the equation.
  

**Returns**:

- `float` - discrete error

<a id="solver_2d.Solver_2D.get_l2_residual"></a>

#### get\_l2\_residual

```python
def get_l2_residual(phi, rhs)
```

UNUSED.

Compute L2 norm of residual array.

**Arguments**:

- `phi` _jax.Array_ - Current value for the unknown `phi`.
- `rhs` _jax.Array_ - Right-hand side of the equation.
  

**Returns**:

- `float` - L2 norm

<a id="solver_2d.Solver_2D.multigrid_cycle"></a>

#### multigrid\_cycle

```python
def multigrid_cycle(phi, rhs, fcycle=False, level=0, skip_pre=False)
```

Recursively defined multigrid cycle.

Behavior is controlled by members:

- `self.fixed` _bool_ - Perform fixed number (`self.max_iters_inner`) of smoothing iterations for coarsest grid level.
- `self.nu1` _int_ - Smoothing iterations before `defect` / restriction operation (which relies on `self.nu1 > 0`).
- `self.nu2` _int_ - Smoothing iterations after `prolongate`.
- `self.gamma` _int_ - Number of recursive calls between `defect` and `prolongate` (1: V-cycle, 2: W-cycle, ...).

**Arguments**:

- `phi` _jax.Array_ - Current value for the unknown `phi`.
- `rhs` _jax.Array_ - Current right-hand side of the equation.
- `level` _int, optional_ - Current multigrid level (`0`: finest/highest resolution, ..., `self.levels-1`: coarsest/lowest resolution). Defaults to `0`.
- `fcycle` _bool, optional_ - Use F-cycle (i.e., do two recursive calls where the first one uses the F-gamma-cycle recursively and the second one performs a regular gamma-cycle). Defaults to `False`.
- `skip_pre` _bool, optional_ - Skip pre-smoothing for this level (not applied recursively). Defaults to `False`.
  

**Returns**:

- `phi` _jax.Array_ - New value for the unknown `phi`.

<a id="solver_2d.Solver_2D.defect"></a>

#### defect

```python
def defect(phi, rhs, level)
```

Compute defect for current solution and assign to coarser grid.

Corresponds to residual computation and full-weighting restriction.

Assumes that at least one RBGS pre-smoothing step was performed (i.e., that `self.nu1 > 0`).

**Arguments**:

- `phi` _jax.Array_ - Current value for the unknown `phi`.
- `rhs` _jax.Array_ - Current right-hand side of the equation.
- `level` _int_ - Current multigrid level.
  

**Returns**:

- `rhs_c` _jax.Array_ - Residual error downsampled to coarser grid.

<a id="solver_2d.Solver_2D.prolongate"></a>

#### prolongate

```python
def prolongate(phi_c, level)
```

Interpolate the computed coarse-grid correction into a finer grid.

**Arguments**:

- `phi_c` _jax.Array_ - Computed coarse-grid correction (from `level + 1`)
- `level` _int_ - Current multigrid level, corresponding to finer grid.
  

**Returns**:

- `corr` _jax.Array_ - Interpolated correction (to be added to the current value of `phi` on the finer grid).

<a id="solver_2d.Solver_2D.red_black_gauss_seidel"></a>

#### red\_black\_gauss\_seidel

```python
def red_black_gauss_seidel(phi, rhs, level)
```

Red-Black Gauss-Seidel (RBGS) routine for smoothing (i.e., reducing high-frequency errors).

**Arguments**:

- `phi` _jax.Array_ - Current value for the unknown `phi`.
- `rhs` _jax.Array_ - Current right-hand side of the equation.
- `level` _int_ - Current multigrid level.
  

**Returns**:

- `phi` _jax.Array_ - New value for the unknown `phi`.

