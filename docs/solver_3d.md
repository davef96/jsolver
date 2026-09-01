<a id="solver_3d"></a>

# solver\_3d

<a id="solver_3d.solve_3d_simple"></a>

#### solve\_3d\_simple

```python
def solve_3d_simple(grid_x, grid_y, grid_z, phi, rhs, lam, D_xx, D_yy, D_zz)
```

Simplified, functionally pure wrapper function with positional arguments.

See `solve_3d`.

<a id="solver_3d.solve_3d_fixed_simple"></a>

#### solve\_3d\_fixed\_simple

```python
def solve_3d_fixed_simple(grid_x, grid_y, grid_z, iters, phi, rhs, lam, D_xx,
                          D_yy, D_zz)
```

Simplified, functionally pure wrapper function with positional arguments and fixed number of solver iterations (smoothing iterations for coarsest level are set to `1`).

The argument `iters` controls the number of multigrid cycles.

See `solve_3d`.

<a id="solver_3d.solve_3d_fixed"></a>

#### solve\_3d\_fixed

```python
def solve_3d_fixed(grid_x, grid_y, grid_z, iters_outer, iters_inner, phi, rhs,
                   lam, D_xx, D_yy, D_zz)
```

Simplified, functionally pure wrapper function with positional arguments and fixed number of solver iterations.

The argument `iters_outer` controls the number of multigrid cycles, `iters_inner` controls the number of smoothing iterations for the coarsest grid level.

See `solve_3d`.

<a id="solver_3d.solve_3d_fcycle_simple"></a>

#### solve\_3d\_fcycle\_simple

```python
def solve_3d_fcycle_simple(grid_x, grid_y, grid_z, phi, rhs, lam, D_xx, D_yy,
                           D_zz)
```

Simplified, functionally pure wrapper function with positional arguments and F-cycle multigrid.

See `solve_3d`.

<a id="solver_3d.solve_3d_fixed_fcycle_simple"></a>

#### solve\_3d\_fixed\_fcycle\_simple

```python
def solve_3d_fixed_fcycle_simple(grid_x, grid_y, grid_z, iters, phi, rhs, lam,
                                 D_xx, D_yy, D_zz)
```

Simplified, functionally pure wrapper function with positional arguments, fixed number of solver iterations (smoothing iterations for coarsest level are set to `1`) and F-cycle multigrid.

The argument `iters` controls the number of multigrid cycles.

See `solve_3d`.

<a id="solver_3d.solve_3d_fixed_fcycle"></a>

#### solve\_3d\_fixed\_fcycle

```python
def solve_3d_fixed_fcycle(grid_x, grid_y, grid_z, iters_outer, iters_inner,
                          phi, rhs, lam, D_xx, D_yy, D_zz)
```

Simplified, functionally pure wrapper function with positional arguments, fixed number of solver iterations and F-cycle multigrid.

The argument `iters_outer` controls the number of multigrid cycles, `iters_inner` controls the number of smoothing iterations for the coarsest grid level.

See `solve_3d`.

<a id="solver_3d.solve_3d_wcycle_simple"></a>

#### solve\_3d\_wcycle\_simple

```python
def solve_3d_wcycle_simple(grid_x, grid_y, phi, rhs, lam, D_xx, D_yy, D_zz)
```

Simplified, functionally pure wrapper function with positional arguments and W-cycle multigrid.

See `solve_3d`.

<a id="solver_3d.solve_3d_fixed_wcycle_simple"></a>

#### solve\_3d\_fixed\_wcycle\_simple

```python
def solve_3d_fixed_wcycle_simple(grid_x, grid_y, iters, phi, rhs, lam, D_xx,
                                 D_yy, D_zz)
```

Simplified, functionally pure wrapper function with positional arguments, fixed number of solver iterations (smoothing iterations for coarsest level are set to `1`) and W-cycle multigrid.

The argument `iters` controls the number of multigrid cycles.

See `solve_3d`.

<a id="solver_3d.solve_3d_fixed_wcycle"></a>

#### solve\_3d\_fixed\_wcycle

```python
def solve_3d_fixed_wcycle(grid_x, grid_y, iters_outer, iters_inner, phi, rhs,
                          lam, D_xx, D_yy, D_zz)
```

Simplified, functionally pure wrapper function with positional arguments, fixed number of solver iterations and W-cycle multigrid.

The argument `iters_outer` controls the number of multigrid cycles, `iters_inner` controls the number of smoothing iterations for the coarsest grid level.

See `solve_3d`.

<a id="solver_3d.solve_3d"></a>

#### solve\_3d

```python
@partial(jax.jit,
         static_argnames=('grid_x', 'grid_y', 'grid_z', 'fixed', 'epsilon',
                          'gamma', 'fcycle', 'nu1', 'nu2', 'max_iters_outer',
                          'max_iters_inner'))
def solve_3d(grid_x,
             grid_y,
             grid_z,
             *,
             fixed=False,
             epsilon=1e-12,
             gamma=1,
             fcycle=False,
             nu1=1,
             nu2=1,
             max_iters_outer=np.iinfo(np.int32).max,
             max_iters_inner=np.iinfo(np.int32).max,
             phi,
             rhs,
             lam,
             D_xx,
             D_yy,
             D_zz)
```

Wrapper function which is functionally pure and thus compatible with `jax` transformations.

Initializes `Solver_3D` object and calls `solve` method.

Arguments after grid objects are set to keyword-only (`*`) for full support of all optional arguments.

Note that `grid_x` `grid_y`, and `grid_y` must be static arguments for `jax.jit` since array shapes and number of multigrid levels are computed from them.

All arrays must be three-dimensional and of compatible shape: `lam`, `D_xx`, `D_yy`, `D_zz` can be scalars, but `D_xx`, `D_yy`, and `D_zz` must either all be scalar or all of the same shape as the other arrays; all non-scalars must be of the same shape.

**Arguments**:

- `grid_x` _grid_1D_ - Linear grid for x-dimension (third index).
- `grid_y` _grid_1D_ - Linear grid for y-dimension (second index).
- `grid_z` _grid_1D_ - Linear grid for z-dimension (first index).
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
- `D_zz_fine` _float or jax.Array_ - Either `zz` diffusion constant (scalar) or `zz` diffusion tensor component for finest/highest resolution (if spatially varying diffusion is desired).

**Returns**:

- `phi` _jax.Array_ - Computed solution for `phi`.
- `count` _int_ - Number of iterations performed.
  

**Raises**:

- `ValueError` - If either grid is not linear, or the grid resolution is not a power of 2 or not efficient for multigrid configuration, or array shapes are incompatible.

<a id="solver_3d.Solver_3D"></a>

## Solver\_3D Objects

```python
class Solver_3D()
```

A class containing methods implementing an iterative geometric multigrid solver for the 3D steady-state diffusion–absorption problem.

Members are used for convenience, as explicit parameter passing would be quite the hassle (for some functions).
Note that all methods are impure due to the implicit `self` parameter and thus not (directly) compatible with `jax` transformations.
However, all invariants are established after calling `compute_discretization` (i.e., no members are modified after this point); hence, `jax` transformations can be applied to subsequent methods by making `self` a static parameter.

<a id="solver_3d.Solver_3D.__init__"></a>

#### \_\_init\_\_

```python
def __init__(grid_x,
             grid_y,
             grid_z,
             fixed=False,
             epsilon=1e-12,
             gamma=1,
             fcycle=False,
             nu1=1,
             nu2=1,
             max_iters_outer=np.iinfo(np.int32).max,
             max_iters_inner=np.iinfo(np.int32).max)
```

Initialize members for solver configuration and compute important compile-time constants.

**Arguments**:

- `grid_x` _grid_1D_ - Linear grid for x-dimension (third index).
- `grid_y` _grid_1D_ - Linear grid for y-dimension (second index).
- `grid_z` _grid_1D_ - Linear grid for z-dimension (first index).
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

- `ValueError` - If either grid is not linear, or the grid resolution is not a power of 2 or not efficient for multigrid configuration.

<a id="solver_3d.Solver_3D.compute_shapes"></a>

#### compute\_shapes

```python
def compute_shapes(mx, my, mz)
```

Compute number of levels and shapes for each level statically (i.e., use `numpy` instead of `jax.numpy`).

All array shapes (and types) must be compile-time constants w.r.t. `jax.jit`!

**Arguments**:

- `mx` _int_ - Resolution of grid for x-dimension (third index). Must be a power of 2.
- `my` _int_ - Resolution of grid for y-dimension (second index). Must be a power of 2.
- `mz` _int_ - Resolution of grid for z-dimension (first index). Must be a power of 2.
  

**Raises**:

- `ValueError` - If resolution of either grid is not a power of 2 or not efficient for multigrid configuration.

<a id="solver_3d.Solver_3D.compute_idel"></a>

#### compute\_idel

```python
def compute_idel(grid_x, grid_y, grid_z)
```

Compute reciprocals of the linear grid's cell width for every level and set `self.idel_x`, `self.idel_y`, and `self.idel_z` accordingly.

**Arguments**:

- `grid_x` _grid_1D_ - Linear grid for x-dimension (third index).
- `grid_y` _grid_1D_ - Linear grid for y-dimension (second index).
- `grid_z` _grid_1D_ - Linear grid for z-dimension (first index).
  

**Raises**:

- `ValueError` - If either grid is not linear.

<a id="solver_3d.Solver_3D.compute_diffusion_tensor"></a>

#### compute\_diffusion\_tensor

```python
def compute_diffusion_tensor(D_xx_fine, D_yy_fine, D_zz_fine)
```

Compute diffusion tensor components for every level of the multigrid structure (if spatially varying diffusion is desired).

Interpolated values between the gridpoints are used for the diffusion.

**Arguments**:

- `D_xx_fine` _jax.Array_ - `xx` diffusion tensor component for finest/highest resolution.
- `D_yy_fine` _jax.Array_ - `yy` diffusion tensor component for finest/highest resolution.
- `D_zz_fine` _jax.Array_ - `zz` diffusion tensor component for finest/highest resolution.
  

**Returns**:

  List of length `self.levels` for each component.

<a id="solver_3d.Solver_3D.compute_lambda"></a>

#### compute\_lambda

```python
def compute_lambda(lambda_fine)
```

Compute `lambda` (spatially varying decay rate) for every level of the multigrid structure.

**Arguments**:

- `lambda_fine` _jax.Array_ - `lambda` for finest/highest resolution.
  

**Returns**:

  List of length `self.levels`.

<a id="solver_3d.Solver_3D.compute_discretization"></a>

#### compute\_discretization

```python
def compute_discretization(lambda_fine, D_xx, D_yy, D_zz)
```

Compute discretization arrays for every level of the multigrid structure and store them as class members (lists of length `self.levels`).

**Arguments**:

- `lambda_fine` _float or jax.Array_ - Either constant or `lambda` for finest/highest resolution.
- `D_xx_fine` _float or jax.Array_ - Either `xx` diffusion constant (scalar) or `xx` diffusion tensor component for finest/highest resolution (if spatially varying diffusion is desired).
- `D_yy_fine` _float or jax.Array_ - Either `yy` diffusion constant (scalar) or `yy` diffusion tensor component for finest/highest resolution (if spatially varying diffusion is desired).
- `D_zz_fine` _float or jax.Array_ - Either `zz` diffusion constant (scalar) or `zz` diffusion tensor component for finest/highest resolution (if spatially varying diffusion is desired).

<a id="solver_3d.Solver_3D.solve"></a>

#### solve

```python
def solve(phi, rhs, lam, D_xx, D_yy, D_zz)
```

Multigrid solver for 3D steady-state diffusion–absorption problem.

Iterate (i.e., call `multigrid_cycle`) until `distance(phi_old, phi) <= self.epsilon`.

**Arguments**:

- `phi` _jax.Array_ - Initial value for the unknown `phi` (usually filled with zeros).
- `rhs` _jax.Array_ - Right-hand side of the equation.
- `lam` _float or jax.Array_ - `lambda` (either constant or spatially varying decay rate).
- `D_xx_fine` _float or jax.Array_ - Either `xx` diffusion constant (scalar) or `xx` diffusion tensor component for finest/highest resolution (if spatially varying diffusion is desired).
- `D_yy_fine` _float or jax.Array_ - Either `yy` diffusion constant (scalar) or `yy` diffusion tensor component for finest/highest resolution (if spatially varying diffusion is desired).
- `D_zz_fine` _float or jax.Array_ - Either `zz` diffusion constant (scalar) or `zz` diffusion tensor component for finest/highest resolution (if spatially varying diffusion is desired).
  

**Returns**:

- `phi` _jax.Array_ - Computed solution for `phi`.
- `count` _int_ - Number of iterations performed.
  

**Raises**:

- `ValueError` - If argument shapes or types are incompatible.

<a id="solver_3d.Solver_3D.distance"></a>

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

<a id="solver_3d.Solver_3D.compute_residual"></a>

#### compute\_residual

```python
def compute_residual(phi, rhs)
```

UNUSED.

Compute residual (3D array) by testing the difference of LHS and RHS of the equation.

**Arguments**:

- `phi` _jax.Array_ - Current value for the unknown `phi`.
- `rhs` _jax.Array_ - Right-hand side of the equation.
  

**Returns**:

- `jax.Array` - residual array

<a id="solver_3d.Solver_3D.compute_error"></a>

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

<a id="solver_3d.Solver_3D.get_l2_residual"></a>

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

<a id="solver_3d.Solver_3D.multigrid_cycle"></a>

#### multigrid\_cycle

```python
def multigrid_cycle(phi, rhs, fcycle=False, level=0)
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
  

**Returns**:

- `phi` _jax.Array_ - New value for the unknown `phi`.

<a id="solver_3d.Solver_3D.defect"></a>

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

<a id="solver_3d.Solver_3D.prolongate"></a>

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

<a id="solver_3d.Solver_3D.red_black_gauss_seidel"></a>

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

