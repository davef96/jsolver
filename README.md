# JAX-Native Differentiable Multigrid Solver for Diffusion–Absorption

[![CI](https://github.com/davef96/jsolver/actions/workflows/tests.yml/badge.svg)](https://github.com/davef96/jsolver/actions/workflows/tests.yml)

Iterative geometric multigrid solver for steady-state diffusion–absorption problem of the following form:

```math
\nabla\cdot\left(D\nabla\phi\right) - \lambda\phi = \mathrm{rhs}
```

- `D`: diffusion tensor field
- `phi`: unknown scalar field (e.g., particle density)
- `lambda`: scalar field containing local absorption coefficients
- `rhs`: scalar field for source term (e.g., particle density)

This PDE is discretized on a regular Cartesian grid using finite differences, and it is solved iteratively by performing multigrid cycles (e.g., V-, W-, or F-cycles). The solver either performs a fixed number of cycles (*fixed mode*) or iterates until the convergence criterion is met (*dynamic mode*).

It is implemented in [`Python`](https://www.python.org/) by using [`jax`](https://github.com/jax-ml/jax) to support automatic differentiation (AD) and to speed up (in particular, GPU-accelerate) the solver for *many* repeated evaluations. Note that the *first* solver, [`jax.jvp`](https://jax.readthedocs.io/en/latest/_autosummary/jax.jvp.html), or [`jax.vjp`](https://jax.readthedocs.io/en/latest/_autosummary/jax.vjp.html) call will be extremely slow due to just-in-time (JIT) compilation. After that, the respective routine can be called with different arguments *of the same type and shape* without triggering recompilation. JIT-compiled functions [can be stored on disk](https://jax.readthedocs.io/en/latest/persistent_compilation_cache.html) to avoid recompilation across runs of your Python script.

Note that reverse-mode AD (e.g., `jax.vjp`) is supported **only in fixed mode**. Reason: the *dynamic* solver implementation relies on a loop of the form `while distance > epsilon`, where [`distance`](docs/solver_2d.md#distance) is a *dynamic value* (i.e., it is *traced* by `jax` since it is computed by `jax` functions) and thus the loop must be implemented in terms of [`jax.lax.while_loop`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.while_loop.html) to be compatible with [`jax.jit`](https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html); however, `jax.lax.while_loop` is incompatible with reverse-mode AD. In fixed mode, the solver uses a [`jax.lax.fori_loop`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.fori_loop.html) with static bounds instead.

## Installation

1. Set up a `venv` using `python -m venv <name>` and then activate it using the appropriate script for your shell, e.g., via `source <name>/bin/activate`.

2. Run `pip install .` to install this project as the module `jsolver` such that all imports can be found; optionally, add `-e` for editable mode. The appropriate `jax` package (CPU-only or with `CUDA12`/`CUDA13`) will be selected, and all requirements will be installed automatically.

## Dependencies

The following packages are used to run and test the solver:

- `jax` ([GitHub](https://github.com/jax-ml/jax), [PyPI](https://pypi.org/project/jax/))
- `pytest` ([GitHub](https://github.com/pytest-dev/pytest), [PyPI](https://pypi.org/project/pytest/))
- `pytest-cov` ([GitHub](https://github.com/pytest-dev/pytest-cov), [PyPI](https://pypi.org/project/pytest-cov/))

## Usage

Linear grid objects ([`grid_1d.py`](jsolver/grid_1d.py)) have to be created to set up the discretized steady-state diffusion–absorption problem.
The solver implementation ([`solver_2d.py`](jsolver/solver_2d.py), [`solver_3d.py`](jsolver/solver_3d.py)) provides some high-level wrapper functions for convenience (see [API Docs](#api-docs)).

Example (solver call for `problem 1` in 2D):

```python
import numpy as np
from jsolver.grid_1d import grid_1D
from jsolver.solver_2d import solve_2d_simple

# create linear grid objects with desired resolution
x_res = 128 # column-dimension
y_res = 128 # row-dimension
grid_x = grid_1D(x_res) # centered linear (regular) grid for interval [0,1]
grid_y = grid_1D(y_res)
arr_shape = (y_res + 3, x_res + 3) # row-dimension comes first in numpy!

# set up problem using numpy ("broadcasting")
D_xx = 1.0
D_yy = 1.0
phi = np.zeros(arr_shape)
lam = 0.0
x = grid_x.centers[np.newaxis]    # row vector
y = grid_y.centers[:, np.newaxis] # column vector
ox = 1. - x
oy = 1. - y
prod_x3 = x * ox ** 3
prod_y = y * oy
phi_ana = prod_x3 * prod_y
rhs = -2 * prod_x3 - 6 * ox * (ox - x) * prod_y

# solve problem
phi_res, iterations = solve_2d_simple(grid_x, grid_y, phi, rhs, lam, D_xx, D_yy)
```

This example can be found in [`example2D.py`](examples/example2D.py).

The following plot shows heatmaps for the computed arrays; `abs_err` represents `abs(phi_res - phi_ana)`.

![Problem 1 Heatmaps](data/heatmap_problem_1_128x128.gif)

Note that the computations for the problem setup are done using [`numpy`](https://github.com/numpy/numpy) (instead of `jax.numpy`) since it is much faster when only called once.
The arrays will be implicitly promoted to [`jax.Array`](https://jax.readthedocs.io/en/latest/_autosummary/jax.Array.html)s for the internal solver implementation upon invocation.

Also note that importing the solver will call `jax.config.update('jax_enable_x64', True)`, which globally sets the default `dtype` width to `64 bit` for **all** `jax` operations, resulting in `int64` and `float64` computations. While the solver requires this setting to work properly, this might not be desired in the context of high-performance machine learning: refer to [`default_dtypes`](https://docs.jax.dev/en/latest/default_dtypes.html) to check whether `jax` now allows to apply this setting locally; if not, the `dtype` of the respective arrays has to be set manually to allow for mixed-precision programs.

For further examples see [`examples`](examples/) (where [`benchmark.py`](examples/benchmark.py) also shows `jax.jvp` and `jax.vjp` calls), [`test_solver_2d.py`](tests/test_solver_2d.py), and [`test_solver_3d.py`](tests/test_solver_3d.py).

<a id="api-docs"></a>

## API Docs

- [boundary_handler_2d](docs/boundary_handler_2d.md), [boundary_handler_3d](docs/boundary_handler_3d.md) (used internally by the solver to apply Dirichlet boundary conditions; calls would need to be changed to apply non-zero values)
- [grid_1d](docs/grid_1d.md)
- [solver_2d](docs/solver_2d.md), [solver_3d](docs/solver_3d.md)

Generated using [`pydoc-markdown`](https://github.com/NiklasRosenstein/pydoc-markdown) in the [`jsolver`](jsolver) directory as follows:

```sh
pydoc-markdown -m <module_name> -I $(pwd) > ../docs/<module_name>.md
```

Note that `<module_name>` must **not** contain the `.py` ending.

## Tests

The tests can be run manually by using `pytest tests`. It is possible to run only a specific test for a specific file: e.g. `pytest tests/test_solver_2d.py::test1`. A coverage report can be created by adding the option `--cov=.`.

Note that the acceptable bounds are set relatively tight (for some tests). On a different system (hardware or installed library versions) you may have to adapt `acceptable_error_abs` or `acceptable_error_rel` for the `setup` routines in [`test_solver_2d.py`](tests/test_solver_2d.py) and [`test_solver_3d.py`](tests/test_solver_3d.py).

## Performance

Remarks, tips, and tricks regarding performance can be found in [`performance.md`](performance.md).

## Further Reading

- The solver is based on [Picard](https://arxiv.org/pdf/1401.4035) (see `p. 6` for details regarding the discretization).
- It is used as `JMGS` in [this paper](https://arxiv.org/pdf/2608.00760) to provide a comparison against `DMGS`, a `C++` implementation of the solver that is interfaced with `jax` and uses a hand-derived adjoint definition, in the context of variational inference.

