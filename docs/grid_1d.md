<a id="grid_1d"></a>

# grid\_1d

<a id="grid_1d.grid_1D"></a>

## grid\_1D Objects

```python
@jax.tree_util.register_static
class grid_1D()
```

A class to represent a one-dimensional grid.

Currently, only linear grids are fully supported.

Developer notes:

Class must be immutable in order to be used as a static argument for `jax.jit(solve)`, which is required since array shapes and number of multigrid levels (which both are computed from the grid_1D members) must be computed statically, i.e., at trace-time by using `numpy` functions instead of their `jax.numpy` counterpart.

In particular, array members (i.e., `centers`, `widths`, `edges`) must be static fields in order to remain unchanged by `jax` transformations.

Since `jax` does not hash array members, `__hash__` and `__eq__` methods must be provided manually such that `jax.jit` can determine whether the given static argument is equivalent to one of a cached, previously compiled function with matching dynamic array shapes.

<a id="grid_1d.grid_1D.__hash__"></a>

#### \_\_hash\_\_

```python
def __hash__()
```

Hash method.

**Returns**:

  Hash of tuple `(mx, centered, grid_type, rim, xb, xe)`.

<a id="grid_1d.grid_1D.__eq__"></a>

#### \_\_eq\_\_

```python
def __eq__(other)
```

Equality method.

**Arguments**:

- `other` _grid_1D_ - Instance to compare with.
  

**Returns**:

  True iff `(mx, centered, grid_type, rim, xb, xe)` tuples match.

<a id="grid_1d.grid_1D.__init__"></a>

#### \_\_init\_\_

```python
def __init__(mx, xb=0.0, xe=1.0, rim=1, grid_type=0, centered=True)
```

Constructor: Initialize essential members and compute rest.

Will compute members `is_linear`, `nx = mx + 1`, `length`, `del` (cell width), `idel` (reciprocal of cell width), and array members corresponding to grid cells: `edges`, `centers`, `widths`.

**Arguments**:

- `mx` _int_ - Resolution.
- `xb` _float, optional_ - Begin value. Defaults to `0.0`.
- `xe` _float, optional_ - End value. Defaults to `1.0`.
- `rim` _int, optional_ - Number of boundary cells. Defaults to `1`.
- `grid_type` _int, optional_ - Use linear grid (`0`) or sinusoidal grid (`1`). Defaults to `0`.
- `centered` _bool, optional_ - Use centered grid. Defaults to `True`.
  

**Raises**:

- `ValueError` - If `grid_type` is not supported.

<a id="grid_1d.grid_1D.compute_dependent_vars"></a>

#### compute\_dependent\_vars

```python
def compute_dependent_vars()
```

Compute other variables from essential members.

<a id="grid_1d.grid_1D.build_grid"></a>

#### build\_grid

```python
def build_grid()
```

Construct grid.

Set array members accordingly.

**Raises**:

- `ValueError` - If `self.grid_type` is not supported.

