<a id="boundary_handler_2d"></a>

# boundary\_handler\_2d

<a id="boundary_handler_2d.do_BCs"></a>

#### do\_BCs

```python
def do_BCs(arr,
           rim=1,
           direction=-1,
           vals_xb=None,
           vals_xe=None,
           vals_yb=None,
           vals_ye=None)
```

Boundary conditions on 2D plane.

**Arguments**:

- `arr` _jax.Array_ - Two-dimensional input array.
- `rim` _int, optional_ - Number of boundary cells (see below). Defaults to `1`.
- `direction` _int, optional_ - Restrict BCs to x-direction (`0`), y-direction (`1`), or allow all directions (`-1`). Defaults to `-1`.
- `vals_xb` _jax.Array, optional_ - One-dimensional array of shape `arr.shape[0]` for setting columns from beginning. Defaults to `None`.
- `vals_xe` _jax.Array, optional_ - One-dimensional array of shape `arr.shape[0]` for setting columns from end. Defaults to `None`.
- `vals_yb` _jax.Array, optional_ - One-dimensional array of shape `arr.shape[1]` for setting rows from beginning. Defaults to `None`.
- `vals_ye` _jax.Array, optional_ - One-dimensional array of shape `arr.shape[1]` for setting rows from end. Defaults to `None`.
  

**Returns**:

- `arr` _jax.Array_ - With first and last `rim + 1` rows / columns (depending on `direction`) set to either zero or the corresponding `vals` array (if given).

