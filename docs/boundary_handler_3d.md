<a id="boundary_handler_3d"></a>

# boundary\_handler\_3d

<a id="boundary_handler_3d.do_BCs"></a>

#### do\_BCs

```python
def do_BCs(arr,
           rim=1,
           direction=-1,
           vals_xb=None,
           vals_xe=None,
           vals_yb=None,
           vals_ye=None,
           vals_zb=None,
           vals_ze=None)
```

Boundary conditions on 3D plane.

**Arguments**:

- `arr` _jax.Array_ - Three-dimensional input array.
- `rim` _int, optional_ - Number of boundary cells (see below). Defaults to `1`.
- `direction` _int, optional_ - Restrict BCs to x-direction (`0`), y-direction (`1`), or allow all directions (`-1`). Defaults to `-1`.
- `vals_xb` _jax.Array, optional_ - Two-dimensional array of shape `arr.shape[0,1]` for setting dims 1,2 from beginning. Defaults to `None`.
- `vals_xe` _jax.Array, optional_ - Two-dimensional array of shape `arr.shape[0,1]` for setting dims 1,2 from end. Defaults to `None`.
- `vals_yb` _jax.Array, optional_ - Two-dimensional array of shape `arr.shape[0,2]` for setting dims 0,2 from beginning. Defaults to `None`.
- `vals_ye` _jax.Array, optional_ - Two-dimensional array of shape `arr.shape[0,2]` for setting dims 0,2 from end. Defaults to `None`.
- `vals_zb` _jax.Array, optional_ - Two-dimensional array of shape `arr.shape[1,2]` for setting dims 1,2 from beginning. Defaults to `None`.
- `vals_ze` _jax.Array, optional_ - Two-dimensional array of shape `arr.shape[1,2]` for setting dims 1,2 from end. Defaults to `None`.
  

**Returns**:

- `arr` _jax.Array_ - With first and last `rim + 1` rows / columns (depending on `direction`) set to either zero or the corresponding `vals` array (if given).

