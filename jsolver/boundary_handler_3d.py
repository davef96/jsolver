import jax.numpy as jnp

def do_BCs3D(arr, rim=1, direction=-1, vals_xb=None, vals_xe=None, vals_yb=None, vals_ye=None, vals_zb=None, vals_ze=None):
    """
    Boundary conditions on 3D plane.

    Args:
        arr (jax.Array): Three-dimensional input array.
        rim (int, optional): Number of boundary cells (see below). Defaults to `1`.
        direction (int, optional): Restrict BCs to x-direction (`0`), y-direction (`1`), or allow all directions (`-1`). Defaults to `-1`.
        vals_xb (jax.Array, optional): Two-dimensional array of shape `arr.shape[0,1]` for setting dims 1,2 from beginning. Defaults to `None`.
        vals_xe (jax.Array, optional): Two-dimensional array of shape `arr.shape[0,1]` for setting dims 1,2 from end. Defaults to `None`.
        vals_yb (jax.Array, optional): Two-dimensional array of shape `arr.shape[0,2]` for setting dims 0,2 from beginning. Defaults to `None`.
        vals_ye (jax.Array, optional): Two-dimensional array of shape `arr.shape[0,2]` for setting dims 0,2 from end. Defaults to `None`.
        vals_zb (jax.Array, optional): Two-dimensional array of shape `arr.shape[1,2]` for setting dims 1,2 from beginning. Defaults to `None`.
        vals_ze (jax.Array, optional): Two-dimensional array of shape `arr.shape[1,2]` for setting dims 1,2 from end. Defaults to `None`.

    Returns:
        arr (jax.Array): With first and last `rim + 1` rows / columns (depending on `direction`) set to either zero or the corresponding `vals` array (if given).
    """
    upper = rim + 1

    if direction == -1 or direction == 0:
        arr = arr.at[:,:, :upper].set(0 if vals_xb is None else vals_xb[:,:, jnp.newaxis])
        arr = arr.at[:,:, -upper:].set(0 if vals_xe is None else vals_xe[:,:, jnp.newaxis])

    if direction == -1 or direction == 1:
        arr = arr.at[:,:upper,:].set(0 if vals_yb is None else vals_yb[:,jnp.newaxis, :])
        arr = arr.at[:,-upper:,:].set(0 if vals_ye is None else vals_ye[:,jnp.newaxis, :])

    if direction == -1 or direction == 2:
        arr = arr.at[:upper,:,:].set(0 if vals_yb is None else vals_yb[jnp.newaxis, :,:])
        arr = arr.at[-upper:,:,:].set(0 if vals_ye is None else vals_ye[jnp.newaxis, :,:])

    return arr

