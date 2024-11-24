import numpy as np
from simple_pytree import Pytree, static_field
from dataclasses import dataclass

@dataclass
class grid_1D(Pytree):
    """
    A class to represent a one-dimensional grid.

    Currently, only linear grids are fully supported.

    Developer notes:

    Class must be immutable in order to be used as a static argument for `jax.jit(solve_2d)`, which is required since array shapes and number of multigrid levels, which are computed from the grid members, must be computed statically, i.e. at trace-time by using `numpy` functions instead of their `jax.numpy` counterpart.

    Array members (i.e. `centers`, `widths`, `edges`) must be static fields in order to remain unchanged by `jax` transformations.

    Since `jax` does not hash array members, `__hash__` and `__eq__` methods must be provided manually such that `jax.jit` can determine whether the given static argument is equivalent to one of a cached, previously compiled function with matching dynamic array shapes.
    """
    mx: int = static_field()
    nx: int = static_field()
    centered : bool = static_field()
    rim : int = static_field()
    xb : float = static_field()
    xe : float = static_field()
    grid_type : int = static_field()
    length : float = static_field()
    del_ : float = static_field()
    idel : float = static_field()
    is_linear : bool = static_field()

    centers : np.ndarray = static_field()
    widths : np.ndarray = static_field()
    edges : np.ndarray = static_field()

    def __hash__(self):
        """
        Hash method.

        Returns:
            Hash of tuple `(mx, centered, grid_type, rim, xb, xe)`.
        """
        return hash((self.mx, self.centered, self.grid_type, self.rim, self.xb, self.xe))

    def __eq__(self, other):
        """
        Equality method.

        Args:
            other (grid_1D): Instance to compare with.

        Returns:
            True iff `(mx, centered, grid_type, rim, xb, xe)` tuples match.
        """
        return isinstance(other, grid_1D) and (self.mx, self.centered, self.grid_type, self.rim, self.xb, self.xe) == (other.mx, other.centered, other.grid_type, other.rim, other.xb, other.xe)

    def __init__(self, mx, xb=0.0, xe=1.0, rim=1, grid_type=0, centered=True):
        """
        Constructor: Initialize essential members and compute rest.

        Will compute members `is_linear`, `nx = mx + 1`, `length`, `del` (cell width), `idel` (reciprocal of cell width), and array members corresponding to grid cells: `edges`, `centers`, `widths`.

        Args:
            mx (int): Resolution.
            xb (float, optional): Begin value. Defaults to `0.0`.
            xe (float, optional): End value. Defaults to `1.0`.
            rim (int, optional): Number of boundary cells. Defaults to `1`.
            grid_type (int, optional): Use linear grid (`0`) or sinusoidal grid (`1`). Defaults to `0`.
            centered (bool, optional): Use centered grid. Defaults to `True`.

        Raises:
            ValueError: If `grid_type` is not supported.
        """
        self.centered = centered
        self.mx = mx
        self.rim = rim
        self.xb = xb
        self.xe = xe
        self.grid_type = grid_type
        self.compute_dependent_vars()
        self.build_grid()

    def compute_dependent_vars(self):
        """
        Compute other variables from essential members.
        """
        self.nx = self.mx + 1
        self.length = self.xe - self.xb
        self.del_ = self.length / (self.mx if self.centered else self.nx)
        self.idel = 1 / self.del_

    def build_grid(self):
        """
        Construct grid.

        Set array members accordingly.

        Raises:
            ValueError: If `self.grid_type` is not supported.
        """
        ipos = np.arange(-self.rim, self.mx + self.rim + 2)

        if self.grid_type == 0:  # Linear grid
            self.edges = (ipos - 0.5 if self.centered else ipos) * self.del_ + self.xb
        elif self.grid_type == 1:  # Sinusoidal grid
            self.edges = (ipos + 2 * np.sin(2 * np.pi * ipos / self.nx)) * self.del_ + self.xb
        else:
            raise ValueError("No such grid grid_type: {}".format(self.grid_type))

        shifted = self.edges[1:]
        edges = self.edges[:-1]
        self.centers = 0.5 * (edges + shifted)
        self.widths = shifted - edges

        self.is_linear = self.grid_type == 0

