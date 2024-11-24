import jax.numpy as jnp
import pytest
from jsolver.boundary_handler_2d import do_BCs

@pytest.fixture
def dim():
    return 131

@pytest.fixture
def arr(dim):
    return jnp.ones((dim, dim))

def test_BCs_default(arr, dim):
    new = do_BCs(arr)
    print("arr", arr)
    print("new", new)
    assert jnp.all(new[0] == jnp.zeros(dim))
    assert jnp.all(new[1] == jnp.zeros(dim))
    assert jnp.all(new[-1] == jnp.zeros(dim))
    assert jnp.all(new[-2] == jnp.zeros(dim))
    assert jnp.all(new[:, 0] == jnp.zeros(dim))
    assert jnp.all(new[:, 1] == jnp.zeros(dim))
    assert jnp.all(new[:, -1] == jnp.zeros(dim))
    assert jnp.all(new[:, -2] == jnp.zeros(dim))

def test_BCs_vals(arr, dim):
    twos = jnp.ones(dim) * 2
    new = do_BCs(arr, vals_xb=twos, vals_xe=twos, vals_yb=twos, vals_ye=twos)
    print("arr", arr)
    print("new", new)
    assert jnp.all(new[0] == twos)
    assert jnp.all(new[1] == twos)
    assert jnp.all(new[-1] == twos)
    assert jnp.all(new[-2] == twos)
    assert jnp.all(new[:, 0] == twos)
    assert jnp.all(new[:, 1] == twos)
    assert jnp.all(new[:, -1] == twos)
    assert jnp.all(new[:, -2] == twos)

def test_BCs_xdir(arr, dim):
    new = do_BCs(arr, direction=0)
    print("arr", arr)
    print("new", new)
    assert jnp.all(new[0] == jnp.pad(jnp.ones(dim-4), (2,2), constant_values=(0,0)))
    assert jnp.all(new[1] == jnp.pad(jnp.ones(dim-4), (2,2), constant_values=(0,0)))
    assert jnp.all(new[-1] == jnp.pad(jnp.ones(dim-4), (2,2), constant_values=(0,0)))
    assert jnp.all(new[-2] == jnp.pad(jnp.ones(dim-4), (2,2), constant_values=(0,0)))
    assert jnp.all(new[:, 0] == jnp.zeros(dim))
    assert jnp.all(new[:, 1] == jnp.zeros(dim))
    assert jnp.all(new[:, -1] == jnp.zeros(dim))
    assert jnp.all(new[:, -2] == jnp.zeros(dim))

def test_BCs_ydir(arr, dim):
    new = do_BCs(arr, direction=1)
    print("arr", arr)
    print("new", new)
    assert jnp.all(new[0] == jnp.zeros(dim))
    assert jnp.all(new[1] == jnp.zeros(dim))
    assert jnp.all(new[-1] == jnp.zeros(dim))
    assert jnp.all(new[-2] == jnp.zeros(dim))
    assert jnp.all(new[:, 0] == jnp.pad(jnp.ones(dim-4), (2,2), constant_values=(0,0)))
    assert jnp.all(new[:, 1] == jnp.pad(jnp.ones(dim-4), (2,2), constant_values=(0,0)))
    assert jnp.all(new[:, -1] == jnp.pad(jnp.ones(dim-4), (2,2), constant_values=(0,0)))
    assert jnp.all(new[:, -2] == jnp.pad(jnp.ones(dim-4), (2,2), constant_values=(0,0)))

if __name__ == "__main__":
    pytest.main()
