# API Docs

- [boundary_handler_2d](boundary_handler_2d.md)
- [boundary_handler_3d](boundary_handler_3d.md)
- [grid_1d](grid_1d.md)
- [solver_2d](solver_2d.md)
- [solver_3d](solver_3d.md)

## Updating the Docs

Generated using [`pydoc-markdown`](https://github.com/NiklasRosenstein/pydoc-markdown) in the [`jsolver`](../jsolver) directory as follows:

```sh
pydoc-markdown -m <module_name> -I $(pwd) > ../docs/<module_name>.md
```

Note that `<module_name>` must **not** contain the `.py` ending.

