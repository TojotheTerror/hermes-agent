"""Test-only cleanup for synthetic modules; never clears lifetime poison."""

import sys


def evict_module_tree(module_name: str) -> None:
    """Remove fixture modules after bounded workers have finished."""
    root = sys.modules.get(module_name)
    for name in list(sys.modules):
        if name == module_name or name.startswith(f"{module_name}."):
            sys.modules.pop(name, None)
    if root is not None and "." in module_name:
        parent_name, child = module_name.rsplit(".", 1)
        parent = sys.modules.get(parent_name)
        if parent is not None and vars(parent).get(child) is root:
            delattr(parent, child)
