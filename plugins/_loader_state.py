"""Interpreter-lifetime state for cooperative managed plugin imports.

Kept outside the reloadable loader and anchored on sys, so removing/reimporting
our modules cannot create a second finder or lose an in-flight thread's stack.
This private object has no reset API; it is not a boundary against hostile Python
mutating sys, sys.modules, or sys.meta_path.
"""

import _imp
import importlib.util
from pathlib import Path
import sys
import threading


class _ImportState:
    def __init__(self):
        self.local = threading.local()
        self.lock = threading.RLock()
        self.failed_domains: set[str] = set()
        self.source_paths: dict[str, Path] = {}

    def check(self, domain: str) -> None:
        with self.lock:
            if any(
                domain == failed
                or domain.startswith(f"{failed}.")
                or failed.startswith(f"{domain}.")
                for failed in self.failed_domains
            ):
                raise ImportError(
                    f"plugin package {domain!r} failed; a fresh process is required"
                )

    def poison(self, domain: str) -> None:
        with self.lock:
            self.failed_domains.add(domain)

    def find_spec(self, fullname: str, path=None, target=None):
        """Resolve only temporarily registered exact source names; no observation."""
        with self.lock:
            source_file = self.source_paths.get(fullname)
        if source_file is None:
            return None
        return importlib.util.spec_from_file_location(
            fullname,
            source_file,
            submodule_search_locations=(
                [str(source_file.parent)] if source_file.name == "__init__.py" else None
            ),
        )


# The import lock protects only singleton publication, never plugin execution.
# sys is interpreter-lifetime storage, not an importlib implementation override.
_imp.acquire_lock()
try:
    if not hasattr(sys, "_hermes_plugin_import_state"):
        sys._hermes_plugin_import_state = _ImportState()
        sys.meta_path.insert(0, sys._hermes_plugin_import_state)
    STATE = sys._hermes_plugin_import_state
finally:
    _imp.release_lock()
