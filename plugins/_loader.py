"""Fail-closed managed imports; see plugins/LOADER_CONTRACT.md for boundaries."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib
import importlib._bootstrap as _bootstrap
import importlib.machinery
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Iterator

from plugins._loader_state import STATE as _STATE


class ModuleLoadDeadlockError(ImportError):
    """A plugin load would complete a cross-module import-lock cycle."""


class ModuleCallbackReentryError(ImportError):
    """Plugin code was requested while an outer loader transaction was held."""


class _MissingDeadlockError(Exception):
    """Sentinel used only when a Python runtime renames its private error."""


_MODULE_LOCK_DEADLOCK_ERRORS = tuple(
    error
    for error in (
        getattr(_bootstrap, "_ModuleLockDeadlockError", None),
        getattr(_bootstrap, "_DeadlockError", None),
    )
    if isinstance(error, type) and issubclass(error, BaseException)
) or (_MissingDeadlockError,)

# Aliases are private diagnostics; lifetime is owned by the stable state object.
_SOURCE_MODULE_FINDER = _STATE
_SOURCE_MODULE_PATHS = _STATE.source_paths


@contextmanager
def _registered_source_module(
    module_name: str,
    source_file: Path,
) -> Iterator[None]:
    """Temporarily teach normal importlib how to find an exact source path."""
    state = _STATE
    canonical_source = source_file.resolve()
    with state.lock:
        previous_source = state.source_paths.get(module_name)
        if previous_source is not None and previous_source != canonical_source:
            raise ImportError(f"plugin module {module_name!r} belongs to another path")
        state.source_paths[module_name] = canonical_source
    try:
        yield
    finally:
        with state.lock:
            if previous_source is None:
                state.source_paths.pop(module_name, None)
            else:
                state.source_paths[module_name] = previous_source


@contextmanager
def module_load_transaction(module_name: str) -> Iterator[None]:
    """Hold CPython's module lock through readiness and failure publication.

    No public importlib API exposes this boundary. `_get_module_lock` and its
    private deadlock exceptions are CPython dependencies. Callbacks must run
    after this context exits, not merely after module execution finishes.
    """
    state = _STATE
    lock = _bootstrap._get_module_lock(module_name)  # type: ignore[attr-defined]
    try:
        lock.acquire()
    except _MODULE_LOCK_DEADLOCK_ERRORS as exc:
        raise ModuleLoadDeadlockError(
            f"deadlock while loading plugin module {module_name!r}"
        ) from exc
    try:
        stack = getattr(state.local, "stack", None)
        if stack is None:
            stack = []
            state.local.stack = stack
        stack.append(module_name)
        try:
            yield
        finally:
            stack.pop()
    finally:
        lock.release()


def ensure_plugin_callbacks_allowed() -> None:
    """Reject plugin callbacks while any outer shared import is in progress."""
    stack = getattr(_STATE.local, "stack", ())
    if stack:
        raise ModuleCallbackReentryError(
            f"plugin callback blocked while loader transaction {stack[-1]!r} is active"
        )


def module_is_initializing(module: ModuleType) -> bool:
    """Return whether importlib is still executing *module*."""
    return bool(getattr(getattr(module, "__spec__", None), "_initializing", False))


def module_matches_path(module: ModuleType, source_file: Path) -> bool:
    """Return whether a cached module came from the requested canonical path."""
    cached_file = getattr(module, "__file__", None)
    if not cached_file:
        return False
    try:
        return Path(cached_file).resolve() == source_file.resolve()
    except (OSError, RuntimeError, TypeError):
        return False


def path_scoped_module_name(namespace: str, child: str, source_file: Path) -> str:
    """Return a stable synthetic name derived from a module's canonical path."""
    canonical_path = source_file.resolve()
    path_digest = hashlib.sha256(os.fsencode(canonical_path)).hexdigest()
    if child.isidentifier():
        module_child = child
    else:
        readable_child = "".join(
            character if character.isascii() and character.isalnum() else "_"
            for character in child
        ).strip("_")
        if not readable_child:
            readable_child = "plugin"
        if readable_child[0].isdigit():
            readable_child = f"_{readable_child}"
        child_digest = hashlib.sha256(os.fsencode(child)).hexdigest()[:12]
        module_child = f"{readable_child[:48]}_{child_digest}"
    return f"{namespace}_{path_digest}.{module_child}"


def _entry_point_module_name(entry_point: Any) -> str:
    """Return the exact import target declared by an entry point."""
    try:
        module_name = getattr(entry_point, "module", "")
    except (AttributeError, TypeError, ValueError):
        module_name = ""
    if not isinstance(module_name, str) or not module_name:
        value = getattr(entry_point, "value", "")
        if isinstance(value, str):
            module_name = value.partition(":")[0].strip()
    if (
        not isinstance(module_name, str)
        or not module_name
        or module_name.startswith(".")
        or module_name.endswith(".")
        or ".." in module_name
    ):
        raise ImportError("plugin entry point has an invalid module target")
    return module_name


def import_entry_point(entry_point: Any) -> Any:
    """Load an entry point, poisoning its package after failed import."""
    ensure_plugin_callbacks_allowed()
    state = _STATE
    module_name = _entry_point_module_name(entry_point)
    try:
        attributes = importlib.metadata.EntryPoint(
            name="plugin", value=entry_point.value, group="plugin"
        ).attr
    except (AttributeError, TypeError, ValueError) as exc:
        raise ImportError("plugin entry point has an invalid attribute target") from exc
    parts = module_name.split(".")
    if (
        len(parts) >= 3
        and parts[0] == "plugins"
        and parts[1]
        in {
            "memory",
            "cron_providers",
            "context_engine",
        }
    ):
        domain = ".".join(parts[:3])
    else:
        domain = parts[0]

    with module_load_transaction(module_name):
        state.check(domain)
        cached = sys.modules.get(module_name)
        if cached is not None and module_is_initializing(cached):
            raise ImportError(f"plugin module {module_name!r} is still initializing")

        try:
            loaded = importlib.import_module(module_name)
            module = sys.modules.get(module_name)
            if module is None or module_is_initializing(module):
                raise ImportError(f"plugin module {module_name!r} is not ready")
        except BaseException:
            state.poison(domain)
            raise
    ensure_plugin_callbacks_allowed()
    try:
        for attribute in attributes.split(".") if attributes else ():
            loaded = getattr(loaded, attribute)
    except BaseException:
        state.poison(domain)
        raise
    return loaded


def ensure_namespace_package(module_name: str, search_path: Path) -> None:
    """Publish an empty, path-backed namespace used by normal importlib loads."""
    canonical_path = search_path.resolve()
    with module_load_transaction(module_name):
        cached = sys.modules.get(module_name)
        if cached is not None:
            if module_is_initializing(cached):
                raise ImportError(
                    f"namespace package {module_name!r} is still initializing"
                )
            try:
                cached_paths = [Path(path).resolve() for path in cached.__path__]
            except (AttributeError, OSError, RuntimeError, TypeError) as exc:
                raise ImportError(
                    f"namespace package {module_name!r} has an invalid search path"
                ) from exc
            if cached_paths != [canonical_path] or getattr(cached, "__file__", None):
                raise ImportError(
                    f"namespace package {module_name!r} belongs to another path"
                )
            return

        spec = importlib.machinery.ModuleSpec(module_name, None, is_package=True)
        spec.submodule_search_locations = [str(canonical_path)]
        sys.modules[module_name] = importlib.util.module_from_spec(spec)


def import_module_from_path(
    module_name: str,
    source_file: Path,
    *,
    package_dir: Path | None = None,
) -> ModuleType:
    """Import ready code, retaining published state and poisoning failed domains."""
    state = _STATE
    domain = module_name.rsplit(".", 1)[0] if package_dir is not None else module_name
    with module_load_transaction(module_name):
        state.check(domain)
        if package_dir is not None:
            namespace_name = module_name.rsplit(".", 1)[0]
            ensure_namespace_package(namespace_name, package_dir)

        cached = sys.modules.get(module_name)
        if cached is not None:
            if module_is_initializing(cached):
                raise ImportError(
                    f"plugin module {module_name!r} is still initializing"
                )
            if not module_matches_path(cached, source_file):
                raise ImportError(
                    f"plugin module {module_name!r} belongs to another path"
                )
        elif any(name.startswith(f"{module_name}.") for name in list(sys.modules)):
            state.poison(domain)
            raise ImportError(
                f"plugin module {module_name!r} has orphaned descendants; a fresh process is required"
            )

        with _registered_source_module(module_name, source_file):
            try:
                module = importlib.import_module(module_name)
                if module_is_initializing(module) or not module_matches_path(
                    module, source_file
                ):
                    raise ImportError(
                        f"plugin module {module_name!r} has an invalid origin"
                    )
            except BaseException:
                state.poison(domain)
                raise
        return module
