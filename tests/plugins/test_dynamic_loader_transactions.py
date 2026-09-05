"""Cross-loader transaction regressions for synthetic plugin packages."""

from __future__ import annotations

import importlib
import importlib.metadata
import subprocess
import sys
import threading
import time
import types
from dataclasses import dataclass
from pathlib import Path

import pytest


@dataclass(frozen=True)
class _LoaderCase:
    name: str
    loader_module: str
    load_function: str
    module_prefix: str
    root_target: str
    root_target_is_getter: bool


_LOADER_CASES = (
    pytest.param(
        _LoaderCase(
            "memory",
            "plugins.memory",
            "load_memory_provider",
            "_hermes_user_memory",
            "_get_user_plugins_dir",
            True,
        ),
        id="memory",
    ),
    pytest.param(
        _LoaderCase(
            "cron",
            "plugins.cron_providers",
            "load_cron_scheduler",
            "_hermes_user_cron",
            "_get_user_plugins_dir",
            True,
        ),
        id="cron",
    ),
    pytest.param(
        _LoaderCase(
            "context",
            "plugins.context_engine",
            "load_context_engine",
            "plugins.context_engine",
            "_CONTEXT_ENGINE_PLUGINS_DIR",
            False,
        ),
        id="context",
    ),
)

# Context engines are repository-scoped; only user memory/cron roots can vary
# by profile inside one process.
_EXTERNAL_LOADER_CASES = _LOADER_CASES[:2]

_BUNDLED_ROOT_TARGETS = {
    "memory": "_MEMORY_PLUGINS_DIR",
    "cron": "_CRON_PLUGINS_DIR",
    "context": "_CONTEXT_ENGINE_PLUGINS_DIR",
}

_BUNDLED_MODULE_PREFIXES = {
    "memory": "plugins.memory",
    "cron": "plugins.cron_providers",
    "context": "plugins.context_engine",
}


def _module_source(
    family: str,
    *,
    prelude: str = "",
    name_expression: str = "'loaded'",
    instance_init: str = "",
    extra_methods: str = "",
    postlude: str = "",
) -> str:
    if family == "memory":
        return (
            "from agent.memory_provider import MemoryProvider\n"
            f"{prelude}"
            "class SyntheticMemory(MemoryProvider):\n"
            f"{instance_init}"
            "    @property\n"
            f"    def name(self): return {name_expression}\n"
            "    def is_available(self): return True\n"
            "    def initialize(self, **kw): pass\n"
            "    def sync_turn(self, *a, **kw): pass\n"
            "    def get_tool_schemas(self): return []\n"
            "    def handle_tool_call(self, *a, **kw): return '{}'\n"
            f"{extra_methods}"
            "def register(ctx): ctx.register_memory_provider(SyntheticMemory())\n"
            f"{postlude}"
        )
    if family == "cron":
        return (
            "from cron.scheduler_provider import CronScheduler\n"
            f"{prelude}"
            "class SyntheticCron(CronScheduler):\n"
            "    @property\n"
            f"    def name(self): return {name_expression}\n"
            "    def start(self, stop_event, **kwargs): pass\n"
            f"{extra_methods}"
            "def register(ctx): ctx.register_cron_scheduler(SyntheticCron())\n"
            f"{postlude}"
        )
    if family == "context":
        return (
            "from agent.context_engine import ContextEngine\n"
            f"{prelude}"
            "class SyntheticContext(ContextEngine):\n"
            "    @property\n"
            f"    def name(self): return {name_expression}\n"
            "    def should_compress(self, prompt_tokens=None): return False\n"
            "    def compress(self, messages, **kwargs): return messages\n"
            "    def update_from_response(self, usage): pass\n"
            "def register(ctx): ctx.register_context_engine(SyntheticContext())\n"
            f"{postlude}"
        )
    raise AssertionError(f"unknown loader family: {family}")


def _write_plugin(
    root: Path,
    slug: str,
    source: str,
    *,
    helper_value: str | None = None,
) -> Path:
    plugin_dir = root / slug
    plugin_dir.mkdir(parents=True)
    init_file = plugin_dir / "__init__.py"
    init_file.write_text(source, encoding="utf-8")
    if helper_value is not None:
        (plugin_dir / "helper.py").write_text(
            f"VALUE = {helper_value!r}\n",
            encoding="utf-8",
        )
    return init_file


def _evict_module_tree(module_name: str) -> None:
    for loaded_name in list(sys.modules):
        if loaded_name == module_name or loaded_name.startswith(f"{module_name}."):
            sys.modules.pop(loaded_name, None)
    if "." not in module_name:
        return
    parent_name, child_name = module_name.rsplit(".", 1)
    parent = sys.modules.get(parent_name)
    if parent is not None and hasattr(parent, child_name):
        delattr(parent, child_name)


def _set_plugins_root(
    monkeypatch: pytest.MonkeyPatch,
    loader_module: types.ModuleType,
    case: _LoaderCase,
    root: Path,
) -> None:
    if case.root_target_is_getter:
        monkeypatch.setattr(loader_module, case.root_target, lambda: root)
    else:
        monkeypatch.setattr(loader_module, case.root_target, root)
        monkeypatch.setattr(loader_module, "__path__", [str(root)])


def _module_name_for(
    loader_module: types.ModuleType,
    case: _LoaderCase,
    package_dir: Path,
) -> str:
    if case.root_target_is_getter:
        return getattr(loader_module, "_provider_module_name")(package_dir)
    return f"{case.module_prefix}.{package_dir.name}"


@pytest.mark.parametrize("case", _LOADER_CASES)
def test_normal_import_waits_for_package_execution(
    case: _LoaderCase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A normal importer must not receive a partially executed package root."""
    loader_module = importlib.import_module(case.loader_module)
    slug = f"visibility{case.name}"
    probe_name = f"_hermes_loader_visibility_{case.name}"
    probe = types.ModuleType(probe_name)
    probe_started = threading.Event()
    probe_release = threading.Event()
    setattr(probe, "started", probe_started)
    setattr(probe, "release", probe_release)
    monkeypatch.setitem(sys.modules, probe_name, probe)

    prelude = (
        f"import {probe_name} as probe\n"
        "probe.started.set()\n"
        "if not probe.release.wait(timeout=5):\n"
        "    raise RuntimeError('visibility probe timed out')\n"
        "READY = 'ready'\n"
    )
    init_file = _write_plugin(
        tmp_path,
        slug,
        _module_source(case.name, prelude=prelude, name_expression="READY"),
    )
    _set_plugins_root(monkeypatch, loader_module, case, tmp_path)
    module_name = _module_name_for(loader_module, case, init_file.parent)
    load = getattr(loader_module, case.load_function)
    loader_result: dict[str, object] = {}
    normal_result: dict[str, object] = {}
    errors: list[str] = []
    normal_started = threading.Event()
    normal_done = threading.Event()

    def dynamic_load() -> None:
        try:
            loader_result["provider"] = load(slug)
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(f"dynamic: {exc!r}")

    def normal_import() -> None:
        normal_started.set()
        try:
            assert module_name is not None
            module = importlib.import_module(module_name)
            normal_result["release_was_set"] = probe_release.is_set()
            normal_result["ready_at_return"] = getattr(module, "READY", None)
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(f"normal: {exc!r}")
        finally:
            normal_done.set()

    dynamic_thread = threading.Thread(target=dynamic_load, daemon=True)
    normal_thread = threading.Thread(target=normal_import, daemon=True)
    _evict_module_tree(module_name)

    try:
        dynamic_thread.start()
        assert probe_started.wait(timeout=2), "package execution never started"
        normal_thread.start()
        assert normal_started.wait(timeout=2), "normal importer never started"
        finished_before_release = normal_done.wait(timeout=0.25)
    finally:
        probe_release.set()
        dynamic_thread.join(timeout=3)
        normal_thread.join(timeout=3)
        _evict_module_tree(module_name)

    assert not dynamic_thread.is_alive(), "dynamic loader did not finish"
    assert not normal_thread.is_alive(), "normal importer did not finish"
    assert errors == []
    assert finished_before_release is False
    assert normal_result == {
        "release_was_set": True,
        "ready_at_return": "ready",
    }
    provider = loader_result["provider"]
    assert provider is not None
    assert getattr(provider, "name") == "ready"


@pytest.mark.parametrize("case", _LOADER_CASES)
def test_waiting_load_is_denied_after_failed_import(
    case: _LoaderCase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed import poisons the package before its waiting loader proceeds."""
    from importlib import _bootstrap

    loader_module = importlib.import_module(case.loader_module)
    load = getattr(loader_module, case.load_function)
    slug = f"waitingretry{case.name}"
    probe_name = f"_hermes_loader_waiting_retry_{case.name}"
    probe = types.ModuleType(probe_name)
    root_started = threading.Event()
    release_failure = threading.Event()
    setattr(probe, "root_started", root_started)
    setattr(probe, "release_failure", release_failure)
    monkeypatch.setitem(sys.modules, probe_name, probe)
    valid_source = _module_source(
        case.name,
        prelude="from .helper import VALUE\n",
        name_expression="VALUE",
    )
    failing_source = _module_source(
        case.name,
        prelude=(
            "from .helper import VALUE\n"
            f"import {probe_name} as probe\n"
            "probe.root_started.set()\n"
            "if not probe.release_failure.wait(timeout=5):\n"
            "    raise RuntimeError('failure release timed out')\n"
            "raise RuntimeError('root failed')\n"
        ),
        name_expression="VALUE",
    )
    init_file = _write_plugin(
        tmp_path,
        slug,
        failing_source,
        helper_value="stale",
    )
    _set_plugins_root(monkeypatch, loader_module, case, tmp_path)
    module_name = _module_name_for(loader_module, case, init_file.parent)
    _evict_module_tree(module_name)
    results: dict[str, object] = {}
    errors: list[BaseException] = []

    def run(label: str) -> None:
        try:
            results[label] = load(slug)
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    first = threading.Thread(target=run, args=("first",), daemon=True)
    second = threading.Thread(target=run, args=("second",), daemon=True)
    waiter_observed = False
    try:
        first.start()
        assert root_started.wait(timeout=2), "failing package never reached its barrier"
        (init_file.parent / "helper.py").write_text(
            "VALUE = 'fresh-value'\n",
            encoding="utf-8",
        )
        init_file.write_text(valid_source, encoding="utf-8")
        importlib.invalidate_caches()
        second.start()
        module_lock = _bootstrap._get_module_lock(module_name)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if getattr(module_lock, "waiters", 0) >= 1:
                waiter_observed = True
                break
            time.sleep(0.005)
    finally:
        release_failure.set()
        first.join(timeout=3)
        second.join(timeout=3)
        _evict_module_tree(module_name)

    assert waiter_observed, "retry did not reach the package import lock"
    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    assert results["first"] is None
    assert results["second"] is None


@pytest.mark.parametrize("case", _LOADER_CASES)
def test_loader_refuses_orphans_from_failed_normal_import(
    case: _LoaderCase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Visible ordinary-import leftovers are refused, not silently repaired."""
    from plugins._loader import ensure_namespace_package

    loader_module = importlib.import_module(case.loader_module)
    load = getattr(loader_module, case.load_function)
    slug = f"normalretry{case.name}"
    valid_source = _module_source(
        case.name,
        prelude="from .helper import VALUE\n",
        name_expression="VALUE",
    )
    failing_source = valid_source.replace(
        "from .helper import VALUE\n",
        "from .helper import VALUE\nraise RuntimeError('normal import failed')\n",
    )
    init_file = _write_plugin(
        tmp_path,
        slug,
        failing_source,
        helper_value="stale",
    )
    _set_plugins_root(monkeypatch, loader_module, case, tmp_path)
    module_name = _module_name_for(loader_module, case, init_file.parent)
    if case.root_target_is_getter:
        ensure_namespace_package(module_name.rsplit(".", 1)[0], tmp_path)
    _evict_module_tree(module_name)

    try:
        with pytest.raises(RuntimeError, match="normal import failed"):
            importlib.import_module(module_name)
        assert module_name not in sys.modules
        assert f"{module_name}.helper" in sys.modules

        (init_file.parent / "helper.py").write_text(
            "VALUE = 'fresh-value'\n",
            encoding="utf-8",
        )
        init_file.write_text(valid_source, encoding="utf-8")
        importlib.invalidate_caches()
        helper = sys.modules[f"{module_name}.helper"]
        provider = load(slug)
        assert sys.modules.get(f"{module_name}.helper") is helper
    finally:
        _evict_module_tree(module_name)

    assert provider is None


@pytest.mark.parametrize("case", _EXTERNAL_LOADER_CASES)
def test_same_slug_from_new_path_loads_requested_module_tree(
    case: _LoaderCase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Switching profile/plugin roots must not reuse another root's package."""
    loader_module = importlib.import_module(case.loader_module)
    slug = f"profile{case.name}"
    source = _module_source(
        case.name,
        prelude="from .helper import VALUE\n",
        name_expression="VALUE",
    )
    root_a = tmp_path / "profile-a"
    root_b = tmp_path / "profile-b"
    _write_plugin(root_a, slug, source, helper_value="profile-a")
    init_b = _write_plugin(root_b, slug, source, helper_value="profile-b")
    load = getattr(loader_module, case.load_function)
    try:
        _set_plugins_root(monkeypatch, loader_module, case, root_a)
        first = load(slug)
        assert first is not None
        first_module_name = type(first).__module__
        _set_plugins_root(monkeypatch, loader_module, case, root_b)
        second = load(slug)
        assert second is not None
        second_module_name = type(second).__module__
        cached_origin = getattr(sys.modules[second_module_name], "__file__", None)
    finally:
        for module_name in {
            name
            for name in (
                locals().get("first_module_name"),
                locals().get("second_module_name"),
            )
            if isinstance(name, str)
        }:
            _evict_module_tree(module_name)

    assert first.name == "profile-a"
    assert second.name == "profile-b"
    assert second is not first
    if case.root_target_is_getter:
        assert second_module_name != first_module_name
    assert cached_origin is not None
    assert Path(cached_origin).resolve() == init_b.resolve()


@pytest.mark.parametrize("case", _LOADER_CASES)
def test_successful_package_load_binds_parent_attribute(
    case: _LoaderCase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every successful package root must have normal parent bindings."""
    loader_module = importlib.import_module(case.loader_module)
    slug = f"parentbinding{case.name}"
    _write_plugin(
        tmp_path,
        slug,
        _module_source(case.name, name_expression=repr(slug)),
    )
    _set_plugins_root(monkeypatch, loader_module, case, tmp_path)
    provider = getattr(loader_module, case.load_function)(slug)
    assert provider is not None
    module_name = type(provider).__module__

    try:
        root_module = sys.modules[module_name]
        parent_name, child_name = module_name.rsplit(".", 1)
        parent_module = sys.modules[parent_name]
        bound_module = getattr(parent_module, child_name, None)
    finally:
        _evict_module_tree(module_name)

    assert bound_module is root_module


@pytest.mark.parametrize("case", _LOADER_CASES)
def test_same_thread_recursive_load_rejects_initializing_module(
    case: _LoaderCase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recursive loader calls must not extract from a partial package root."""
    loader_module = importlib.import_module(case.loader_module)
    load = getattr(loader_module, case.load_function)
    slug = f"recursive{case.name}"
    probe_name = f"_hermes_loader_recursive_{case.name}"
    probe = types.ModuleType(probe_name)
    probe.recursing = False
    probe.nested = "not-called"
    probe.register_calls = 0
    monkeypatch.setitem(sys.modules, probe_name, probe)
    postlude = (
        f"import {probe_name} as probe\n"
        "_original_register = register\n"
        "def register(ctx):\n"
        "    probe.register_calls += 1\n"
        "    _original_register(ctx)\n"
        "if not probe.recursing:\n"
        "    probe.recursing = True\n"
        "    probe.nested = probe.load()\n"
    )
    _write_plugin(
        tmp_path,
        slug,
        _module_source(case.name, name_expression=repr(slug), postlude=postlude),
    )
    _set_plugins_root(monkeypatch, loader_module, case, tmp_path)
    probe.load = lambda: load(slug)

    provider = load(slug)
    module_name = type(provider).__module__ if provider is not None else None
    if module_name is not None:
        _evict_module_tree(module_name)

    assert provider is not None
    assert probe.nested is None
    assert probe.register_calls == 1


@pytest.mark.parametrize("case", _LOADER_CASES)
def test_register_callback_can_wait_for_same_loader_thread(
    case: _LoaderCase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider callbacks run after import locks have been released."""
    loader_module = importlib.import_module(case.loader_module)
    load = getattr(loader_module, case.load_function)
    slug = f"callbackreentry{case.name}"
    probe_name = f"_hermes_loader_callback_reentry_{case.name}"
    probe = types.ModuleType(probe_name)
    probe.spawned = False
    probe.blocked = False
    probe.done = threading.Event()
    probe.worker = None
    probe.worker_provider = None
    monkeypatch.setitem(sys.modules, probe_name, probe)
    postlude = (
        "import threading\n"
        f"import {probe_name} as probe\n"
        "_original_register = register\n"
        "def register(ctx):\n"
        "    if not probe.spawned:\n"
        "        probe.spawned = True\n"
        "        probe.worker = threading.Thread(target=probe.load_in_worker, daemon=True)\n"
        "        probe.worker.start()\n"
        "        if not probe.done.wait(timeout=1):\n"
        "            probe.blocked = True\n"
        "    _original_register(ctx)\n"
    )
    _write_plugin(
        tmp_path,
        slug,
        _module_source(case.name, name_expression=repr(slug), postlude=postlude),
    )
    _set_plugins_root(monkeypatch, loader_module, case, tmp_path)

    def load_in_worker() -> None:
        try:
            probe.worker_provider = load(slug)
        finally:
            probe.done.set()

    probe.load_in_worker = load_in_worker
    provider = load(slug)
    assert probe.done.wait(timeout=3), "callback worker did not finish"
    assert probe.worker is not None
    probe.worker.join(timeout=1)
    module_names = {
        type(candidate).__module__
        for candidate in (provider, probe.worker_provider)
        if candidate is not None
    }
    for module_name in module_names:
        _evict_module_tree(module_name)

    assert not probe.worker.is_alive()
    assert probe.blocked is False
    assert provider is not None
    assert probe.worker_provider is not None


@pytest.mark.parametrize("case", _LOADER_CASES)
def test_bundled_origin_mismatch_fails_closed_without_replacement(
    case: _LoaderCase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stable bundled name must never hot-replace another canonical path."""
    loader_module = importlib.import_module(case.loader_module)
    slug = f"originmismatch{case.name}"
    provider_dir = tmp_path / slug
    _write_plugin(
        tmp_path,
        slug,
        _module_source(case.name, name_expression=repr(slug)),
    )
    monkeypatch.setattr(loader_module, _BUNDLED_ROOT_TARGETS[case.name], tmp_path)
    module_name = f"{_BUNDLED_MODULE_PREFIXES[case.name]}.{slug}"
    stale_root = types.ModuleType(module_name)
    stale_root.__file__ = str(tmp_path / "other" / slug / "__init__.py")
    stale_child = types.ModuleType(f"{module_name}.helper")
    monkeypatch.setitem(sys.modules, module_name, stale_root)
    monkeypatch.setitem(sys.modules, f"{module_name}.helper", stale_child)
    parent_name, child_name = module_name.rsplit(".", 1)
    parent = sys.modules[parent_name]
    monkeypatch.setattr(parent, child_name, stale_root, raising=False)

    provider = getattr(loader_module, case.load_function)(slug)

    assert provider is None
    assert sys.modules[module_name] is stale_root
    assert sys.modules[f"{module_name}.helper"] is stale_child
    assert getattr(parent, child_name) is stale_root


@pytest.mark.parametrize("case", _EXTERNAL_LOADER_CASES)
def test_concurrent_profile_loads_keep_lazy_imports_path_scoped(
    case: _LoaderCase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same-slug profile providers must retain their own lazy-import package."""
    loader_module = importlib.import_module(case.loader_module)
    load = getattr(loader_module, case.load_function)
    slug = f"concurrent{case.name}"
    source = _module_source(
        case.name,
        name_expression=repr(slug),
        extra_methods=(
            "    def lazy_value(self):\n"
            "        from .lazy import VALUE\n"
            "        return VALUE\n"
        ),
    )
    root_a = tmp_path / "profile-a"
    root_b = tmp_path / "profile-b"
    _write_plugin(root_a, slug, source)
    _write_plugin(root_b, slug, source)
    (root_a / slug / "lazy.py").write_text("VALUE = 'profile-a'\n", encoding="utf-8")
    (root_b / slug / "lazy.py").write_text("VALUE = 'profile-b'\n", encoding="utf-8")

    profile_root = threading.local()
    monkeypatch.setattr(
        loader_module,
        case.root_target,
        lambda: getattr(profile_root, "value", None),
    )
    first_loaded = threading.Event()
    second_loaded = threading.Event()
    providers: dict[str, object] = {}
    lazy_values: dict[str, str] = {}
    errors: list[str] = []

    def load_first() -> None:
        try:
            profile_root.value = root_a
            providers["a"] = load(slug)
            first_loaded.set()
            if not second_loaded.wait(timeout=5):
                raise RuntimeError("profile B did not finish loading")
            lazy_values["a"] = getattr(providers["a"], "lazy_value")()
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(f"profile-a: {exc!r}")

    def load_second() -> None:
        try:
            if not first_loaded.wait(timeout=5):
                raise RuntimeError("profile A did not finish loading")
            profile_root.value = root_b
            providers["b"] = load(slug)
            lazy_values["b"] = getattr(providers["b"], "lazy_value")()
            second_loaded.set()
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(f"profile-b: {exc!r}")
            second_loaded.set()

    threads = [
        threading.Thread(target=load_first, daemon=True),
        threading.Thread(target=load_second, daemon=True),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=7)

    module_names = {
        type(provider).__module__
        for provider in providers.values()
        if provider is not None
    }
    for module_name in module_names:
        _evict_module_tree(module_name)

    assert not any(thread.is_alive() for thread in threads)
    assert errors == []
    assert providers["a"] is not None
    assert providers["b"] is not None
    assert type(providers["a"]).__module__ != type(providers["b"]).__module__
    assert lazy_values == {"a": "profile-a", "b": "profile-b"}


@pytest.mark.parametrize("case", _LOADER_CASES)
def test_failed_directory_keeps_consumed_helper_but_requires_fresh_process(
    case, tmp_path, monkeypatch
):
    loader_module = importlib.import_module(case.loader_module)
    slug = f"retained{case.name}"
    probe = types.ModuleType(f"_retained_probe_{case.name}")
    probe.started = threading.Event()
    probe.release = threading.Event()
    monkeypatch.setitem(sys.modules, probe.__name__, probe)
    valid = _module_source(
        case.name, prelude="from .helper import VALUE\n", name_expression="VALUE"
    )
    init_file = _write_plugin(
        tmp_path,
        slug,
        valid + f"\nimport {probe.__name__} as probe\nprobe.started.set()\n"
        "assert probe.release.wait(5)\nraise RuntimeError('root failed')\n",
        helper_value="retained",
    )
    _set_plugins_root(monkeypatch, loader_module, case, tmp_path)
    module_name = _module_name_for(loader_module, case, init_file.parent)
    load = getattr(loader_module, case.load_function)
    results = []
    worker = threading.Thread(target=lambda: results.append(load(slug)), daemon=True)
    worker.start()
    try:
        assert probe.started.wait(3)
        partial_root = sys.modules[module_name]
        helper = importlib.import_module(f"{module_name}.helper")
        assert partial_root.helper is helper
    finally:
        probe.release.set()
        worker.join(5)
    try:
        assert not worker.is_alive()
        assert results == [None]
        assert sys.modules.get(f"{module_name}.helper") is helper
        assert partial_root.helper is helper
        init_file.write_text(valid, encoding="utf-8")
        (init_file.parent / "helper.py").write_text(
            "VALUE = 'corrected-fresh'\n", encoding="utf-8"
        )
        importlib.invalidate_caches()
        assert load(slug) is None
        _evict_module_tree(module_name)
        assert load(slug) is None
        repo = Path(__file__).resolve().parents[2]
        script = """
import sys
from pathlib import Path
repo = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(repo))
import plugins._loader as shared
assert Path(shared.__file__).resolve() == repo / 'plugins' / '_loader.py'
import importlib
family = importlib.import_module(sys.argv[2])
root = Path(sys.argv[3])
if sys.argv[4] == '_CONTEXT_ENGINE_PLUGINS_DIR':
    setattr(family, sys.argv[4], root)
    family.__path__ = [str(root)]
else:
    setattr(family, sys.argv[4], lambda: root)
provider = getattr(family, sys.argv[5])(sys.argv[6])
assert provider is not None
assert provider.name == 'corrected-fresh'
"""
        child = subprocess.run(
            [
                sys.executable,
                "-I",
                "-c",
                script,
                str(repo),
                case.loader_module,
                str(tmp_path),
                case.root_target,
                case.load_function,
                slug,
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert child.returncode == 0, child.stdout + child.stderr
    finally:
        _evict_module_tree(module_name)


def test_cached_memory_module_reinvokes_register_per_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cache code while re-running the provider-construction callback per load."""
    import plugins.memory as memory_plugins

    slug = "isolatedmemory"
    source = _module_source(
        "memory",
        name_expression="'isolatedmemory'",
        instance_init="    def __init__(self): self.pending_syncs = []\n",
        postlude=(
            "REGISTER_CALLS = 0\n"
            "_original_register = register\n"
            "def register(ctx):\n"
            "    global REGISTER_CALLS\n"
            "    REGISTER_CALLS += 1\n"
            "    _original_register(ctx)\n"
        ),
    )
    init_file = _write_plugin(tmp_path, slug, source)
    monkeypatch.setattr(memory_plugins, "_get_user_plugins_dir", lambda: tmp_path)
    module_name = memory_plugins._provider_module_name(init_file.parent)
    _evict_module_tree(module_name)

    try:
        first = memory_plugins.load_memory_provider(slug)
        cached_after_first = sys.modules[module_name]
        second = memory_plugins.load_memory_provider(slug)
        cached_after_second = sys.modules[module_name]
    finally:
        _evict_module_tree(module_name)

    assert first is not None
    assert second is not None
    assert cached_after_second is cached_after_first
    assert cached_after_second.REGISTER_CALLS == 2
    assert second is not first
    getattr(first, "pending_syncs").append("agent-a")
    assert getattr(second, "pending_syncs") == []


def test_cross_family_nested_loads_do_not_deadlock(
    tmp_path: Path,
) -> None:
    """Cross-family cycles must fail nested loads instead of invoking callbacks."""
    memory_root = tmp_path / "memory"
    cron_root = tmp_path / "cron"
    probe_name = "_hermes_loader_abba_probe"
    barrier_prelude = (
        "import threading\n"
        f"import {probe_name} as probe\n"
        "try:\n"
        "    probe.barrier.wait(timeout=2)\n"
        "except threading.BrokenBarrierError:\n"
        "    pass\n"
    )
    _write_plugin(
        memory_root,
        "abbamemory",
        _module_source(
            "memory",
            prelude=f"{barrier_prelude}probe.load_cron()\n",
            name_expression="'abbamemory'",
        ),
    )
    _write_plugin(
        cron_root,
        "abbacron",
        _module_source(
            "cron",
            prelude=f"{barrier_prelude}probe.load_memory()\n",
            name_expression="'abbacron'",
        ),
    )

    script = """
import sys
import threading
import time
import types
from pathlib import Path

import plugins.cron_providers as cron_plugins
import plugins.memory as memory_plugins

memory_root = Path(sys.argv[1])
cron_root = Path(sys.argv[2])
memory_plugins._get_user_plugins_dir = lambda: memory_root
cron_plugins._get_user_plugins_dir = lambda: cron_root
probe = types.ModuleType("_hermes_loader_abba_probe")
probe.barrier = threading.Barrier(2)
probe.nested = []

def nested(label, load):
    result = load()
    probe.nested.append((label, result is None))
    return result

probe.load_memory = lambda: nested(
    "memory", lambda: memory_plugins.load_memory_provider("abbamemory")
)
probe.load_cron = lambda: nested(
    "cron", lambda: cron_plugins.load_cron_scheduler("abbacron")
)
sys.modules[probe.__name__] = probe

start = threading.Barrier(2)
results = {}
errors = []

def run(label, load):
    try:
        start.wait(timeout=2)
        results[label] = load()
    except BaseException as exc:
        errors.append(f"{label}: {exc!r}")

threads = [
    threading.Thread(
        target=run,
        args=("memory", lambda: memory_plugins.load_memory_provider("abbamemory")),
        daemon=True,
    ),
    threading.Thread(
        target=run,
        args=("cron", lambda: cron_plugins.load_cron_scheduler("abbacron")),
        daemon=True,
    ),
]
for thread in threads:
    thread.start()
deadline = time.monotonic() + 5
for thread in threads:
    thread.join(timeout=max(0, deadline - time.monotonic()))
alive = [thread.name for thread in threads if thread.is_alive()]
if alive:
    print(f"deadlocked={alive}")
    raise SystemExit(2)
if errors:
    print(f"errors={errors}")
    raise SystemExit(3)
if results["memory"] is None or results["cron"] is None:
    print(f"missing_provider={results}")
    raise SystemExit(4)
if sorted(probe.nested) != [("cron", True), ("memory", True)]:
    print(f"nested_loads_did_not_both_fail_closed={probe.nested}")
    raise SystemExit(5)
print(
    f"completed=memory:{results['memory'].name},cron:{results['cron'].name};"
    f"nested={probe.nested}"
)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(memory_root), str(cron_root)],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "completed=memory:abbamemory,cron:abbacron" in completed.stdout


class _EntryPoints(list):
    """Small selectable entry-point collection used by public discovery."""

    def select(self, *, group: str):
        return [entry_point for entry_point in self if entry_point.group == group]


def _install_memory_entry_points(
    monkeypatch: pytest.MonkeyPatch,
    memory_plugins: types.ModuleType,
    *entry_points: importlib.metadata.EntryPoint,
) -> None:
    monkeypatch.setattr(memory_plugins, "_get_user_plugins_dir", lambda: None)
    monkeypatch.setattr(memory_plugins, "_get_project_plugins_dir", lambda: None)
    monkeypatch.setattr(
        memory_plugins.importlib.metadata,
        "entry_points",
        lambda: _EntryPoints(entry_points),
    )


def _install_memory_entry_point(
    monkeypatch: pytest.MonkeyPatch,
    memory_plugins: types.ModuleType,
    *,
    name: str,
    value: str,
) -> importlib.metadata.EntryPoint:
    entry_point = importlib.metadata.EntryPoint(
        name=name,
        value=value,
        group=memory_plugins.ENTRY_POINTS_GROUP,
    )
    _install_memory_entry_points(monkeypatch, memory_plugins, entry_point)
    return entry_point


def _prepare_memory_entry_point_sibling_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    memory_plugins: types.ModuleType,
    *,
    package_name: str,
) -> tuple[types.ModuleType, object, types.ModuleType]:
    package_dir = tmp_path / package_name
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text(
        "PREEXISTING = object()\n",
        encoding="utf-8",
    )
    (package_dir / "fail_helper.py").write_text(
        "VALUE = 'failed-transaction'\n",
        encoding="utf-8",
    )
    (package_dir / "good_helper.py").write_text(
        "VALUE = 'good'\n",
        encoding="utf-8",
    )
    (package_dir / "shared_helper.py").write_text(
        "raise RuntimeError('first shared helper attempt failed')\n",
        encoding="utf-8",
    )

    probe_name = f"_{package_name}_probe"
    probe = types.ModuleType(probe_name)
    setattr(probe, "fail_started", threading.Event())
    setattr(probe, "release_failure", threading.Event())
    monkeypatch.setitem(sys.modules, probe_name, probe)
    (package_dir / "fail.py").write_text(
        "try:\n"
        "    from . import shared_helper\n"
        "except RuntimeError:\n"
        "    pass\n"
        "from . import fail_helper\n"
        f"import {probe_name} as probe\n"
        "probe.fail_started.set()\n"
        "if not probe.release_failure.wait(timeout=5):\n"
        "    raise RuntimeError('entry-point sibling release timed out')\n"
        "raise RuntimeError('entry-point sibling failed')\n",
        encoding="utf-8",
    )
    (package_dir / "good.py").write_text(
        _module_source(
            "memory",
            prelude=(
                "from . import good_helper, shared_helper\n"
                "VALUE = good_helper.VALUE\n"
                "SHARED_VALUE = shared_helper.VALUE\n"
            ),
            name_expression="VALUE",
        ),
        encoding="utf-8",
    )

    monkeypatch.syspath_prepend(str(tmp_path))
    package_before = importlib.import_module(package_name)
    marker_before = package_before.PREEXISTING
    entry_points = (
        importlib.metadata.EntryPoint(
            name="failing-sibling",
            value=f"{package_name}.fail:VALUE",
            group=memory_plugins.ENTRY_POINTS_GROUP,
        ),
        importlib.metadata.EntryPoint(
            name="successful-sibling",
            value=f"{package_name}.good:SyntheticMemory",
            group=memory_plugins.ENTRY_POINTS_GROUP,
        ),
    )
    _install_memory_entry_points(monkeypatch, memory_plugins, *entry_points)
    return package_before, marker_before, probe


def _assert_successful_sibling_survived_failure(
    package_name: str,
    package_before: types.ModuleType,
    marker_before: object,
    good_module: types.ModuleType,
    good_helper: types.ModuleType,
    shared_helper: types.ModuleType,
) -> None:
    package_after = sys.modules.get(package_name)
    assert package_after is package_before
    assert getattr(package_after, "PREEXISTING", None) is marker_before
    assert getattr(good_module, "VALUE", None) == "good"
    assert getattr(good_module, "SHARED_VALUE", None) == "shared-good"
    assert sys.modules.get(f"{package_name}.good") is good_module
    assert sys.modules.get(f"{package_name}.good_helper") is good_helper
    assert sys.modules.get(f"{package_name}.shared_helper") is shared_helper
    assert getattr(package_after, "good", None) is good_module
    assert getattr(package_after, "good_helper", None) is good_helper
    assert getattr(package_after, "shared_helper", None) is shared_helper
    assert f"{package_name}.fail" not in sys.modules
    assert f"{package_name}.fail_helper" in sys.modules
    assert not hasattr(package_after, "fail")
    assert package_after.fail_helper is sys.modules[f"{package_name}.fail_helper"]


def test_memory_entry_point_waiters_refuse_failed_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Package failure denies its target and another entry point in that package."""
    from importlib import _bootstrap

    import plugins.memory as memory_plugins

    package_name = "ful103_retry_entrypoint_package"
    provider_name = "retry-entrypoint"
    package_dir = tmp_path / package_name
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text(
        "PREEXISTING = object()\n",
        encoding="utf-8",
    )
    helper_file = package_dir / "helper.py"
    helper_file.write_text("VALUE = 'stale'\n", encoding="utf-8")
    provider_file = package_dir / "provider.py"
    probe_name = "_ful103_retry_entrypoint_probe"
    probe = types.ModuleType(probe_name)
    probe.started = threading.Event()
    probe.release_failure = threading.Event()
    monkeypatch.setitem(sys.modules, probe_name, probe)

    valid_source = _module_source(
        "memory",
        prelude="from .helper import VALUE\n",
        name_expression="VALUE",
    )
    provider_file.write_text(
        valid_source.replace(
            "from .helper import VALUE\n",
            "from .helper import VALUE\n"
            f"import {probe_name} as probe\n"
            "probe.started.set()\n"
            "if not probe.release_failure.wait(timeout=5):\n"
            "    raise RuntimeError('entry-point failure release timed out')\n"
            "raise RuntimeError('entry-point provider failed')\n",
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    package_before = importlib.import_module(package_name)
    marker_before = package_before.PREEXISTING
    _install_memory_entry_point(
        monkeypatch,
        memory_plugins,
        name=provider_name,
        value=f"{package_name}.provider:register",
    )
    module_name = f"{package_name}.provider"
    results: dict[str, object] = {}
    errors: list[BaseException] = []

    def load(label: str) -> None:
        try:
            results[label] = memory_plugins.load_memory_provider(
                provider_name,
                register_skills=False,
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    first = threading.Thread(target=load, args=("first",), daemon=True)
    second = threading.Thread(target=load, args=("second",), daemon=True)
    waiter_observed = False
    try:
        first.start()
        assert probe.started.wait(timeout=2), "failed entry point never reached barrier"
        helper_file.write_text("VALUE = 'fresh-value'\n", encoding="utf-8")
        provider_file.write_text(valid_source, encoding="utf-8")
        importlib.invalidate_caches()
        second.start()
        module_lock = _bootstrap._get_module_lock(module_name)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if getattr(module_lock, "waiters", 0) >= 1:
                waiter_observed = True
                break
            time.sleep(0.005)
    finally:
        probe.release_failure.set()
        first.join(timeout=3)
        second.join(timeout=3)

    package_after = sys.modules.get(package_name)
    try:
        assert waiter_observed, "retry did not wait on the entry-point module lock"
        assert not first.is_alive()
        assert not second.is_alive()
        assert errors == []
        assert results["first"] is None
        assert results["second"] is None
        _install_memory_entry_point(
            monkeypatch,
            memory_plugins,
            name=provider_name,
            value=f"{package_name}.helper:VALUE",
        )
        from plugins._loader import import_entry_point

        with pytest.raises(ImportError, match="fresh process"):
            import_entry_point(memory_plugins.find_provider_entry_point(provider_name))
        assert package_after is package_before
        assert getattr(package_after, "PREEXISTING", None) is marker_before
    finally:
        _evict_module_tree(package_name)


def test_failed_memory_entry_point_preserves_successful_entry_point_sibling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed public load cannot erase a concurrent successful sibling load."""
    import plugins.memory as memory_plugins

    package_name = "ful103_public_entrypoint_sibling_package"
    package_before, marker_before, probe = _prepare_memory_entry_point_sibling_race(
        tmp_path,
        monkeypatch,
        memory_plugins,
        package_name=package_name,
    )
    fail_started = getattr(probe, "fail_started")
    release_failure = getattr(probe, "release_failure")
    results: dict[str, object] = {}
    errors: list[str] = []

    def load_failing_sibling() -> None:
        try:
            results["failing"] = memory_plugins.load_memory_provider(
                "failing-sibling",
                register_skills=False,
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(f"failing: {exc!r}")

    def load_successful_sibling() -> None:
        try:
            results["successful"] = memory_plugins.load_memory_provider(
                "successful-sibling",
                register_skills=False,
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(f"successful: {exc!r}")

    failing_thread = threading.Thread(target=load_failing_sibling, daemon=True)
    successful_thread = threading.Thread(target=load_successful_sibling, daemon=True)
    good_module_name = f"{package_name}.good"
    try:
        failing_thread.start()
        assert fail_started.wait(timeout=2), "failing entry point missed its barrier"
        (tmp_path / package_name / "shared_helper.py").write_text(
            "VALUE = 'shared-good'\n",
            encoding="utf-8",
        )
        importlib.invalidate_caches()
        successful_thread.start()
        successful_thread.join(timeout=3)
        successful_finished_before_failure = not successful_thread.is_alive()
        good_module_before_failure = sys.modules.get(good_module_name)
        good_helper_before_failure = sys.modules.get(f"{package_name}.good_helper")
        shared_helper_before_failure = sys.modules.get(f"{package_name}.shared_helper")
    finally:
        release_failure.set()
        failing_thread.join(timeout=3)
        successful_thread.join(timeout=3)

    try:
        assert successful_finished_before_failure
        assert not failing_thread.is_alive()
        assert not successful_thread.is_alive()
        assert errors == []
        assert results["failing"] is None
        successful_provider = results["successful"]
        assert successful_provider is not None
        assert getattr(successful_provider, "name") == "good"
        assert isinstance(good_module_before_failure, types.ModuleType)
        assert isinstance(good_helper_before_failure, types.ModuleType)
        assert isinstance(shared_helper_before_failure, types.ModuleType)
        _assert_successful_sibling_survived_failure(
            package_name,
            package_before,
            marker_before,
            good_module_before_failure,
            good_helper_before_failure,
            shared_helper_before_failure,
        )
    finally:
        _evict_module_tree(package_name)


def test_failed_memory_entry_point_preserves_normal_import_sibling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed public load cannot erase an ordinary importlib sibling load."""
    import plugins.memory as memory_plugins

    package_name = "ful103_normal_import_sibling_package"
    package_before, marker_before, probe = _prepare_memory_entry_point_sibling_race(
        tmp_path,
        monkeypatch,
        memory_plugins,
        package_name=package_name,
    )
    fail_started = getattr(probe, "fail_started")
    release_failure = getattr(probe, "release_failure")
    results: dict[str, object] = {}
    errors: list[str] = []

    def load_failing_sibling() -> None:
        try:
            results["failing"] = memory_plugins.load_memory_provider(
                "failing-sibling",
                register_skills=False,
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(f"failing: {exc!r}")

    def import_successful_sibling() -> None:
        try:
            module = importlib.import_module(f"{package_name}.good")
            results["successful_module"] = module
            results["successful_value"] = module.VALUE
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(f"successful: {exc!r}")

    failing_thread = threading.Thread(target=load_failing_sibling, daemon=True)
    successful_thread = threading.Thread(target=import_successful_sibling, daemon=True)
    try:
        failing_thread.start()
        assert fail_started.wait(timeout=2), "failing entry point missed its barrier"
        (tmp_path / package_name / "shared_helper.py").write_text(
            "VALUE = 'shared-good'\n",
            encoding="utf-8",
        )
        importlib.invalidate_caches()
        successful_thread.start()
        successful_thread.join(timeout=3)
        successful_finished_before_failure = not successful_thread.is_alive()
        good_module_before_failure = sys.modules.get(f"{package_name}.good")
        good_helper_before_failure = sys.modules.get(f"{package_name}.good_helper")
        shared_helper_before_failure = sys.modules.get(f"{package_name}.shared_helper")
    finally:
        release_failure.set()
        failing_thread.join(timeout=3)
        successful_thread.join(timeout=3)

    try:
        assert successful_finished_before_failure
        assert not failing_thread.is_alive()
        assert not successful_thread.is_alive()
        assert errors == []
        assert results["failing"] is None
        assert results["successful_value"] == "good"
        assert results["successful_module"] is good_module_before_failure
        assert isinstance(good_module_before_failure, types.ModuleType)
        assert isinstance(good_helper_before_failure, types.ModuleType)
        assert isinstance(shared_helper_before_failure, types.ModuleType)
        _assert_successful_sibling_survived_failure(
            package_name,
            package_before,
            marker_before,
            good_module_before_failure,
            good_helper_before_failure,
            shared_helper_before_failure,
        )
    finally:
        _evict_module_tree(package_name)


@pytest.mark.parametrize("consumer", ["public", "ordinary", "reload", "finder-failure"])
def test_failed_entry_point_retains_consumed_sibling_without_managed_retry(
    consumer, tmp_path, monkeypatch
):
    import plugins.memory as memory_plugins

    package = f"ful103_cached_{consumer.replace('-', '_')}"
    package_dir = tmp_path / package
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("MARKER = object()\n", encoding="utf-8")
    good_file = package_dir / "good.py"
    good_source = _module_source(
        "memory", prelude="VALUE = 'good'\n", name_expression="VALUE"
    )
    good_file.write_text(good_source, encoding="utf-8")
    probe = types.ModuleType(f"_{package}_probe")
    probe.started = threading.Event()
    probe.release = threading.Event()
    monkeypatch.setitem(sys.modules, probe.__name__, probe)
    bad_file = package_dir / "bad.py"
    bad_file.write_text(
        f"from . import good\nimport {probe.__name__} as probe\n"
        "probe.started.set()\nassert probe.release.wait(5)\n"
        "raise RuntimeError('outer failure')\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    parent = importlib.import_module(package)
    marker = parent.MARKER
    bad_ep = importlib.metadata.EntryPoint(
        name="bad",
        value=f"{package}.bad:register",
        group=memory_plugins.ENTRY_POINTS_GROUP,
    )
    good_ep = importlib.metadata.EntryPoint(
        name="good",
        value=f"{package}.good:SyntheticMemory",
        group=memory_plugins.ENTRY_POINTS_GROUP,
    )
    _install_memory_entry_points(monkeypatch, memory_plugins, bad_ep, good_ep)
    results = []
    worker = threading.Thread(
        target=lambda: results.append(
            memory_plugins.load_memory_provider("bad", register_skills=False)
        ),
        daemon=True,
    )
    worker.start()
    try:
        assert probe.started.wait(3)
        good = sys.modules[f"{package}.good"]
        if consumer == "public":
            provider = memory_plugins.load_memory_provider(
                "good", register_skills=False
            )
            assert provider is not None
            assert provider.name == "good"
        else:
            assert importlib.import_module(f"{package}.good") is good
        if consumer == "reload":
            good_file.write_text(
                "VALUE = 'corrupted'\nraise RuntimeError('reload failure')\n",
                encoding="utf-8",
            )
            importlib.invalidate_caches()
            with pytest.raises(RuntimeError, match="reload failure"):
                importlib.reload(good)
        elif consumer == "finder-failure":

            class FailingFinder:
                def find_spec(self, fullname, path=None, target=None):
                    if fullname == good.__name__ and target is good:
                        good.VALUE = "corrupted"
                        raise RuntimeError("finder failure")

            finder = FailingFinder()
            # After the source finder, before PathFinder: a downstream fault.
            sys.meta_path.insert(1, finder)
            try:
                with pytest.raises(RuntimeError, match="finder failure"):
                    importlib.reload(good)
            finally:
                sys.meta_path.remove(finder)
    finally:
        probe.release.set()
        worker.join(5)
    try:
        assert not worker.is_alive()
        assert results == [None]
        assert sys.modules[package] is parent
        assert parent.MARKER is marker
        assert sys.modules.get(f"{package}.good") is good
        assert parent.good is good
        good_file.write_text(good_source, encoding="utf-8")
        bad_file.write_text(good_source, encoding="utf-8")
        importlib.invalidate_caches()
        assert memory_plugins.load_memory_provider("bad", register_skills=False) is None
        assert (
            memory_plugins.load_memory_provider("good", register_skills=False) is None
        )
        _evict_module_tree(package)
        assert (
            memory_plugins.load_memory_provider("good", register_skills=False) is None
        )
    finally:
        _evict_module_tree(package)


@pytest.mark.parametrize("attribute", ["dynamic", "holder.dynamic"])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("package", [False, True])
def test_entry_point_dynamic_attributes_run_outside_all_managed_locks(
    attribute, nested, package, tmp_path, monkeypatch
):
    import plugins.memory as memory_plugins
    from plugins._loader import module_load_transaction

    module_name = (
        f"ful103_dynamic_attribute_{package}_{nested}_{attribute.replace('.', '_')}"
    )
    probe = types.ModuleType("_ful103_dynamic_probe")
    probe.called = False
    probe.blocked = False
    probe.worker = None
    probe.result = None
    monkeypatch.setitem(sys.modules, probe.__name__, probe)
    source_file = tmp_path / f"{module_name}.py"
    if package:
        package_dir = tmp_path / module_name
        package_dir.mkdir()
        source_file = package_dir / "__init__.py"
    source_file.write_text(
        _module_source("memory") + "import _ful103_dynamic_probe as probe\n"
        "def __getattr__(name):\n"
        "    if name != 'dynamic': raise AttributeError(name)\n"
        "    probe.resolve()\n"
        "    return SyntheticMemory\n"
        "class Holder:\n"
        "    @property\n"
        "    def dynamic(self):\n"
        "        probe.resolve()\n"
        "        return SyntheticMemory\n"
        "holder = Holder()\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    _install_memory_entry_point(
        monkeypatch, memory_plugins, name="dynamic", value=f"{module_name}:{attribute}"
    )
    load = lambda: memory_plugins.load_memory_provider("dynamic", register_skills=False)

    def in_worker():
        probe.result = load()

    def resolve():
        if probe.called:
            return
        probe.called = True
        probe.worker = threading.Thread(target=in_worker, daemon=True)
        probe.worker.start()
        probe.worker.join(3)
        probe.blocked = probe.worker.is_alive()

    probe.resolve = resolve
    try:
        if nested:
            with module_load_transaction("ful103_outer_attribute"):
                result = load()
            assert result is None
            assert probe.called is False
            result = load()
        else:
            result = load()
        if probe.worker is not None:
            probe.worker.join(5)
        assert probe.blocked is False
        assert result is not None
        assert probe.result is not None
        assert probe.worker is not None and not probe.worker.is_alive()
    finally:
        if probe.worker is not None:
            probe.worker.join(5)
        _evict_module_tree(module_name)


@pytest.mark.parametrize("case", _LOADER_CASES)
def test_entry_point_poison_stops_bundled_aliases_not_other_providers(
    case, tmp_path, monkeypatch
):
    from plugins._loader import import_entry_point

    family = importlib.import_module(case.loader_module)
    monkeypatch.setattr(family, _BUNDLED_ROOT_TARGETS[case.name], tmp_path)
    monkeypatch.setattr(family, "__path__", [str(tmp_path)])
    slug = f"badbundled{case.name}"
    good_slug = f"goodbundled{case.name}"
    bad = _write_plugin(
        tmp_path,
        slug,
        _module_source(case.name) + "raise RuntimeError('failed bundled')\n",
    )
    _write_plugin(tmp_path, good_slug, _module_source(case.name))
    name = f"{_BUNDLED_MODULE_PREFIXES[case.name]}.{slug}"
    ep = importlib.metadata.EntryPoint(
        name="bad", value=f"{name}:register", group="test"
    )
    try:
        with pytest.raises(RuntimeError, match="failed bundled"):
            import_entry_point(ep)
        bad.write_text(_module_source(case.name), encoding="utf-8")
        importlib.invalidate_caches()
        assert getattr(family, case.load_function)(slug) is None
        other = getattr(family, case.load_function)(good_slug)
        assert other is not None, "entry-point failure poisoned an unrelated provider"
        assert other.name == "loaded"
        with pytest.raises(ImportError, match="fresh process"):
            import_entry_point(ep)
    finally:
        _evict_module_tree(name)
        _evict_module_tree(f"{_BUNDLED_MODULE_PREFIXES[case.name]}.{good_slug}")


def test_nested_cli_discovery_refuses_dynamic_attribute_callbacks(
    tmp_path, monkeypatch
):
    import plugins.memory as memory
    from plugins._loader import module_load_transaction

    probe = types.ModuleType("_ful103_cli_attribute_probe")
    probe.calls = 0
    monkeypatch.setitem(sys.modules, probe.__name__, probe)
    init_file = _write_plugin(tmp_path, "nestedcli", "# MemoryProvider\n")
    (init_file.parent / "cli.py").write_text(
        "import _ful103_cli_attribute_probe as probe\n"
        "def __getattr__(name):\n"
        "    if name != 'register_cli': raise AttributeError(name)\n"
        "    probe.calls += 1\n"
        "    return lambda parser: None\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(memory, "_get_user_plugins_dir", lambda: tmp_path)
    monkeypatch.setattr(memory, "_get_active_memory_provider", lambda: "nestedcli")
    with module_load_transaction("ful103_outer_cli"):
        commands = memory.discover_plugin_cli_commands()
    assert commands == []
    assert probe.calls == 0
    commands = memory.discover_plugin_cli_commands()
    assert len(commands) == 1
    assert callable(commands[0]["setup_fn"])
    assert probe.calls == 1


@pytest.mark.parametrize("failure_type", [KeyboardInterrupt, SystemExit, ValueError])
@pytest.mark.parametrize("predicate", ["module_matches_path", "module_is_initializing"])
def test_post_readiness_exception_denies_confirmed_waiter(
    failure_type, predicate, tmp_path, monkeypatch
):
    """Escaping validation faults poison before the exact target lock is released."""
    from importlib import _bootstrap

    import plugins._loader as shared
    import plugins.memory as memory

    init_file = _write_plugin(
        tmp_path,
        "postreadiness",
        _module_source("memory", prelude="from . import helper\n"),
        helper_value="retained",
    )
    monkeypatch.setattr(memory, "_get_user_plugins_dir", lambda: tmp_path)
    name = memory._provider_module_name(init_file.parent)
    parent_name, child = name.rsplit(".", 1)
    shared.ensure_namespace_package(parent_name, tmp_path)
    parent = sys.modules[parent_name]
    entered = threading.Event()
    release = threading.Event()
    failure = failure_type("post-readiness failure")
    original_check = getattr(shared, predicate)

    def check(module, *args):
        if module.__name__ == name and not entered.is_set():
            assert module.__spec__._initializing is False
            entered.set()
            assert release.wait(5), "validation barrier timed out"
            raise failure
        return original_check(module, *args)

    monkeypatch.setattr(shared, predicate, check)
    results = {}
    errors = {}
    stacks = {}

    def run(label):
        try:
            if label == "first":
                results[label] = shared.import_module_from_path(name, init_file)
            else:
                results[label] = memory.load_memory_provider(
                    init_file.parent.name, register_skills=False
                )
        except BaseException as exc:
            errors[label] = exc
        finally:
            stacks[label] = tuple(getattr(shared._STATE.local, "stack", ()))

    first = threading.Thread(target=run, args=("first",), daemon=True)
    waiter = threading.Thread(target=run, args=("waiter",), daemon=True)
    lock = _bootstrap._get_module_lock(name)
    first.start()
    try:
        assert entered.wait(3), "post-import validation not reached"
        module = sys.modules[name]
        helper = importlib.import_module(f"{name}.helper")
        assert parent.__dict__[child] is module
        assert module.helper is helper
        waiter.start()
        deadline = time.monotonic() + 3
        while not lock.waiters and time.monotonic() < deadline:
            time.sleep(0.005)
        assert lock.waiters, "same-target waiter never reached the module lock"
    finally:
        release.set()
        first.join(5)
        if waiter.ident is not None:
            waiter.join(5)
    try:
        assert not first.is_alive() and not waiter.is_alive()
        assert errors == {"first": failure}, errors
        assert stacks == {"first": (), "waiter": ()}
        assert name not in shared._SOURCE_MODULE_PATHS
        assert lock.owner is None and lock.count == 0
        assert sys.modules[parent_name] is parent
        assert sys.modules[name] is module
        assert sys.modules[f"{name}.helper"] is helper
        assert parent.__dict__[child] is module
        assert module.helper is helper
        assert results == {"waiter": None}, "waiter reused failed validation's module"
        with pytest.raises(ImportError, match="fresh process"):
            shared.import_module_from_path(name, init_file)
        assert (
            memory.load_memory_provider(init_file.parent.name, register_skills=False)
            is None
        )
    finally:
        _evict_module_tree(name)


@pytest.mark.parametrize("refusal", ["wrong-origin", "initializing"])
def test_preflight_refusal_does_not_poison_healthy_cached_module(
    refusal, tmp_path, monkeypatch
):
    import plugins._loader as shared
    import plugins.memory as memory

    init_file = _write_plugin(tmp_path, "preflightcontrol", _module_source("memory"))
    monkeypatch.setattr(memory, "_get_user_plugins_dir", lambda: tmp_path)
    name = memory._provider_module_name(init_file.parent)

    def load():
        return memory.load_memory_provider(init_file.parent.name, register_skills=False)

    try:
        first = load()
        assert first is not None
        module = sys.modules[name]
        original_file = module.__file__
        if refusal == "wrong-origin":
            module.__file__ = str(tmp_path / "other.py")
        else:
            module.__spec__._initializing = True
        assert load() is None
        assert sys.modules[name] is module
        assert name not in shared._STATE.failed_domains
        module.__file__ = original_file
        module.__spec__._initializing = False
        second = load()
        assert second is not None and second is not first
        assert sys.modules[name] is module
        assert name not in shared._SOURCE_MODULE_PATHS
        shared.ensure_plugin_callbacks_allowed()
    finally:
        _evict_module_tree(name)


def test_invalid_origin_after_execution_is_retained_and_poisoned(tmp_path, monkeypatch):
    import plugins.memory as memory

    slug = "originchanged"
    init_file = _write_plugin(
        tmp_path, slug, _module_source("memory") + "__file__ = 'changed-origin'\n"
    )
    monkeypatch.setattr(memory, "_get_user_plugins_dir", lambda: tmp_path)
    name = memory._provider_module_name(init_file.parent)
    try:
        assert memory.load_memory_provider(slug, register_skills=False) is None
        module = sys.modules.get(name)
        assert module is not None, "readiness rejection erased a published module"
        parent_name, child = name.rsplit(".", 1)
        assert getattr(sys.modules[parent_name], child) is module
        init_file.write_text(_module_source("memory"), encoding="utf-8")
        _evict_module_tree(name)
        assert memory.load_memory_provider(slug, register_skills=False) is None
    finally:
        _evict_module_tree(name)


@pytest.mark.parametrize("case", _EXTERNAL_LOADER_CASES)
def test_poisoned_profile_denies_entrypoint_aliases_but_other_profile_loads(
    case, tmp_path, monkeypatch
):
    from plugins._loader import import_entry_point

    family = importlib.import_module(case.loader_module)
    slug = "profilepoison"
    first_root, second_root = tmp_path / "first", tmp_path / "second"
    init_file = _write_plugin(
        first_root,
        slug,
        _module_source(case.name, prelude="from . import helper\n")
        + "raise RuntimeError('failed profile')\n",
        helper_value="retained",
    )
    _write_plugin(second_root, slug, _module_source(case.name))
    name = family._provider_module_name(init_file.parent)
    load = getattr(family, case.load_function)
    try:
        _set_plugins_root(monkeypatch, family, case, first_root)
        assert load(slug) is None
        init_file.write_text(_module_source(case.name), encoding="utf-8")
        importlib.invalidate_caches()
        for target in (f"{name}:register", f"{name}.helper:VALUE"):
            ep = importlib.metadata.EntryPoint(name="alias", value=target, group="test")
            with pytest.raises(ImportError, match="fresh process"):
                import_entry_point(ep)
        _evict_module_tree(name)
        assert load(slug) is None
        _set_plugins_root(monkeypatch, family, case, second_root)
        other = load(slug)
        assert other is not None
        assert other.name == "loaded"
        _evict_module_tree(type(other).__module__)
    finally:
        _evict_module_tree(name)


def test_public_package_entrypoint_cannot_bypass_failed_submodule_domain(
    tmp_path, monkeypatch
):
    import plugins.memory as memory

    package = "ful103_public_package_alias"
    package_dir = tmp_path / package
    package_dir.mkdir()
    (package_dir / "helper.py").write_text("VALUE = 'healthy'\n", encoding="utf-8")
    (package_dir / "__init__.py").write_text(
        _module_source(
            "memory",
            prelude=f"from {package} import helper\n",
            name_expression="helper.VALUE",
        ),
        encoding="utf-8",
    )
    (package_dir / "bad.py").write_text(
        "from . import helper\nhelper.VALUE = 'affected'\nraise RuntimeError('package failed')\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    bad = importlib.metadata.EntryPoint(
        name="bad-package",
        value=f"{package}.bad:register",
        group=memory.ENTRY_POINTS_GROUP,
    )
    alias = importlib.metadata.EntryPoint(
        name="package-alias",
        value=f"{package}:register",
        group=memory.ENTRY_POINTS_GROUP,
    )
    _install_memory_entry_points(monkeypatch, memory, bad, alias)
    helper = importlib.import_module(f"{package}.helper")
    try:
        assert memory.find_provider_dir(alias.name) == package_dir
        assert memory.load_memory_provider(bad.name, register_skills=False) is None
        assert sys.modules.get(f"{package}.helper") is helper
        assert helper.VALUE == "affected"
        assert memory.load_memory_provider(alias.name, register_skills=False) is None
        assert sys.modules[f"{package}.helper"] is helper
    finally:
        _evict_module_tree(package)
        _evict_module_tree(memory._provider_module_name(package_dir))


def test_same_thread_recursive_memory_entry_point_rejects_partial_module(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A recursive bare entry point cannot construct from its partial module."""
    import plugins.memory as memory_plugins

    module_name = "ful103_recursive_entrypoint"
    provider_name = "recursive-entrypoint"
    probe_name = "_ful103_recursive_entrypoint_probe"
    probe = types.ModuleType(probe_name)
    probe.recursing = False
    probe.nested = "not-called"
    monkeypatch.setitem(sys.modules, probe_name, probe)
    (tmp_path / f"{module_name}.py").write_text(
        _module_source(
            "memory",
            instance_init=(
                "    def __init__(self):\n"
                "        self.ready_seen = globals().get('READY', 'absent')\n"
            ),
            name_expression="self.ready_seen",
            postlude=(
                f"import {probe_name} as probe\n"
                "def make_provider(): return SyntheticMemory()\n"
                "if not probe.recursing:\n"
                "    probe.recursing = True\n"
                "    probe.nested = probe.load()\n"
                "READY = 'ready'\n"
            ),
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    _install_memory_entry_point(
        monkeypatch,
        memory_plugins,
        name=provider_name,
        value=f"{module_name}:make_provider",
    )
    probe.load = lambda: memory_plugins.load_memory_provider(
        provider_name,
        register_skills=False,
    )

    try:
        provider = memory_plugins.load_memory_provider(
            provider_name,
            register_skills=False,
        )
        nested = probe.nested
    finally:
        sys.modules.pop(module_name, None)

    assert provider is not None
    assert provider.ready_seen == "ready"
    assert nested is None


def test_nested_cross_family_load_skips_callback_under_outer_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An inner register callback cannot wait on a worker needing the outer lock."""
    import plugins.cron_providers as cron_plugins
    import plugins.memory as memory_plugins

    memory_root = tmp_path / "memory"
    cron_root = tmp_path / "cron"
    memory_slug = "outercallbackmemory"
    cron_slug = "innercallbackcron"
    probe_name = "_ful103_outer_transaction_callback_probe"
    probe = types.ModuleType(probe_name)
    probe.inner_provider = "not-called"
    probe.callback_calls = 0
    probe.worker_blocked = False
    probe.worker = None
    probe.worker_provider = None
    probe.worker_done = threading.Event()
    monkeypatch.setitem(sys.modules, probe_name, probe)

    memory_init = _write_plugin(
        memory_root,
        memory_slug,
        _module_source(
            "memory",
            prelude=(
                f"import {probe_name} as probe\n"
                "probe.inner_provider = probe.load_cron()\n"
            ),
            name_expression=repr(memory_slug),
        ),
    )
    cron_init = _write_plugin(
        cron_root,
        cron_slug,
        _module_source(
            "cron",
            name_expression=repr(cron_slug),
            postlude=(
                "import threading\n"
                f"import {probe_name} as probe\n"
                "_original_register = register\n"
                "def register(ctx):\n"
                "    probe.callback_calls += 1\n"
                "    probe.worker = threading.Thread(\n"
                "        target=probe.reload_outer, daemon=True\n"
                "    )\n"
                "    probe.worker.start()\n"
                "    if not probe.worker_done.wait(timeout=0.5):\n"
                "        probe.worker_blocked = True\n"
                "    _original_register(ctx)\n"
            ),
        ),
    )
    monkeypatch.setattr(memory_plugins, "_get_user_plugins_dir", lambda: memory_root)
    monkeypatch.setattr(cron_plugins, "_get_user_plugins_dir", lambda: cron_root)
    memory_module = memory_plugins._provider_module_name(memory_init.parent)
    cron_module = cron_plugins._provider_module_name(cron_init.parent)

    def reload_outer() -> None:
        try:
            probe.worker_provider = memory_plugins.load_memory_provider(
                memory_slug,
                register_skills=False,
            )
        finally:
            probe.worker_done.set()

    probe.load_cron = lambda: cron_plugins.load_cron_scheduler(cron_slug)
    probe.reload_outer = reload_outer
    try:
        provider = memory_plugins.load_memory_provider(
            memory_slug,
            register_skills=False,
        )
        if probe.worker is not None:
            assert probe.worker_done.wait(timeout=3), "callback worker did not finish"
            probe.worker.join(timeout=1)
    finally:
        _evict_module_tree(memory_module)
        _evict_module_tree(cron_module)

    assert provider is not None
    assert probe.inner_provider is None
    assert probe.callback_calls == 0
    assert probe.worker_blocked is False


@pytest.mark.parametrize("case", _EXTERNAL_LOADER_CASES)
@pytest.mark.parametrize(
    "slug",
    [
        pytest.param("dotted.name", id="dotted"),
        pytest.param("hyphen-name", id="hyphen-control"),
        pytest.param("n" * 240, id="long-control"),
    ],
)
def test_public_external_provider_load_accepts_discovered_directory_names(
    case: _LoaderCase,
    slug: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every publicly discovered external directory name remains loadable."""
    loader_module = importlib.import_module(case.loader_module)
    init_file = _write_plugin(
        tmp_path,
        slug,
        _module_source(case.name, name_expression=repr(slug)),
    )
    _set_plugins_root(monkeypatch, loader_module, case, tmp_path)
    if case.name == "memory":
        discovered = slug in loader_module.list_memory_provider_names()
        provider = loader_module.load_memory_provider(slug, register_skills=False)
    else:
        discovered = slug in {
            name
            for name, _description, _available in loader_module.discover_cron_schedulers()
        }
        provider = loader_module.load_cron_scheduler(slug)
    module_name = loader_module._provider_module_name(init_file.parent)

    try:
        assert discovered is True
        assert provider is not None
        assert getattr(provider, "name") == slug
    finally:
        _evict_module_tree(module_name)
