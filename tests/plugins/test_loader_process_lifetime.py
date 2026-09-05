"""Bounded, isolated interpreter-lifetime regressions for the shared loader."""

from pathlib import Path
import subprocess
import sys

import pytest


_REPO = Path(__file__).resolve().parents[2]
_BOOTSTRAP = """
import sys
import os
from pathlib import Path
repo = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(repo))
os.environ['HERMES_HOME'] = str(Path(sys.argv[2]) / 'home')
import plugins._loader as loader
assert Path(loader.__file__).resolve() == repo / 'plugins' / '_loader.py'
"""


def _run(script, tmp_path, *args):
    child = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            _BOOTSTRAP + script,
            str(_REPO),
            str(tmp_path),
            *args,
        ],
        capture_output=True,
        text=True,
        timeout=25,
    )
    assert child.returncode == 0, child.stdout + child.stderr


@pytest.mark.parametrize(
    "fault",
    [
        "nul",
        "success",
        pytest.param("runtime", marks=pytest.mark.linux_only),
        pytest.param("sigint", marks=pytest.mark.linux_only),
    ],
)
def test_post_readiness_path_fault_requires_fresh_interpreter(fault, tmp_path):
    _run(
        r"""
import importlib
import signal
import subprocess
import plugins.memory as memory
from tests.plugins.loader_test_support import evict_module_tree

root = Path(sys.argv[2])
plugin = root / 'postreadiness'
plugin.mkdir()
source = plugin / '__init__.py'
valid = (
    "from . import helper\n"
    "def register(ctx): ctx.register_memory_provider(helper.VALUE)\n"
)
(plugin / 'helper.py').write_text("VALUE = object()\n", encoding='utf-8')
fault = sys.argv[3]
source.write_text(valid + ("__file__ = 'bad\\0origin'\n" if fault == 'nul' else ''),
                  encoding='utf-8')
memory._get_user_plugins_dir = lambda: root
name = memory._provider_module_name(plugin)
parent_name, child_name = name.rsplit('.', 1)
loader.ensure_namespace_package(parent_name, root)
parent = sys.modules[parent_name]
original_lstat = os.lstat
reached = False

def lstat(path, *args, **kwargs):
    global reached
    frame = sys._getframe(1)
    while frame is not None:
        if (frame.f_code.co_name == 'module_matches_path'
                and frame.f_code.co_filename == loader.__file__):
            assert sys.modules[name].__spec__._initializing is False
            reached = True
            if fault == 'sigint':
                # Only this isolated, throwaway interpreter receives SIGINT.
                os.kill(os.getpid(), signal.SIGINT)
            if fault == 'runtime':
                raise RuntimeError('path resolution fault')
            break
        frame = frame.f_back
    return original_lstat(path, *args, **kwargs)

os.lstat = lstat
caught = None
try:
    loaded = loader.import_module_from_path(name, source)
except BaseException as exc:
    caught = type(exc)
finally:
    os.lstat = original_lstat
expected = {'nul': ValueError, 'runtime': ImportError,
            'sigint': KeyboardInterrupt, 'success': None}[fault]
assert caught is expected, (caught, expected)
if fault in {'runtime', 'sigint'}:
    assert reached, 'actual post-import Path.resolve/lstat not reached'
module = sys.modules[name]
helper = sys.modules[name + '.helper']
assert sys.modules[parent_name] is parent
assert vars(parent)[child_name] is module
assert module.helper is helper
assert not loader._SOURCE_MODULE_PATHS
assert not getattr(loader._STATE.local, 'stack', ())
lock = importlib._bootstrap._get_module_lock(name)
assert lock.owner is None and lock.count == 0

def load():
    return memory.load_memory_provider(plugin.name, register_skills=False)

if fault == 'success':
    assert loaded is module
    assert load() is helper.VALUE
    assert name not in loader._STATE.failed_domains
else:
    assert name in loader._STATE.failed_domains, 'validation failure did not publish poison'
    source.write_text(valid, encoding='utf-8')
    module.__file__ = str(source)
    importlib.invalidate_caches()
    assert load() is None, 'corrected origin bypassed poison'
    assert importlib.reload(module) is module
    assert load() is None, 'ordinary reload bypassed poison'
    assert sys.modules[name] is module
    assert module.helper is helper
    assert vars(parent)[child_name] is module
    evict_module_tree(name)  # Fixture-only eviction; never clears lifetime poison.
    state = loader._STATE
    importlib.reload(loader)
    del sys.modules['plugins._loader']
    current = importlib.import_module('plugins._loader')
    assert current._STATE is state
    try:
        current.import_module_from_path(name, source)
    except ImportError as exc:
        assert 'fresh process' in str(exc)
    else:
        raise AssertionError('cache eviction/loader reimport bypassed poison')
    assert load() is None
    fresh = '''
import sys, os
from pathlib import Path
repo, root = map(Path, sys.argv[1:3])
sys.path.insert(0, str(repo))
os.environ['HERMES_HOME'] = str(root / 'fresh-home')
import plugins._loader as loader
assert Path(loader.__file__).resolve() == repo / 'plugins' / '_loader.py'
import plugins.memory as memory
memory._get_user_plugins_dir = lambda: root
assert memory.load_memory_provider('postreadiness', register_skills=False) is not None
assert not loader._STATE.failed_domains
'''
    result = subprocess.run([sys.executable, '-I', '-c', fresh, str(repo), str(root)],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
""",
        tmp_path,
        fault,
    )


@pytest.mark.parametrize("failure", ["RuntimeError", "SystemExit"])
def test_dynamic_attribute_failure_retains_code_and_requires_new_interpreter(
    failure, tmp_path
):
    _run(
        """
import importlib
from importlib.metadata import EntryPoint
import subprocess
import types
import plugins.memory as memory

root = Path(sys.argv[2])
sys.path.insert(0, str(root))
module_name = 'lifetime_dynamic_failure'
probe = types.ModuleType('_attribute_failure_probe')
sys.modules[probe.__name__] = probe
valid = (
    "from agent.memory_provider import MemoryProvider\\n"
    "class Provider(MemoryProvider):\\n"
    "    name = 'fresh-provider'\\n"
    "    def is_available(self): return True\\n"
    "    def initialize(self, **kw): pass\\n"
    "    def sync_turn(self, *a, **kw): pass\\n"
    "    def get_tool_schemas(self): return []\\n"
    "    def handle_tool_call(self, *a, **kw): return '{}'\\n"
    "def factory(): return Provider()\\n"
)
source = root / (module_name + '.py')
source.write_text(valid + (
    "import sys, _attribute_failure_probe as probe\\n"
    "probe.module = sys.modules[__name__]\\n"
    "def __getattr__(name):\\n"
    "    if name != 'dynamic': raise AttributeError(name)\\n"
    f"    raise {sys.argv[3]}('attribute failure')\\n"
), encoding='utf-8')
ep = EntryPoint(name='dynamic-failure', value=module_name + ':dynamic', group=memory.ENTRY_POINTS_GROUP)
memory._iter_entry_points = lambda: [ep]
def load():
    return memory.load_memory_provider(ep.name, register_skills=False)
if sys.argv[3] == 'SystemExit':
    try:
        load()
    except SystemExit:
        pass
    else:
        raise AssertionError('SystemExit swallowed')
else:
    assert load() is None
assert sys.modules.get(module_name) is probe.module, 'attribute failure erased ready module'
probe.module.dynamic = probe.module.factory
source.write_text(valid + "dynamic = factory\\n", encoding='utf-8')
importlib.invalidate_caches()
assert load() is None, 'attribute correction bypassed poison'
del sys.modules[module_name]
importlib.reload(loader)
del sys.modules['plugins._loader']
current = importlib.import_module('plugins._loader')
try:
    current.import_entry_point(ep)
except ImportError as exc:
    assert 'fresh process' in str(exc)
else:
    raise AssertionError('cache eviction/reimport bypassed attribute failure')
child_script = '''
import sys
from pathlib import Path
repo, root = map(Path, sys.argv[1:3])
sys.path[:0] = [str(repo), str(root)]
import plugins._loader as loader
assert Path(loader.__file__).resolve() == repo / 'plugins' / '_loader.py'
import plugins.memory as memory
from importlib.metadata import EntryPoint
ep = EntryPoint(name='dynamic-failure', value='lifetime_dynamic_failure:dynamic', group=memory.ENTRY_POINTS_GROUP)
memory._iter_entry_points = lambda: [ep]
provider = memory.load_memory_provider(ep.name, register_skills=False)
assert provider is not None
assert provider.name == 'fresh-provider'
'''
child = subprocess.run([sys.executable, '-I', '-c', child_script, str(repo), str(root)],
                       capture_output=True, text=True, timeout=10)
assert child.returncode == 0, child.stdout + child.stderr
""",
        tmp_path,
        failure,
    )


@pytest.mark.parametrize("change", ["reload", "reimport"])
@pytest.mark.parametrize("failure", ["RuntimeError", "SystemExit", "success"])
def test_active_loader_change_preserves_state_and_releases_waiters(
    change, failure, tmp_path
):
    _run(
        """
import importlib
from importlib import _bootstrap
import threading
import time
import types
import plugins.memory as memory
import plugins.cron_providers as cron

root = Path(sys.argv[2]) / 'providers'
root.mkdir()
plugin = root / 'outer'
plugin.mkdir()
inner = root / 'inner'
inner.mkdir()
probe = types.ModuleType('_lifetime_probe')
probe.entered = threading.Event()
probe.release = threading.Event()
probe.callback_calls = 0
probe.nested = 'not called'
sys.modules[probe.__name__] = probe
inner.joinpath('__init__.py').write_text(
    "import _lifetime_probe as probe\\n"
    "def register(ctx):\\n"
    "    probe.callback_calls += 1\\n"
    "    ctx.register_cron_scheduler(object())\\n", encoding='utf-8')
valid = "def register(ctx): ctx.register_memory_provider(object())\\n"
failure = sys.argv[4]
source = (
    "import _lifetime_probe as probe\\n"
    "probe.entered.set()\\nassert probe.release.wait(8)\\n"
    "probe.nested = probe.load_inner()\\n" + valid
)
if failure != 'success':
    source += f"raise {failure}('fixture failure')\\n"
plugin.joinpath('__init__.py').write_text(source, encoding='utf-8')
memory._get_user_plugins_dir = lambda: root
cron._get_user_plugins_dir = lambda: root
probe.load_inner = lambda: cron.load_cron_scheduler('inner')
name = memory._provider_module_name(plugin)
finder = loader._SOURCE_MODULE_FINDER
results = {}
errors = {}
def load(label):
    try:
        results[label] = memory.load_memory_provider('outer', register_skills=False)
    except BaseException as exc:
        errors[label] = type(exc).__name__
threads = [threading.Thread(target=load, args=(label,), daemon=True)
           for label in ('first', 'waiter')]
threads[0].start()
assert probe.entered.wait(3)
threads[1].start()
lock = _bootstrap._get_module_lock(name)
deadline = time.monotonic() + 3
while not lock.waiters and time.monotonic() < deadline:
    time.sleep(0.005)
assert lock.waiters, 'waiter never reached module lock'
try:
    if sys.argv[3] == 'reload':
        current = importlib.reload(loader)
    else:
        del sys.modules['plugins._loader']
        current = importlib.import_module('plugins._loader')
    importlib.reload(cron)
    cron._get_user_plugins_dir = lambda: root
finally:
    probe.release.set()
    for thread in threads:
        thread.join(5)
assert not any(thread.is_alive() for thread in threads), 'loader change stranded waiter'
assert current._SOURCE_MODULE_FINDER is finder, 'source finder identity split'
assert sum(item is finder for item in sys.meta_path) == 1
assert probe.nested is None
assert probe.callback_calls == 0, 'new generation ran callback under outer transaction'
current.ensure_plugin_callbacks_allowed()
assert not current._SOURCE_MODULE_PATHS, 'source map leaked'
if failure == 'success':
    assert errors == {}, errors
    assert all(results.get(label) is not None for label in ('first', 'waiter'))
else:
    if failure == 'SystemExit':
        assert errors == {'first': 'SystemExit'}, errors
    else:
        assert errors == {}, errors
        assert results['first'] is None
    assert results['waiter'] is None
    plugin.joinpath('__init__.py').write_text(valid, encoding='utf-8')
    importlib.invalidate_caches()
    for key in list(sys.modules):
        if key == name or key.startswith(name + '.'):
            del sys.modules[key]
    current = importlib.reload(current)
    del sys.modules['plugins._loader']
    current = importlib.import_module('plugins._loader')
    try:
        current.import_module_from_path(name, plugin / '__init__.py')
    except ImportError as exc:
        assert 'fresh process' in str(exc)
    else:
        raise AssertionError('corrected code bypassed poison after cache eviction/reimport')
assert cron.load_cron_scheduler('inner') is not None
assert probe.callback_calls == 1
""",
        tmp_path,
        change,
        failure,
    )
