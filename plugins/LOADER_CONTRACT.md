# Managed provider imports

`plugins._loader` is the import/readiness boundary used by memory providers
(directory and distribution entry points), cron providers, context engines and
lightweight memory CLI discovery. It is cooperative Python plugin-loading
reliability, **not a sandbox**.

## Failure means a new interpreter

Eligibility has two states: eligible and poisoned. Poison is monotonic for the
interpreter lifetime. Importlib owns the separate importing/ready lifecycle and
module cache. A managed import whose execution or post-import readiness check
fails with any `BaseException` poisons its domain before releasing the exact
module lock. Entry-point attribute-resolution failures also poison that domain,
but attribute resolution runs **after** the lock is released. The original
exception propagates; later calls raise an `ImportError` explaining that a fresh
process is required. Public provider APIs retain their existing `None`/empty
result handling for ordinary exceptions. Registration and provider-constructor
errors retain their existing compatibility/fallback behavior and do not themselves
poison a successfully imported module; unlike entry-point attribute-resolution
failures, they occur after the managed import/readiness boundary has succeeded.

Correcting source, evicting cached modules, or reloading/removing/reimporting the
loader does not clear poison. There is no production reset API. Start a new
process with corrected source; restarting a running service is an operator
decision, not a loader action.

Hermes does not erase or restore failed module trees or parent bindings.
Importlib may remove its own failed target, while successfully published
helpers remain cached. Objects already returned to public or ordinary importlib
consumers are not revoked. Missing directory roots with retained descendants
are refused and poisoned rather than silently repaired. Preflight origin or
initializing-module mismatches are refused without rewriting the cache.

## Domain and alias rules

| Managed target | Poison domain |
| --- | --- |
| Directory provider | Its module-name root and descendants |
| Lightweight CLI file | Its isolated, code-less synthetic namespace |
| Ordinary distribution entry point | Its top-level package (or bare module) |
| Entry point inside `plugins.memory.X`, `plugins.cron_providers.X`, `plugins.context_engine.X` | The provider root through `X`, not all of `plugins` |

A request is refused if its domain contains, equals, or is contained by any
poisoned prefix. Consequently another attribute, a cached helper/sibling in the
same entry-point package, an entry point naming a poisoned directory package,
or a directory load after an entry-point failure in that package cannot bypass
the check. The scope is derived by the import helper, not passed in by its
caller. Full memory-provider entry points retain their canonical module and
attribute route even when their module is a package: the entry-point package
directory fallback in `find_provider_dir` is for CLI/config assets only, not
provider execution. Native bundled/user/project directory precedence is unchanged.
Different path-scoped profile namespaces and unrelated provider roots remain eligible. A successful in-flight sibling admitted before a different
target fails may finish; this is not an all-package atomic transaction.

The identity here is **cached package state, not equivalent source code**.
Separate synthetic namespaces deliberately have independent package caches
(for example memory, cron and CLI namespaces, or a directory import versus an
ordinary installed-package import). Absolute imports, references to another
package's objects, deliberate `sys.modules` aliases, and arbitrary shared
third-party dependencies are not an isolation boundary this loader can enforce.
A package failure may therefore require broader application recovery than the
named domain. It never implies transactional rollback of Python side effects.

## Locks, callbacks and lifetime

The exact CPython module lock is held while checking cached readiness,
registering an exact source path, importing, validating and publishing poison.
Ordinary importlib callers participate in the same module lock. Hermes uses
CPython-private `importlib._bootstrap._get_module_lock`, private deadlock
exception types and `ModuleSpec._initializing`; other interpreters are not
validated by these tests.

Registration, provider constructors, entry-point module `__getattr__` /
descriptors, and CLI attribute extraction run outside managed import locks.
Extraction under any outer managed import is refused before invoking those
callbacks. This includes nested entry points after loader reload/reimport.
Provider module bodies still execute under Python import locks: arbitrary module
bodies joining other threads can deadlock. The loader does not promise universal
deadlock prevention or repair unsafe external `reload()` / cache mutation.

`plugins._loader_state` publishes one private interpreter-lifetime object on
`sys`, under the short import-publication lock. That object owns the thread-local
stack, poison set, temporary source map and finder identity. Loader reload and
removal/reimport reuse it. Transaction and mapping finalizers capture their
state, stack and lock; `finally` releases them on `BaseException`. The state
mutex only protects small state operations, not plugin execution or callbacks.

The finder has one job: resolve a temporarily registered **exact** module name
to its canonical source file, including external dotted/hyphenated/long directory
names whose path-scoped synthetic names differ from filesystem spelling.
Descendants use normal path resolution. It does not observe ownership, infer
successful publication, intercept cache hits, or monkeypatch importlib.

## Regression suites

- `tests/plugins/test_dynamic_loader_transactions.py`: public family loads,
  normal-import readiness, retained cache consumers, poisoned aliases, profile
  isolation, corrected-source refusal/new-process success and callback liveness.
- `tests/plugins/test_loader_process_lifetime.py`: bounded isolated processes for
  active loader reload/reimport, waiters, `BaseException`, and entry-point dynamic
  attribute failure followed by corrected source in a new interpreter.

All fixtures use temporary roots and synthetic providers. Process poisoning is
not reset between tests; tests use independent module domains or subprocesses.
