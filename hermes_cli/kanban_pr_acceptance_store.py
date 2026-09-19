"""Persist acceptance with the same ownership snapshot as the terminal write."""
from __future__ import annotations

from hermes_cli.kanban_db_connect import write_txn
from hermes_cli.kanban_pr_acceptance import _PR, collect_acceptance


def _snapshot(conn, task_id):
    row = conn.execute("SELECT current_run_id, status, completion_contract FROM tasks WHERE id=?", (task_id,)).fetchone()
    return tuple(row) if row else None


def prepare_acceptance(conn, task_id, expected_run_id, metadata):
    snapshot = _snapshot(conn, task_id)
    if snapshot is None:
        return False
    run_id, status, contract = snapshot
    if not contract or contract == "local-only":
        return None
    if status not in {"running", "ready", "blocked", "review"} or (expected_run_id is not None and run_id != expected_run_id):
        return False
    published_pr = metadata.get("published_pr") if isinstance(metadata, dict) else None
    match = _PR.fullmatch(published_pr) if isinstance(published_pr, str) else None
    # Publication binds once. Retrying cannot replace the task's PR with a green sibling.
    if match and contract == match[1]:
        with write_txn(conn):
            if _snapshot(conn, task_id) != snapshot:
                return False
            conn.execute("UPDATE tasks SET completion_contract=? WHERE id=?", (published_pr, task_id))
        snapshot = (run_id, status, published_pr)
        contract = published_pr
    from hermes_cli.kanban_pr_exception import load_exception, collect_owner_exception
    policy, exception = None, None
    try:
        policy = load_exception(conn, task_id, contract)
    except Exception:
        # Config loader failures must never resurrect a prior approval. Do not
        # persist parser messages: they can contain arbitrary config contents.
        exception = {"ok": False, "classification": "invalid_policy",
                     "detail": "Owner exception configuration is unreadable or invalid.",
                     "recovery": "Correct owner policy and retry."}
    receipt = collect_acceptance(contract, published_pr)
    if not receipt["ok"] and policy is not None:
        if published_pr is not None and published_pr != contract:
            exception = {"ok": False, "classification": "stale",
                         "detail": "Published PR differs from owner-approved contract.",
                         "recovery": "Use the persisted exact PR."}
        else:
            exception = collect_owner_exception(policy[1])
    return snapshot, receipt, exception if not receipt["ok"] else None, policy


def record_acceptance(conn, task_id, acceptance):
    """Called under complete_task's write_txn, before its terminal UPDATE."""
    from hermes_cli.kanban_db import _append_event
    snapshot, receipt, exception, policy = acceptance
    if _snapshot(conn, task_id) != snapshot:
        return False
    _append_event(conn, task_id, "pr_acceptance", receipt, run_id=snapshot[0])
    if exception is not None:
        if exception["ok"]:
            from hermes_cli.kanban_pr_exception import load_exception
            try:
                unchanged = load_exception(conn, task_id, snapshot[2]) == policy
            except Exception:
                unchanged = False
            if not unchanged:
                exception = {**exception, "ok": False, "classification": "stale",
                             "detail": "Owner exception changed during collection; retry with current policy."}
        _append_event(conn, task_id, "pr_owner_exception", exception, run_id=snapshot[0])
        receipt = exception
    if not receipt["ok"]:
        detail = f"PR acceptance {receipt['classification']}: {receipt.get('detail', '')} {receipt['recovery']}"
        conn.execute("UPDATE tasks SET last_failure_error=? WHERE id=?", (detail, task_id))
    return receipt["ok"]
