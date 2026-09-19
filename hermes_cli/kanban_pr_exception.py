"""Trusted local-owner exceptions, never repository-required CI acceptance.

Config is authority under Hermes' same-UID trust model, not a signed approval.
Only the shared completion boundary calls this; worker metadata is not policy.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

from hermes_cli.config_effective import load_user_config_effective
from hermes_cli.kanban_pr_acceptance import _PR, _api, _classify
from hermes_constants import hermes_home_key

_FIELDS = {"board", "task_id", "pr_url", "head_sha", "tree_sha",
           "approval_reason", "approval_reference", "evidence_receipt"}
_SHA = re.compile(r"[0-9a-f]{40}")


def load_exception(conn, task_id: str, contract: str):
    """Select exact policy in the current profile, bound to the actual database.

    A noncanonical/custom DB path has no board-slug identity and cannot use an
    exception. Never use a process's active-board hint to label another DB.
    """
    from hermes_cli import kanban_db as kb

    entries = load_user_config_effective(fail_closed=True).get("kanban", {}).get("pr_acceptance_exceptions", [])
    if not isinstance(entries, list):
        raise ValueError("Owner exceptions must be a list")
    selected = None
    seen = set()
    for entry in entries:
        if (not isinstance(entry, dict) or set(entry) != _FIELDS or
                any(not isinstance(v, str) or not v.strip() or any(ord(c) < 32 for c in v)
                    for v in entry.values())):
            raise ValueError("Incomplete owner exception")
        if (not kb._BOARD_SLUG_RE.fullmatch(entry["board"]) or
                not re.fullmatch(r"t_[0-9a-f]{8}", entry["task_id"]) or
                not _PR.fullmatch(entry["pr_url"]) or
                not _SHA.fullmatch(entry["head_sha"]) or not _SHA.fullmatch(entry["tree_sha"])):
            raise ValueError("Invalid owner exception identity")
        key = (entry["board"], entry["task_id"])
        if key in seen:
            raise ValueError("Ambiguous owner exception")
        seen.add(key)
        if entry["task_id"] != task_id or entry["pr_url"] != contract:
            continue
        path = next((row[2] for row in conn.execute("PRAGMA database_list") if row[1] == "main"), "")
        expected = (kb.kanban_home() / "kanban.db" if entry["board"] == "default"
                    else kb.board_dir(entry["board"]) / "kanban.db")
        if path and Path(path).resolve() == expected.resolve():
            selected = (hermes_home_key(), entry)
    return selected


def collect_owner_exception(entry: dict) -> dict:
    receipt = {"ok": False, "classification": "infra", "scope": entry,
               "pr_url": entry["pr_url"], "head_sha": entry["head_sha"],
               "tree_sha": entry["tree_sha"], "checks": [],
               "remote_required_ci_accepted": False,
               "recovery": "Verify the exact owner policy and release evidence, then retry completion."}
    try:
        match = _PR.fullmatch(entry["pr_url"])
        if match is None:
            raise ValueError("Invalid exact PR URL")
        repo, number = match[1], match[2]
        sha = entry["head_sha"]
        pr = _api(f"repos/{repo}/pulls/{number}")
        merge = pr["merge_commit_sha"]
        if (pr["merged"] is not True or pr["state"] != "closed" or pr["head"]["sha"] != sha
                or not isinstance(merge, str) or not _SHA.fullmatch(merge)):
            receipt.update(classification="stale", detail="PR is not the approved merged head.")
            return receipt
        for commit in (sha, merge):
            obj = _api(f"repos/{repo}/git/commits/{commit}")
            if obj["sha"] != commit or obj["tree"]["sha"] != entry["tree_sha"]:
                receipt.update(classification="stale", detail="Head or merge tree differs from owner approval.")
                return receipt
        receipt["merge_sha"] = merge
        pages = _api(f"repos/{repo}/commits/{sha}/check-runs?per_page=100&filter=latest", paginate=True)
        if any(not isinstance(page, dict) or not isinstance(page.get("check_runs"), list)
               for page in pages):
            raise ValueError("Invalid check-run pagination shape")
        runs = [run for page in pages for run in page["check_runs"]]
        if len({r["id"] for r in runs}) != pages[0]["total_count"]:
            raise ValueError("Incomplete check-run pagination")
        status_pages = _api(f"repos/{repo}/commits/{sha}/statuses?per_page=100", paginate=True)
        if any(not isinstance(page, list) for page in status_pages):
            raise ValueError("Invalid status pagination shape")
        statuses = [s for page in status_pages for s in page]
        latest = {}
        for status in statuses:
            context = status["context"]
            if context not in latest or status["id"] > latest[context]["id"]:
                latest[context] = status
        for check in runs + [{**s, "sha": sha} for s in latest.values()]:
            is_run = "conclusion" in check
            outcome = check["conclusion"] if is_run else check["state"]
            classification = _classify(check, sha, outcome, is_run)
            receipt["checks"].append({"id": check["id"], "name": check["name"] if is_run else check["context"],
                                      "head_sha": check.get("head_sha", check.get("sha")),
                                      "url": check.get("html_url") or check.get("target_url"),
                                      "classification": classification, "conclusion": outcome})
        current = _api(f"repos/{repo}/pulls/{number}")
        if (current["head"]["sha"] != sha or current["merge_commit_sha"] != merge
                or current["merged"] is not True or current["state"] != "closed"
                or current["base"]["ref"] != pr["base"]["ref"]):
            receipt.update(classification="stale", detail="Merged PR changed during collection.")
            return receipt
        failure = next((c["classification"] for c in receipt["checks"] if c["classification"] != "success"), None)
        if failure:
            receipt.update(classification=failure, detail="Owner exception does not waive non-success exact-head checks.")
            return receipt
        receipt.update(ok=True, classification="owner_exception",
                       detail="Owner-authorized local release evidence; repository-required CI was NOT accepted.")
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError, IndexError):
        receipt.update(classification="infra", detail="Exact merged release evidence unavailable or incomplete.")
    return receipt
