"""Exact owner policy through real config, gh subprocess transport and SQLite."""
import copy
import json
import os
import sys
import threading
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.config import set_config_value
from hermes_cli.kanban_db_connect import connect

URL = "https://github.com/acme/repo/pull/7"
HEAD, TREE, MERGE = "a" * 40, "b" * 40, "c" * 40


def policy(task):
    return dict(board="default", task_id=task, pr_url=URL, head_sha=HEAD,
                tree_sha=TREE, approval_reason="Owner accepts local release evidence, not required CI",
                approval_reference="owner-decision-7", evidence_receipt="release-receipt.json")


def configure(entries):
    set_config_value("kanban.pr_acceptance_exceptions", json.dumps(entries))


def events(conn, task, kind):
    return [json.loads(row[0]) for row in conn.execute(
        "SELECT payload FROM task_events WHERE task_id=? AND kind=? ORDER BY id", (task, kind))]


@pytest.fixture
def release(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    for key in ("HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD", "HERMES_KANBAN_TASK", "HERMES_KANBAN_RUN_ID"):
        monkeypatch.delenv(key, raising=False)
    state = dict(head=HEAD, head_tree=TREE, merge_tree=TREE, merged=True,
                 conclusion="success", check_status="completed", requests=[])

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            state["requests"].append(self.path)
            if state.get("error") and state["error"] in self.path:
                self.send_error(403)
                return
            if self.path == "/graphql":
                value = {"data": {"repository": {"pullRequest": {
                    "headRefOid": state["head"], "baseRefName": "main", "state": "MERGED",
                    "baseRef": {"branchProtectionRule": None}}}}}
            elif "/rules/branches/" in self.path:
                self.send_error(403)
                return
            elif "/pulls/7" == self.path[-8:]:
                value = dict(head={"sha": state["head"]}, base={"ref": "main"},
                             state="closed", merged=state["merged"], merge_commit_sha=MERGE)
            elif "/git/commits/" in self.path:
                sha = self.path.rsplit("/", 1)[1]
                value = {"sha": sha, "tree": {"sha": state["head_tree" if sha == HEAD else "merge_tree"]}}
            elif "/check-runs" in self.path:
                run = dict(id=42, name="optional", head_sha=state["head"],
                           status=state["check_status"], conclusion=state["conclusion"],
                           html_url="https://github.com/acme/repo/actions/runs/42")
                value = [{"total_count": state.get("total_count", 1),
                          "check_runs": state.get("runs_page", [] if state.get("no_runs") else [run])}]
                if state.get("race"):
                    state["race"]()
            elif "/statuses" in self.path:
                value = [state.get("status_page", state.get("statuses", []))]
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(value).encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    shim = tmp_path / "bin"
    shim.mkdir()
    gh = shim / "gh"
    gh.write_text(f"#!{sys.executable}\nimport sys,json,urllib.request\n"
                  "assert '--slurp' not in sys.argv\n"
                  f"u='http://127.0.0.1:{server.server_port}/'+sys.argv[2]\n"
                  "value=json.loads(urllib.request.urlopen(u).read())\n"
                  "pages=value if '--paginate' in sys.argv else [value]\n"
                  "print('\\n'.join(map(json.dumps,pages)))\n")
    gh.chmod(0o755)
    monkeypatch.setenv("PATH", str(shim) + os.pathsep + os.environ["PATH"])
    try:
        with closing(connect()) as conn:
            task = kb.create_task(conn, title="Released", completion_contract=URL)
            yield conn, task, state, home
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.linux_only
def test_owner_exception_completes_only_with_separate_truthful_receipt(release):
    conn, task, state, home = release
    assert not kb.complete_task(conn, task, summary="local evidence alone is not authority")
    configure([policy(task)])
    child = kb.create_task(conn, title="Next phase", parents=[task])
    assert kb.complete_task(conn, task, summary="Owner-approved exact local release")
    assert kb.get_task(conn, task).status == "done"
    assert kb.get_task(conn, child).status == "ready"
    assert kb.get_task(conn, task).completion_contract == URL
    strict = events(conn, task, "pr_acceptance")[-1]
    receipt = events(conn, task, "pr_owner_exception")[-1]
    assert not strict["ok"] and strict["classification"] == "infra"
    assert receipt["ok"] and receipt["classification"] == "owner_exception"
    assert receipt["scope"] == policy(task)
    assert receipt["head_sha"] == HEAD and receipt["tree_sha"] == TREE
    assert receipt["merge_sha"] == MERGE
    assert receipt["checks"][0]["id"] == 42
    assert receipt["remote_required_ci_accepted"] is False


@pytest.mark.linux_only
@pytest.mark.parametrize("change", ["remove", "replace", "malformed"])
def test_owner_policy_rechecked_after_network_collection(release, change):
    conn, task, state, home = release
    configure([policy(task)])
    def mutate():
        if change == "malformed":
            (home / "config.yaml").write_text("kanban: [broken")
        else:
            configure([] if change == "remove" else [{**policy(task), "approval_reference": "new-decision"}])
    state["race"] = mutate
    assert not kb.complete_task(conn, task, summary="revoked during collection")
    assert kb.get_task(conn, task).status != "done"
    receipt = events(conn, task, "pr_owner_exception")[-1]
    assert not receipt["ok"] and receipt["classification"] == "stale"


@pytest.mark.linux_only
@pytest.mark.parametrize("field,value", [
    ("board", "other"), ("task_id", "t_00000000"), ("pr_url", URL + "0"),
    ("head_sha", "d" * 40), ("tree_sha", "d" * 40),
    ("head_sha", "short"), ("board", "*"), ("pr_url", "acme/repo"),
    ("approval_reason", ""), ("approval_reference", None), ("evidence_receipt", ""),
])
def test_owner_exception_is_exact_not_worker_metadata(release, field, value):
    conn, task, state, home = release
    invalid = {**policy(task), field: value}
    configure([invalid])
    assert not kb.complete_task(conn, task, metadata={"pr_acceptance_exceptions": [policy(task)]})
    assert kb.get_task(conn, task).completion_contract == URL
    assert kb.get_task(conn, task).status != "done"
    configure([policy(task)])
    assert kb.complete_task(conn, task)


@pytest.mark.linux_only
@pytest.mark.parametrize("fault", [
    {"merged": False}, {"head": "d" * 40}, {"head_tree": "d" * 40}, {"merge_tree": "d" * 40},
    {"error": "/pulls/"}, {"error": "/git/commits/"}, {"error": "/check-runs"}, {"error": "/statuses"},
    {"conclusion": "failure"}, {"check_status": "in_progress"}, {"conclusion": "cancelled"},
    {"conclusion": "neutral"}, {"conclusion": "skipped"}, {"conclusion": None},
    {"total_count": 2}, {"no_runs": True}, {"status_page": {}},
    pytest.param({"total_count": 0, "runs_page": {}}, id="malformed-check-page"),
    {"statuses": [{"id": 2, "context": "legacy", "state": "pending"}]},
    {"statuses": [{"id": 2, "context": "legacy", "state": "failure"}]},
])
def test_owner_exception_never_waives_bad_release_or_checks(release, fault):
    conn, task, state, home = release
    configure([policy(task)])
    original = copy.deepcopy(state)
    state.update(fault)
    assert not kb.complete_task(conn, task)
    assert kb.get_task(conn, task).status != "done"
    assert not events(conn, task, "pr_owner_exception")[-1]["ok"]
    state.clear()
    state.update(original)
    assert kb.complete_task(conn, task)


@pytest.mark.linux_only
def test_owner_exception_cannot_attach_to_reclaimed_run(release):
    conn, task, state, home = release
    configure([policy(task)])
    owner = kb.claim_task(conn, task)
    def reclaim():
        with closing(connect()) as rival:
            assert kb.block_task(rival, task, reason="reclaimed")
            assert kb.unblock_task(rival, task)
            state["replacement"] = kb.claim_task(rival, task).current_run_id
    state["race"] = reclaim
    assert not kb.complete_task(conn, task, expected_run_id=owner.current_run_id)
    assert kb.get_task(conn, task).current_run_id == state["replacement"]
    assert not events(conn, task, "pr_acceptance")
    assert not events(conn, task, "pr_owner_exception")


@pytest.mark.linux_only
def test_native_complete_uses_profile_config_not_launch_profile(release, monkeypatch, tmp_path, request):
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from agent import secret_scope
    from tools import kanban_tools as kt
    conn, task, state, home = release
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    secret_token = secret_scope.set_secret_scope({})
    request.addfinalizer(lambda: secret_scope.reset_secret_scope(secret_token))
    a_token = set_hermes_home_override(home)
    request.addfinalizer(lambda: reset_hermes_home_override(a_token))
    configure([policy(task)])
    other = tmp_path / "profile-b"
    other.mkdir()
    # A -> B -> A: same actual board, policy belongs only to A.
    token = set_hermes_home_override(other)
    try:
        result = json.loads(kt._handle_complete({"task_id": task, "summary": "B cannot use A approval"}))
        assert "PR acceptance infra" in result["error"], result
        assert kb.get_task(conn, task).status != "done"
        assert events(conn, task, "pr_acceptance")[-1]["classification"] == "infra"
    finally:
        reset_hermes_home_override(token)
    owner = kb.claim_task(conn, task)
    monkeypatch.setenv("HERMES_KANBAN_TASK", task)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(owner.current_run_id))
    result = json.loads(kt._handle_complete({"summary": "Exact approved local release"}))
    assert result["ok"], result
    assert kb.get_task(conn, task).status == "done"
    assert events(conn, task, "pr_owner_exception")[-1]["classification"] == "owner_exception"


@pytest.mark.linux_only
def test_exception_board_identity_comes_from_connection(release, tmp_path):
    conn, task, state, home = release
    configure([policy(task)])
    # A clone has the same task id but is not the approved board.
    with closing(connect(board="other")) as other:
        conn.backup(other)
        assert not kb.complete_task(other, task)
        assert kb.get_task(other, task).status != "done"
        configure([{**policy(task), "board": "other"}])
        assert kb.complete_task(other, task)
    assert not kb.complete_task(conn, task)
    configure([policy(task)])
    assert kb.complete_task(conn, task)


@pytest.mark.linux_only
def test_empty_checks_are_local_exception_not_green_ci(release):
    conn, task, state, home = release
    configure([policy(task)])
    state.update(no_runs=True, total_count=0)
    assert kb.complete_task(conn, task)
    receipt = events(conn, task, "pr_owner_exception")[-1]
    assert receipt["checks"] == []
    assert receipt["remote_required_ci_accepted"] is False


@pytest.mark.linux_only
@pytest.mark.parametrize("shape", ["missing", "extra", "duplicate", "mapping", "null"])
def test_invalid_owner_policy_does_not_authorize(release, shape):
    conn, task, state, home = release
    entry = policy(task)
    values = {"missing": [{k: v for k, v in entry.items() if k != "approval_reference"}],
              "extra": [{**entry, "wildcard": True}], "duplicate": [entry, entry],
              "mapping": entry, "null": None}
    if shape in ("mapping", "null"):
        # The writer rejects these shapes; exercise runtime rejection too, as
        # malformed policy can still arrive through a hand-edited config.
        with pytest.raises(SystemExit) as rejected:
            configure(values[shape])
        assert rejected.value.code == 1
        import yaml
        (home / "config.yaml").write_text(yaml.safe_dump({
            "kanban": {"pr_acceptance_exceptions": values[shape]},
        }))
    else:
        configure(values[shape])
    assert not kb.complete_task(conn, task)
    assert events(conn, task, "pr_owner_exception")[-1]["classification"] == "invalid_policy"
