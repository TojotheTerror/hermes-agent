"""gh before --slurp emits sequential JSON pages, not one JSON array."""
import json
import subprocess

import pytest

from hermes_cli.kanban_pr_acceptance import _api


@pytest.mark.parametrize("pages", [
    [[{"type": "first"}], [{"type": "last"}]],
    [{"total_count": 2, "check_runs": [{"id": 1}]},
     {"total_count": 2, "check_runs": [{"id": 2}]}],
    [[]],
])
def test_paginated_api_preserves_all_page_shapes(monkeypatch, pages):
    def gh(command, **kwargs):
        assert "--paginate" in command
        assert "--slurp" not in command
        return subprocess.CompletedProcess(command, 0, " \n" + "\n".join(map(json.dumps, pages)) + "\n")
    monkeypatch.setattr(subprocess, "run", gh)
    assert _api("fixture", paginate=True) == pages


@pytest.mark.parametrize("output", ['[]\n{', '[]\ngarbage', '', '  ', '[]\nnull', '[]\n{"errors":[{}]}'])
def test_paginated_api_rejects_incomplete_or_invalid_pages(monkeypatch, output):
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(a, 0, output))
    with pytest.raises(ValueError):
        _api("fixture", paginate=True)
