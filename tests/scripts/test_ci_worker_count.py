"""The workflow passes one test worker per available logical CPU."""
import os
from pathlib import Path
import subprocess

import pytest
import yaml


@pytest.mark.linux_only
def test_python_workflow_worker_count_matches_host(tmp_path):
    root = Path(__file__).resolve().parents[2]
    workflow = yaml.safe_load((root / '.github/workflows/tests.yml').read_text())
    step = next(s for s in workflow['jobs']['test']['steps'] if s.get('name') == 'Run tests')
    (tmp_path / '.venv/bin').mkdir(parents=True)
    (tmp_path / '.venv/bin/activate').write_text('')
    (tmp_path / 'scripts').mkdir()
    runner = tmp_path / 'scripts/run_tests.sh'
    runner.write_text('#!/bin/sh\nprintf "%s" "$HERMES_TEST_WORKERS"\n')
    runner.chmod(0o755)
    result = subprocess.run(['bash', '-e', '-c', step['run']], cwd=tmp_path,
                            env={**os.environ, **{k: str(v) for k, v in step.get('env', {}).items()}},
                            text=True, capture_output=True, check=True)
    expected = subprocess.check_output(['python', '-c', 'import os; print(os.cpu_count() or 1)'], text=True)
    assert int(result.stdout) == int(expected)
