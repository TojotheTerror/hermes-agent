"""Exercise the attribution workflow against real, divergent Git histories."""
import os
from pathlib import Path
import subprocess

import pytest
import yaml


WORKFLOW = Path(__file__).resolve().parents[2] / '.github/workflows/contributor-check.yml'


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()


@pytest.mark.linux_only
@pytest.mark.parametrize('mapped', [True, False])
def test_pinned_pr_checks_only_real_delta(tmp_path, mapped):
    repo = tmp_path / 'repo'
    repo.mkdir()
    git(repo, 'init', '-b', 'main')
    git(repo, 'config', 'user.name', 'Fixture')
    git(repo, 'config', 'user.email', 'historic@example.invalid')
    git(repo, 'commit', '--allow-empty', '-m', 'root')
    git(repo, 'remote', 'add', 'origin', str(repo))
    git(repo, 'update-ref', 'refs/remotes/origin/main', 'HEAD')
    git(repo, 'checkout', '-b', 'baseline')
    git(repo, 'commit', '--allow-empty', '-m', 'historic baseline change')
    base = git(repo, 'rev-parse', 'HEAD')
    git(repo, 'checkout', '-b', 'repair')
    git(repo, 'config', 'user.email', 'repair@example.invalid')
    git(repo, 'commit', '--allow-empty', '-m', 'repair')
    head = git(repo, 'rev-parse', 'HEAD')
    (repo / 'scripts').mkdir()
    (repo / 'scripts/release.py').write_text('AUTHOR_MAP = {}\n')
    (repo / 'contributors/emails').mkdir(parents=True)
    if mapped:
        (repo / 'contributors/emails/repair@example.invalid').write_text('fixture\n')
    # A checkout merge can include base-only commits: they are not PR authors.
    git(repo, 'checkout', 'baseline')
    git(repo, 'config', 'user.email', 'base-only@example.invalid')
    git(repo, 'commit', '--allow-empty', '-m', 'base advanced')
    git(repo, 'merge', '--no-ff', 'repair', '-m', 'synthetic checkout merge')
    step = next(s for s in yaml.safe_load(WORKFLOW.read_text())['jobs']['check-attribution']['steps'] if s.get('id') == 'check-emails')
    result = subprocess.run(
        ['bash', '-eo', 'pipefail', '-c', step['run']], cwd=repo,
        env={**os.environ, 'PR_BASE_SHA': base, 'PR_HEAD_SHA': head,
             'GITHUB_OUTPUT': str(tmp_path / 'output')}, text=True, capture_output=True,
    )
    assert result.returncode == (0 if mapped else 1), result.stdout + result.stderr
    if not mapped:
        assert 'repair@example.invalid' in result.stdout
        assert 'historic@example.invalid' not in result.stdout
        assert 'base-only@example.invalid' not in result.stdout


@pytest.mark.linux_only
@pytest.mark.parametrize('mode', ['fallback', 'partial', 'invalid', 'missing-object'])
def test_attribution_fallback_and_bad_event_data(tmp_path, mode):
    git(tmp_path, 'init', '-b', 'main')
    git(tmp_path, 'config', 'user.name', 'Fixture')
    git(tmp_path, 'config', 'user.email', 'fixture@example.invalid')
    git(tmp_path, 'commit', '--allow-empty', '-m', 'base')
    git(tmp_path, 'remote', 'add', 'origin', str(tmp_path))
    git(tmp_path, 'update-ref', 'refs/remotes/origin/main', 'HEAD')
    sha = git(tmp_path, 'rev-parse', 'HEAD')
    values = {
        'fallback': ('', ''),
        'partial': (sha, ''),
        'invalid': ('$(touch injected)', sha),
        'missing-object': ('f' * 40, sha),
    }
    base, head = values[mode]
    step = next(s for s in yaml.safe_load(WORKFLOW.read_text())['jobs']['check-attribution']['steps'] if s.get('id') == 'check-emails')
    result = subprocess.run(
        ['bash', '-eo', 'pipefail', '-c', step['run']], cwd=tmp_path,
        env={**os.environ, 'PR_BASE_SHA': base, 'PR_HEAD_SHA': head,
             'GITHUB_OUTPUT': str(tmp_path / 'output')}, capture_output=True, text=True,
    )
    assert (result.returncode == 0) == (mode == 'fallback')
    assert not (tmp_path / 'injected').exists()
    if mode == 'fallback':
        assert (tmp_path / 'output').read_text().strip() == 'review_status=[]'
