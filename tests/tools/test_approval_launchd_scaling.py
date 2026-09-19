"""Approval detection keeps whole-command launchd coverage without suffix rescans."""

import subprocess
import sys

import pytest

from tools.approval_detection import detect_dangerous_command


@pytest.mark.parametrize("verb", ["stop", "kickstart", "bootout", "unload", "kill", "disable", "remove"])
@pytest.mark.parametrize("label_first", [False, True])
def test_launchd_approval_matches_label_and_verb_in_either_order(verb, label_first):
    label = "label=ai.hermes.gateway"
    lifecycle = f'launchctl {verb} "$label"'
    parts = [label, lifecycle] if label_first else [lifecycle, label]
    command = "printf before\n" + "\n".join(parts)

    dangerous, key, description = detect_dangerous_command(command)

    assert dangerous is True
    assert key == description == "stop/restart hermes launchd service (kills running agents)"


@pytest.mark.parametrize("command", [
    "printf before; launchctl stop com.example.worker",
    "label=ai.hermes.gateway\nlaunchctl print \"$label\"",
    "label=ai.hermes.gateway\nlaunchctl stopper \"$label\"",
    "label=myhermes\nlaunchctl stop \"$label\"",
])
def test_launchd_approval_requires_both_label_and_lifecycle_verb(command):
    assert detect_dangerous_command(command) == (False, None, None)


def test_approval_many_benign_segments_complete_within_deadline():
    # Bound the real detector in a disposable process: a regex regression must
    # fail this test, not hang the entire file until the runner's 300s deadline.
    completed = subprocess.run(
        [sys.executable, "-c", """
from tools.approval_detection import detect_dangerous_command
for count in (2000, 4000):
    command = ';'.join(f'printf segment-{index}' for index in range(count))
    assert detect_dangerous_command(command) == (False, None, None)
"""],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
