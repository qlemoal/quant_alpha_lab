'''The Claude Code guard (.claude/hooks/hooks.py) refuses writes to data/raw and to old signal runs.'''

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('claude_hooks', ROOT / '.claude' / 'hooks' / 'hooks.py')
hooks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hooks)


@pytest.fixture(autouse=True)
def fake_repo(tmp_path, monkeypatch):
    (tmp_path / 'data' / 'raw' / 'src').mkdir(parents=True)
    (tmp_path / 'data' / 'clean' / 'equities' / 'US' / 'X' / 'daily' / 'runs' / '2026-01-01_old' / 'signals').mkdir(parents=True)
    monkeypatch.setattr(hooks, 'ROOT', tmp_path)
    monkeypatch.setattr(hooks, 'RAW', tmp_path / 'data' / 'raw')
    return tmp_path


OLD = 'data/clean/equities/US/X/daily/runs/2026-01-01_old/signals'


@pytest.mark.parametrize('command', [
    'rm -rf data/raw/src',
    'rtk rm -rf data',
    'cd data && rm -r raw',
    'echo x > data/raw/a.csv',
    'cp a.csv data/raw/',
    f'rm {OLD}/signals.parquet',
    f'mv {OLD} /tmp/x',
    f'rm -rf {OLD}/..',                      # the whole run folder
    'python scripts/pipeline/04_generate_signals.py --date 2026-01-01 --tag old',
    'python scripts/pipeline/run_experiment.py --date=2026-01-01 --tag=old',
])
def test_denied(command, fake_repo):
    assert hooks.check_command(command, str(fake_repo))[0] == 'deny'


@pytest.mark.parametrize('command', [
    'ls data/raw',
    'cat data/raw/a.csv > /tmp/a.csv',
    'cp data/raw/a.csv /tmp/a.csv',
    f'cat {OLD}/config.json',
    'rm -rf tmp',
    'python scripts/pipeline/04_generate_signals.py --date 2026-01-01 --tag new',
    'python scripts/pipeline/run_experiment.py --date 2026-01-01 --tag old --from 07',
])
def test_allowed(command, fake_repo):
    assert hooks.check_command(command, str(fake_repo)) is None


def test_heredoc_body_is_data_unless_a_shell_reads_it(fake_repo):
    run = 'python scripts/pipeline/04_generate_signals.py --date 2026-01-01 --tag old'
    assert hooks.check_command(f"cat > notes.md <<'EOF'\n{run}\nEOF", str(fake_repo)) is None
    assert hooks.check_command(f"bash <<'EOF'\n{run}\nEOF", str(fake_repo))[0] == 'deny'
    assert hooks.check_command(f"cat <<EOF\nx\nEOF\n{run}", str(fake_repo))[0] == 'deny'


def test_download_asks(fake_repo):
    assert hooks.check_command('python scripts/pipeline/01_download_data.py', str(fake_repo))[0] == 'ask'
