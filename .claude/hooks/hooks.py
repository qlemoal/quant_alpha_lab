#!/usr/bin/env python3
"""Claude Code hooks. `guard` (PreToolUse) protects data/raw and old signal runs. `test` (PostToolUse) runs pytest.

shortcut: the guard reads the command text, so it stops accidents, not a script that writes by itself
(or a path held in a shell variable). Use file permissions if you need a hard lock.
"""
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'data' / 'raw'
RUN_STAGES = ('signals', 'signal_eval')
PYTEST_PYTHON = '/Users/lemoal/miniforge3/envs/quantlab311/bin/python'
RAW_WRITERS = ('01_download_data', '01_import_crsp', '01_import_vendor_data',
               'make_synthetic_market', 'run_synthetic_reference', 'survivorship_lab')


def run_dirs():
    return [p for s in RUN_STAGES for p in ROOT.glob(f'data/clean/*/*/*/*/runs/*/{s}') if p.is_dir()]


def inside(path, parent):
    return path == parent or parent in path.parents


def protected(path):
    """Why a write to this path is refused, or None."""
    if inside(path, RAW):
        return 'data/raw is read-only'
    for run in run_dirs():
        if inside(path, run):
            return f'{run.parent.name}/{run.name} belongs to an existing run; use a new tag'
    return None


def destroys(path):
    """As protected(), and also true for a parent folder: removing it removes what is inside."""
    return protected(path) or next(
        (f'{path.name}/ contains protected data' for p in [RAW, *run_dirs()] if p.exists() and path in p.parents), None)


def resolve(arg, cwd):
    arg = re.split(r'[*?\[{]', arg)[0] or '.'          # a glob is judged by its fixed prefix
    return Path(os.path.normpath(os.path.join(cwd, os.path.expanduser(arg))))


def check_command(command, cwd):
    """(decision, reason) or None."""
    # A heredoc body is data (a document, a script for python), except when a shell reads it.
    command = re.sub(r'(?m)^(?!.*\b(?:ba|z)?sh\b[^\n]*<<)(.*?<<-?[ \t]*([\'"]?)(\w+)\2[^\n]*)\n(?s:.*?)^[ \t]*\3[ \t]*$',
                     r'\1', command)
    lex = shlex.shlex(command.replace('\n', ' ; '), posix=True, punctuation_chars=True)
    lex.whitespace_split = True
    try:
        tokens = list(lex)
    except ValueError:
        tokens = command.split()
    segments = [[]]
    for t in tokens:
        if t and set(t) <= set(';&|()'):
            segments.append([])
        else:
            segments[-1].append(t)

    ask = None
    for seg in segments:
        # redirections: the target is written
        for i, t in enumerate(seg[:-1]):
            if t in ('>', '>>', '>|') and (why := protected(resolve(seg[i + 1], cwd))):
                return 'deny', why
        while seg and (seg[0] in ('rtk', 'proxy', 'sudo', 'env', 'command', 'time', 'nohup') or re.match(r'\w+=', seg[0])):
            seg = seg[1:]
        if not seg:
            continue
        name, args = os.path.basename(seg[0]), [a for a in seg[1:] if not a.startswith('-')]
        if name == 'cd' and args:
            cwd = str(resolve(args[0], cwd))
            continue

        targets = []
        if name in ('rm', 'rmdir', 'mv', 'truncate', 'shred', 'tee', 'touch'):
            targets = args
        elif name in ('cp', 'rsync', 'ln'):
            targets = args[-1:]
        elif name == 'sed' and any(a.startswith(('-i', '--in-place')) for a in seg[1:]):
            targets = args[1:]
        elif name == 'find' and ('-delete' in seg or '-exec' in seg):
            targets = args[:1]
        elif name == 'dd':
            targets = [a[3:] for a in args if a.startswith('of=')]
        elif name == 'git' and args[:1] in (['rm'], ['mv'], ['restore'], ['checkout']):
            targets = args[1:]
        elif name == 'git' and args[:1] == ['clean']:
            ask = 'git clean can delete data files'
        for a in targets:
            if why := destroys(resolve(a, cwd)):
                return 'deny', why

    # pipeline scripts: matched on the whole command, whatever starts them
    if any(re.search(rf'{w}(\.py|\b)', command) for w in RAW_WRITERS) and re.search(r'\bpython|\./|-m ', command):
        ask = 'this script writes into data/raw'
    date = re.search(r'--date[= ]+["\']?([\d-]+)', command)
    if date and '--check' not in command:
        tag = re.search(r'--tag[= ]+["\']?([\w.-]+)', command)
        run_id = f'{date[1]}_{tag[1] if tag else "default"}'
        first = re.search(r'--(?:from|only)[= ]+0?(\d)', command)
        stages = []
        if '04_generate_signals' in command or ('run_experiment' in command and (not first or first[1] == '4')):
            stages.append('signals')
        if '05_evaluate_signals' in command or ('run_experiment' in command and (not first or first[1] in '45')):
            stages.append('signal_eval')
        for run in run_dirs():
            if run.parent.name == run_id and run.name in stages:
                return 'deny', f'runs/{run_id}/{run.name} already exists and would be overwritten; use a new --tag'

    return ('ask', ask) if ask else None


def guard(event):
    tool_input = event.get('tool_input', {})
    cwd = event.get('cwd') or str(ROOT)
    if event.get('tool_name') == 'Bash':
        result = check_command(tool_input.get('command', ''), cwd)
    else:
        path = tool_input.get('file_path') or tool_input.get('notebook_path')
        why = protected(Path(os.path.normpath(os.path.join(cwd, path)))) if path else None
        result = ('deny', why) if why else None
    if result:
        print(json.dumps({'hookSpecificOutput': {
            'hookEventName': 'PreToolUse', 'permissionDecision': result[0],
            'permissionDecisionReason': f'Project guard: {result[1]}.'}}))


def test(event):
    path = Path(event.get('tool_input', {}).get('file_path') or '')
    code = [ROOT / d for d in ('src', 'scripts', 'config', 'tests', '.claude/hooks')]
    if path.suffix not in ('.py', '.json') or not any(c in path.parents for c in code):
        return 0
    python = PYTEST_PYTHON if os.path.exists(PYTEST_PYTHON) else sys.executable
    run = subprocess.run([python, '-m', 'pytest', '-q', '-x', '--disable-warnings'],
                         cwd=ROOT, capture_output=True, text=True)
    if run.returncode:
        print(f'pytest failed after the edit of {path.name}:\n{(run.stdout + run.stderr)[-4000:]}', file=sys.stderr)
        return 2                                     # exit 2 sends stderr back to Claude
    return 0


if __name__ == '__main__':
    payload = json.load(sys.stdin)
    sys.exit(test(payload) if sys.argv[1] == 'test' else guard(payload))
