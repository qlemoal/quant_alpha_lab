'''
One printing style for every stage of the pipeline, written for a human who reads the log once.

    ==============================================================================
    STAGE 05 | EVALUATE SIGNALS
    Which signals carry real information about future returns?
    ==============================================================================

    [2/6] Test each signal: is its mean IC above zero?
          why   the IC series has memory (overlapping labels); a block bootstrap keeps it
          block length                  48 days
          [PASS] 3 of 15 signals have p < 5% on their own

Every step says WHAT it does, WHY, and the RESULT. Results that are a judgement carry a tag:
    [PASS]  the check is fine            [FAIL]  the check failed (the stage may stop)
    [WARN]  look at this before you trust the numbers        [NOTE]  context, nothing to do
'''

import textwrap
from pathlib import Path

WIDTH = 100
INDENT = '      '
ROOT = Path(__file__).resolve().parents[2]


def _wrap(text: str, first: str, rest: str) -> str:
    return textwrap.fill(str(text), width=WIDTH, initial_indent=first, subsequent_indent=rest)


def stage(code: str, title: str, question: str) -> None:
    '''Header of a stage: its number, its name, and the question it answers.'''
    print('=' * WIDTH)
    print(f'STAGE {code} | {title.upper()}')
    print(question)
    print('=' * WIDTH)


def step(i: int, n: int, title: str, why: "str | None" = None) -> None:
    '''Start of a step: what it does, and (optional) why.'''
    print('\n' + _wrap(title, f'[{i}/{n}] ', ' ' * (len(str(i)) + len(str(n)) + 4)))
    if why:
        print(_wrap(why, f'{INDENT}why   ', INDENT + '      '))


def info(label: str, value) -> None:
    '''A plain number or fact: "label ...... value".'''
    print(_wrap(value, f'{INDENT}{label:<30}', INDENT + ' ' * 30))


def _tag(tag: str, text: str) -> None:
    print(_wrap(text, f'{INDENT}[{tag}] ', INDENT + ' ' * (len(tag) + 3)))


def ok(text: str) -> None:
    _tag('PASS', text)


def fail(text: str) -> None:
    _tag('FAIL', text)


def warn(text: str) -> None:
    _tag('WARN', text)


def note(text: str) -> None:
    _tag('NOTE', text)


def verdict(passed: bool, text: str) -> None:
    '''[PASS] or [FAIL] depending on `passed`.'''
    (ok if passed else fail)(text)


def table_row(cells: list, widths: list) -> None:
    '''One row of a fixed-width table. widths: one int per column (negative = left-aligned).'''
    print(INDENT + ' '.join(str(c).ljust(-w) if w < 0 else str(c).rjust(w) for c, w in zip(cells, widths)))


def table(header: list, rows: list, widths: list) -> None:
    '''A small fixed-width table: header, a line, the rows.'''
    table_row(header, widths)
    print(INDENT + '-' * (sum(abs(w) for w in widths) + len(widths) - 1))
    for r in rows:
        table_row(r, widths)


def done(outputs: dict, next_cmd: "str | None" = None) -> None:
    '''End of a stage: the files written, and the next command.'''
    print('\n' + '-' * WIDTH)
    print('Saved:')
    for label, path in outputs.items():
        print(f'{INDENT}{label:<22}{short_path(path)}')
    if next_cmd:
        print(f'Next:  {next_cmd}')
    print('-' * WIDTH)


def short_path(path) -> str:
    '''A path relative to the repository root when it is inside it (easier to read).'''
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def num(x, spec: str = '.3f', na: str = 'n/a') -> str:
    '''Format a number that may be None or NaN.'''
    try:
        if x is None or x != x:
            return na
        return format(x, spec)
    except (TypeError, ValueError):
        return na
