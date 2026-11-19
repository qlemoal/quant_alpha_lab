'''
Reading the benchmark signals (cheat and placebo) in the FDR table of stage 05.

WHAT IS ASKED
    1. PLACEBOS (pure noise, true IC = 0): how many pass the selection? Only a few by chance are acceptable:
       about alpha_p x n_placebo with a raw p-value below alpha_p, and almost none after the FDR step.
    2. DILUTION: the FDR step controls the SHARE of false signals among the selected ones.
       Add signals that are surely true (cheats), and the same share allows more weak signals to pass.
       The report compares the selection of the real signals
           - inside the full family (real + cheat + placebo), the one 05 uses,
           - alone (the family without benchmarks).
       The difference is the effect of the benchmarks on the selection.

    The family of the test also changes with the placebos: more tests means a stricter threshold
    for each of them. The comparison above shows this too.

INPUT   two tables from fdr_report(): the full family, and the real signals alone (None if there are no benchmarks).
'''

import polars as pl

from src.signals.cheat import signal_group

RAW_P_LEVEL = 0.05          # raw p-value level used to count "looks significant by chance" among placebos


def _selected(fdr: pl.DataFrame, alpha: float) -> list:
    '''Signals with q < alpha (the rule of stage 05).'''
    return fdr.filter(pl.col('qvalue') < alpha)['signal'].to_list()


def benchmark_report(fdr_all: pl.DataFrame, fdr_real: "pl.DataFrame | None", alpha: float) -> "dict | None":
    '''
    Summary of what the selection did with the benchmark signals. None if the family has no benchmark.
    fdr_all:  fdr_report() of every signal of the run.
    fdr_real: fdr_report() of the real signals only (same alpha).
    '''
    groups = {s: signal_group(s) for s in fdr_all['signal'].to_list()}
    if all(g == 'real' for g in groups.values()):
        return None

    out = {'alpha': alpha, 'n_signals': fdr_all.height}
    sel_all = _selected(fdr_all, alpha)
    for g in ('cheat', 'placebo', 'real'):
        names = [s for s, gg in groups.items() if gg == g]
        sub = fdr_all.filter(pl.col('signal').is_in(names))
        out[g] = {
            'n': len(names),
            'selected_q': [s for s in sel_all if groups[s] == g],
            'n_bh': int(sub['survives_bh'].sum()) if names else 0,
            'n_by': int(sub['survives_by'].sum()) if names else 0,
        }

    pl_names = [s for s, g in groups.items() if g == 'placebo']
    if pl_names:
        p = fdr_all.filter(pl.col('signal').is_in(pl_names))['pvalue'].drop_nulls()
        out['placebo'].update({
            'share_p_below_005': float((p < RAW_P_LEVEL).sum() / max(len(p), 1)),
            'expected_p_below_005': RAW_P_LEVEL * len(pl_names),
            'n_p_below_005': int((p < RAW_P_LEVEL).sum()),
            'min_p': float(p.min()) if len(p) else None,
            'min_q': float(fdr_all.filter(pl.col('signal').is_in(pl_names))['qvalue'].min()),
        })

    if fdr_real is not None:
        with_b = out['real']['selected_q']
        without_b = _selected(fdr_real, alpha)
        out['real'].update({
            'selected_with_benchmarks': with_b,
            'selected_alone': without_b,
            'only_thanks_to_benchmarks': [s for s in with_b if s not in without_b],
            'lost_because_of_benchmarks': [s for s in without_b if s not in with_b],
        })
    return out


def format_benchmark_report(rep: dict) -> list:
    '''Plain-text lines for the console.'''
    a = rep['alpha']
    lines = [f'  Benchmark signals in the family ({rep["n_signals"]} signals, selection q < {a}):']
    for g in ('cheat', 'placebo', 'real'):
        r = rep[g]
        if r['n']:
            lines.append(f'    {g:<8} {r["n"]:3d} signals   selected {len(r["selected_q"]):3d}   BH {r["n_bh"]:3d}   BY {r["n_by"]:3d}')
    pl_ = rep['placebo']
    if pl_['n']:
        lines.append(f'    placebo with raw p < {RAW_P_LEVEL}: {pl_["n_p_below_005"]} (about {pl_["expected_p_below_005"]:.1f} expected by chance)'
                     f'   smallest q {pl_["min_q"]:.3f}')
        if pl_['selected_q']:
            lines.append(f'    WARNING: placebo signals selected: {pl_["selected_q"]}. '
                         f'A false discovery rate of {a} allows some, but check the Newey-West lag.')
        else:
            lines.append('    no placebo selected (as expected)')
    real = rep['real']
    if 'selected_alone' in real:
        lines.append(f'    real signals selected: {len(real["selected_with_benchmarks"])} inside the full family, '
                     f'{len(real["selected_alone"])} when tested alone')
        if real['only_thanks_to_benchmarks']:
            lines.append(f'    selected only thanks to the benchmarks (dilution): {real["only_thanks_to_benchmarks"]}')
        if real['lost_because_of_benchmarks']:
            lines.append(f'    lost because of the benchmarks (more tests): {real["lost_because_of_benchmarks"]}')
    return lines