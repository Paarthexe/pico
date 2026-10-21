"""Portable evidence reports, notebooks, and chart assets."""
from dataclasses import asdict
from io import BytesIO
import json
import zipfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from pico.agent import Investigation
from pico.analysis import Evidence


def generate_plot(e: Evidence) -> bytes:
    h = e.hypothesis
    table = pd.DataFrame(e.table)
    with plt.rc_context({'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.labelcolor': '#121c22', 'text.color': '#121c22'}):
        fig, ax = plt.subplots(figsize=(8, 4.5), layout='constrained')
        if h['kind'] == 'correlation':
            ax.scatter(table[h['x']], table[h['y']], s=12, alpha=.35, color='#017ca8')
            ax.set(xlabel=h['x'], ylabel=h['y'])
        elif h['kind'] == 'numeric_groups':
            ax.bar(table['group'].astype(str), table['median'], color='#017ca8')
            ax.set(xlabel=h['x'], ylabel=f"Median {h['y']}")
        else:
            # Each group/stratum sums to 100%; no privileged positive label assumed.
            table['label'] = table['group'] + (' / ' + table['stratum'] if h['control'] else '')
            pivot = table.pivot(index='label', columns='category', values='proportion').fillna(0)
            pivot.plot.bar(stacked=True, ax=ax, colormap='tab20', width=.7)
            ax.set(xlabel=h['x'], ylabel=f"Proportion of {h['y']}", ylim=(0, 1))
            ax.legend(title=h['y'], bbox_to_anchor=(1.01, 1), loc='upper left')
        ax.set_title(f"{e.id} · {h['x']} and {h['y']}", loc='left', pad=16)
        ax.tick_params(axis='x', labelrotation=35)
        output = BytesIO()
        fig.savefig(output, format='png', dpi=140)
        plt.close(fig)
    return output.getvalue()


def markdown_report(run: Investigation) -> str:
    lines = ['# Pico investigation', '', f'**Question:** {run.question or "Explore associations with the selected target."}',
             f'**Target:** {run.target}', f'**Planner:** {run.mode}',
             f'**Dataset:** {run.profile["rows"]:,} rows, {run.profile["columns"]} columns', '',
             'All results are exploratory associations. Adaptive selection and multiple comparisons '
             'can exaggerate apparent signals. Validate on independent data before acting.', '']
    if not run.evidence:
        lines += ['No eligible analysis produced evidence.', '']
    for e in run.evidence:
        lines += [f'## {e.id}: {e.hypothesis["x"]} and {e.hypothesis["y"]}', '',
                  e.conclusion, '', f'Usable observations: {e.n}.',
                  f'Raw p-value: {e.p_value:.4g}.' if e.p_value is not None else 'No pooled p-value reported.', '',
                  f'![{e.id} analysis chart](charts/{e.id}.png)', '',
                  *[f'- {c}' for c in e.limitations], '',
                  '```python', e.code, '```', '']
    lines += ['## Investigation log', '', *[f'- **{a["stage"]}:** {a["message"]}' for a in run.activity], '',
              '## Suggested next steps', '', 'Confirm candidate associations on held-out data, inspect measurement '
              'quality and possible leakage, and consult domain context before interpreting mechanisms.', '',
              f'Stopped: {run.stop_reason}']
    return '\n'.join(lines)


def notebook(run: Investigation) -> str:
    cells = [dict(cell_type='markdown', metadata={}, source=['# Pico reproducible investigation\n',
              'Install pico-eda and put the original CSV beside this notebook as dataset.csv.\n',
              'These are exploratory associations, not causal estimates.'])]
    for e in run.evidence:
        cells.append(dict(cell_type='markdown', metadata={}, source=[f'## {e.id}\n', e.conclusion]))
        cells.append(dict(cell_type='code', metadata={}, execution_count=None, outputs=[],
                          source=e.code.splitlines(keepends=True)))
    return json.dumps(dict(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3',
                      'language': 'python', 'name': 'python3'}}, nbformat=4, nbformat_minor=5), indent=2)


def export_bundle(run: Investigation) -> bytes:
    output = BytesIO()
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('report.md', markdown_report(run))
        archive.writestr('evidence.json', json.dumps(asdict(run), indent=2, allow_nan=False))
        archive.writestr('investigation.ipynb', notebook(run))
        for e in run.evidence:
            archive.writestr(f'charts/{e.id}.png', generate_plot(e))
            archive.writestr(f'code/{e.id}.py', e.code)
    return output.getvalue()
