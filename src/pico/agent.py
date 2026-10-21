"""Investigation orchestration. Providers, planning policy, and executors are replaceable."""
from collections.abc import Callable
import pandas as pd
from pico.analysis import execute_analysis
from pico.config import MAX_INVESTIGATIONS
from pico.data import profile_dataset
from pico.models import Evidence, Investigation
from pico.ollama import PlannerError
from pico.planning import choose_target, follow_up, ollama_choice, ranked_candidates

EventCallback = Callable[[str, str], None]


def _emit(run: Investigation, callback: EventCallback | None, stage: str, message: str) -> None:
    run.activity.append(dict(stage=stage, message=message))
    if callback:
        callback(stage, message)


def _compare_strata(run: Investigation, evidence: Evidence, emit: EventCallback) -> None:
    if evidence.hypothesis['kind'] != 'stratified':
        return
    prior = next((e for e in run.evidence if e.hypothesis['kind'] == 'groups'
                  and e.hypothesis['x'] == evidence.hypothesis['x']), None)
    if prior and prior.effect and evidence.effect is not None:
        change = (evidence.effect / prior.effect - 1) * 100
        emit('Evaluate', f'Within-stratum association is {abs(change):.0f}% '
             f'{"smaller" if change < 0 else "larger"} than {prior.id}; this comparison is descriptive.')


def _catalog_loop(df: pd.DataFrame, run: Investigation, emit: EventCallback) -> None:
    pool = ranked_candidates(df, run.target)
    seen = set()
    emit('Plan', f'{len(pool)} eligible hypotheses; budget {run.budget} analyses.')
    for index in range(1, run.budget + 1):
        pool = [h for h in pool if h.key not in seen]
        if not pool:
            run.stop_reason = 'No untested eligible hypotheses remain.'
            return
        choice, reason = 0, pool[0].reason
        if run.model:
            try:
                choice, reason = ollama_choice(pool, run, run.model)
            except PlannerError as exc:
                emit('Planner fallback', f'{exc} Using adaptive rules.')
                run.mode = 'Local adaptive planner (Ollama fallback)'
        hypothesis = pool.pop(choice)
        seen.add(hypothesis.key)
        emit('Investigate', reason)
        try:
            evidence = execute_analysis(df, hypothesis, index)
        except (ValueError, TypeError, KeyError) as exc:
            emit('Analysis skipped', f'{hypothesis.x} / {hypothesis.y}: {exc}')
            continue
        run.evidence.append(evidence)
        emit('Evaluate', f'{evidence.id}: {evidence.conclusion} n={evidence.n}.')
        follow = follow_up(df, evidence)
        if follow and follow.key not in seen and index < run.budget:
            pool.insert(0, follow)
            emit('Replan', follow.reason)
        _compare_strata(run, evidence, emit)
    run.stop_reason = f'Reached the {run.budget}-analysis budget.'


def investigate(df: pd.DataFrame, target: str, question: str = '', budget: int = 5,
                model: str | None = None, on_event: EventCallback | None = None,
                *, engine: str = 'catalog', executor=None, planner=None) -> Investigation:
    if type(budget) is not int or not 1 <= budget <= MAX_INVESTIGATIONS:
        raise ValueError(f'Investigation budget must be between 1 and {MAX_INVESTIGATIONS}.')
    if target not in df or df[target].nunique() < 2:
        raise ValueError('Choose a target column with at least two distinct values.')
    if engine not in {'catalog', 'python'}:
        raise ValueError('Unknown analysis engine.')
    if engine == 'python' and not model:
        raise ValueError('Generated Python requires a local model.')
    mode = ('Ollama / generated Python' if engine == 'python' else
            'Ollama' if model else 'Local adaptive planner')
    run = Investigation(profile_dataset(df), target, question, mode, budget, engine=engine, model=model)

    def emit(stage: str, message: str) -> None:
        _emit(run, on_event, stage, message)

    emit('Profile', f'Inspected {len(df):,} rows and {len(df.columns)} columns.')
    if engine == 'python':
        # Import lazily: catalog analysis does not require a container runtime.
        from pico.python_agent import python_loop
        python_loop(df, run, emit, executor=executor, planner=planner)
    else:
        _catalog_loop(df, run, emit)
    emit('Complete', run.stop_reason)
    return run
