"""Shared analysis pipeline modeled on UDFS_final/optimize_gpu.py.

Each project keeps a local copy and supplies its own expression/data adapter.
Selection uses validation IC and training correlations; fitting uses training SSE.
"""
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ALGORITHM = 'udfs-validation-greedy-lbfgsb-v1'


def digest(path):
    state = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            state.update(block)
    return state.hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def add_combination_arguments(parser):
    parser.add_argument('--n-factors', type=int, nargs='+', default=[1, 10, 20, 50, 100])
    parser.add_argument('--min-valid-ic', type=float, default=0.01)
    parser.add_argument('--corr-threshold', type=float, default=0.7)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--selection-mode', choices=('reuse', 'recompute'), default='reuse',
                        help='Reuse selected_idx only when input, settings, code and data match')
    parser.add_argument('--optimizer-maxiter', type=int, default=15000)
    parser.add_argument('--optimizer-maxfun', type=int, default=15000)
    parser.add_argument('--optimizer-ftol', type=float, default=2.220446049250313e-9)
    parser.add_argument('--optimizer-gtol', type=float, default=1e-5)


def validate_arguments(args, parser):
    if min(args.n_factors) < 1:
        parser.error('--n-factors must be positive')
    if not math.isfinite(args.min_valid_ic) or args.min_valid_ic < 0:
        parser.error('--min-valid-ic must be finite and nonnegative')
    if not math.isfinite(args.corr_threshold) or not 0 <= args.corr_threshold <= 1:
        parser.error('--corr-threshold must be between 0 and 1')
    if min(args.optimizer_maxiter, args.optimizer_maxfun) < 1:
        parser.error('Optimizer iteration/evaluation limits must be positive')
    if any(not math.isfinite(v) or v < 0 for v in (args.optimizer_ftol, args.optimizer_gtol)):
        parser.error('Optimizer tolerances must be finite and nonnegative')


def array_backend(device):
    if device == 'cpu':
        return np
    try:
        import cupy as cp
        if not cp.cuda.is_available():
            raise RuntimeError('No CUDA device')
        cp.zeros(1).sum().item()
    except Exception as error:
        raise RuntimeError('GPU fitting requires working CuPy/CUDA; use --device cpu for CPU L-BFGS-B') from error
    return cp


def fit_linear(x, y, args):
    """Same SSE, analytical gradient, zero initialization and L-BFGS-B as UDFS."""
    from scipy.optimize import minimize
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.ndim != 2 or not len(y) or x.shape[0] != len(y) or not x.shape[1]:
        raise ValueError('Empty or misaligned regression inputs')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Nonfinite regression inputs')
    xp = array_backend(args.device)
    matrix, target = xp.asarray(x), xp.asarray(y)

    def loss(weights):
        residual = matrix @ xp.asarray(weights) - target
        return float(xp.dot(residual, residual))

    def gradient(weights):
        residual = matrix @ xp.asarray(weights) - target
        result = 2 * matrix.T @ residual
        return np.asarray(result) if args.device == 'cpu' else xp.asnumpy(result)

    options = dict(maxiter=args.optimizer_maxiter, maxfun=args.optimizer_maxfun,
                   ftol=args.optimizer_ftol, gtol=args.optimizer_gtol)
    result = minimize(loss, np.zeros(x.shape[1]), jac=gradient, method='L-BFGS-B', options=options)
    final_loss = float(result.fun)
    grad_norm = float(np.max(np.abs(result.jac)))
    diagnostics = dict(solver='scipy.optimize.minimize/L-BFGS-B', device=args.device,
                       objective='sum((X @ w - y) ** 2)', initialization='zeros',
                       success=bool(result.success), status=int(result.status), message=str(result.message),
                       iterations=int(result.nit), function_evaluations=int(result.nfev),
                       loss=final_loss if math.isfinite(final_loss) else None,
                       gradient_inf_norm=grad_norm if math.isfinite(grad_norm) else None,
                       finite_weights=bool(np.isfinite(result.x).all()), options=options)
    return np.asarray(result.x), diagnostics


def ranked_candidates(rows, min_valid_ic):
    """Stable ranking, signed IC preserved in outputs; abs(IC) used only for selection."""
    ranked, rejected = [], []
    for source_index, original in enumerate(rows):
        row = {**original, 'source_index': source_index}
        try:
            valid_ic = float(row['ic_valid'])
            if row.get('status_valid', 'ok') != 'ok' or not math.isfinite(valid_ic):
                raise ValueError('Invalid validation IC')
        except (KeyError, TypeError, ValueError):
            rejected.append({**row, 'status': 'invalid_validation_ic'})
            continue
        if abs(valid_ic) <= min_valid_ic:
            rejected.append({**row, 'status': 'below_valid_threshold'})
        else:
            ranked.append({**row, 'ic_valid': valid_ic, 'status': 'eligible'})
    ranked.sort(key=lambda row: -abs(row['ic_valid']))
    for index, row in enumerate(ranked):
        row['selected_idx'] = index
    return ranked, rejected


INDEX_FIELDS = ['selected_idx', 'source_index', 'candidate_id', 'expr_id', 'expr']
FACTOR_FIELDS = ['source_index', 'selected_idx', 'candidate_id', 'expr_id', 'expr',
                 'ic_train', 'ic_valid', 'status', 'error']
ERROR_FIELDS = ['phase', 'expr_id', 'expr', 'error']
SPLITS = ('train', 'valid', 'test')
METRICS = ('ic', 'ric', 'icir', 'ricir')


def checkpoint_contract(config, legacy=False):
    """Stable identity for an existing combination, including legacy outputs."""
    sources = config.get('sources_sha256', {})
    return {key: config[key] for key in (
        'algorithm', 'method', 'instruments', 'seed', 'ic_source',
        'arguments', 'data_signature', 'min_valid_ic',
        'corr_threshold', 'device', 'optimizer') if key in config} | {
        'native_sources_sha256': {key: value for key, value in sources.items()
                                  if not key.startswith('analyze_result/')}
    } | ({'factor_counts': config['factor_counts']} if legacy else {})


def checkpoint_identity(config, legacy=False):
    return hashlib.sha256(json.dumps(checkpoint_contract(config, legacy), sort_keys=True,
                                     allow_nan=False).encode()).hexdigest()


def selection_identity(contract):
    """Ignore analysis code hashes while retaining data, parameters and ranking."""
    stable = json.loads(json.dumps(contract))
    context = stable.get('context', {})
    sources = context.get('sources_sha256', {})
    context['sources_sha256'] = {key: value for key, value in sources.items()
                                 if not key.startswith('analyze_result/')}
    return hashlib.sha256(json.dumps(stable, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def fit_files(output, k):
    return [output / f'optimizer_{k}.json', output / f'selected_factors_{k}.csv']


def result_files(output, k):
    return [output / f'metrics_{k}.json', output / f'daily_metrics_{k}.csv',
            output / f'ic_{k}.csv']


def load_fit(output, k, selected):
    optimizer_path, factors_path = fit_files(output, k)
    if not optimizer_path.is_file() or not factors_path.is_file():
        return None
    try:
        for marker in (output / f'fit_checkpoint_{k}.json', output / f'checkpoint_{k}.json'):
            if marker.is_file():
                saved = json.loads(marker.read_text(encoding='utf-8'))
                if any(saved['files_sha256'][path.name] != digest(path)
                       for path in (optimizer_path, factors_path)):
                    return None
        optimizer = json.loads(optimizer_path.read_text(encoding='utf-8'))
        factors = read_csv(factors_path)
        n = min(k, len(selected))
        if not optimizer['success'] or not optimizer['finite_weights'] or optimizer['loss'] is None:
            return None
        if len(factors) != n or any(
                row['candidate_id'] != expected['candidate_id'] or
                row['expr'] != expected['expr']
                for row, expected in zip(factors, selected[:n])):
            return None
        weights = np.asarray([float(row['weight']) for row in factors])
        if not np.isfinite(weights).all():
            return None
        return weights, optimizer
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def completed_k(output, k, selected):
    if load_fit(output, k, selected) is None:
        return False
    metrics_path, daily_path, ic_path = result_files(output, k)
    if not all(path.is_file() for path in (metrics_path, daily_path, ic_path)):
        return False
    try:
        metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
        daily, ic = read_csv(daily_path), read_csv(ic_path)
        if int(metrics['requested_k']) != k or int(metrics['actual_k']) != min(k, len(selected)):
            return False
        if any(not math.isfinite(float(metrics[f'{split}_{key}']))
               for split in SPLITS for key in ('ic', 'ric')):
            return False
        if len(ic) != 1 or int(ic[0]['actual_k']) != int(metrics['actual_k']):
            return False
        if any(float(ic[0][key]) != float(metrics[f'test_{key}']) for key in ('ic', 'ric')):
            return False
        if not daily or {row['split'] for row in daily} != set(SPLITS):
            return False
        return True
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def record_completed_k(output, k, identity):
    files = fit_files(output, k) + result_files(output, k)
    write_json(output / f'checkpoint_{k}.json', {
        'contract_sha256': identity,
        'files_sha256': {path.name: digest(path) for path in files},
    })


def record_fit(output, k, identity):
    files = fit_files(output, k)
    write_json(output / f'fit_checkpoint_{k}.json', {
        'contract_sha256': identity,
        'files_sha256': {path.name: digest(path) for path in files},
    })


def cached_completed_k(output, k, selected, identities):
    """Only reuse a verified checkpoint; legacy IC migration is a separate command."""
    marker = output / f'checkpoint_{k}.json'
    files = fit_files(output, k) + result_files(output, k)
    if marker.is_file():
        try:
            saved = json.loads(marker.read_text(encoding='utf-8'))
            return (saved['contract_sha256'] in identities and
                    all(saved['files_sha256'][path.name] == digest(path) for path in files)
                    and completed_k(output, k, selected))
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return False
    return False


def restore_selection(output, candidates):
    path = output / 'selected_idx.csv'
    if not path.is_file():
        return None
    try:
        stored = read_csv(path)
        by_id = {row['candidate_id']: row for row in candidates}
        if not stored or len(stored) != len({row['candidate_id'] for row in stored}):
            return None
        selected = []
        for row in stored:
            candidate = by_id[row['candidate_id']]
            if any(str(candidate.get(key, '')) != row[key] for key in INDEX_FIELDS):
                return None
            candidate['status'] = 'selected'
            selected.append(candidate)
        return selected
    except (OSError, ValueError, KeyError):
        return None


def phase_path(output, split, k):
    return output / f'checkpoint_{split}_{k}.json'


def load_phase(output, split, k, identities, factor_digest):
    path = phase_path(output, split, k)
    if not path.is_file():
        return None
    try:
        state = json.loads(path.read_text(encoding='utf-8'))
        if (state['contract_sha256'] not in identities or
                state['selected_factors_sha256'] != factor_digest or
                not all(key in state['metrics'] for key in METRICS) or
                not isinstance(state['daily'], list)):
            return None
        return state['metrics'], state['daily']
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def choose_factors(candidates, train, maximum, threshold, cache_dir, contract, mode):
    cache_dir = Path(cache_dir)
    metadata_path, indices_path = cache_dir / 'selection_metadata.json', cache_dir / 'selected_idx.csv'
    progress_path = cache_dir / 'selection_progress.json'
    cache_note = 'recompute requested' if mode == 'recompute' else 'no matching cache'
    identity = selection_identity(contract)
    allowed_identities = {identity}
    if mode == 'reuse' and metadata_path.is_file():
        try:
            old_metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
            if selection_identity(old_metadata['contract']) == identity:
                allowed_identities.add(old_metadata['contract_sha256'])
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            pass
    cached_selection = None
    if mode == 'reuse' and metadata_path.is_file() and indices_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text())
            if metadata['contract_sha256'] not in allowed_identities:
                raise ValueError('input/settings/code/data changed')
            if metadata['selected_idx_sha256'] != digest(indices_path):
                raise ValueError('selected_idx checksum mismatch')
            if metadata['selected_count'] < maximum and not metadata['exhausted']:
                raise ValueError('cached selection is too short for requested K')
            with indices_path.open(newline='', encoding='utf-8') as handle:
                indices = list(csv.DictReader(handle))
            if len(indices) != metadata['selected_count']:
                raise ValueError('cached selection count mismatch')
            cached_selection, used = [], set()
            for stored in indices:
                index = int(stored['selected_idx'])
                if index in used or not 0 <= index < len(candidates):
                    raise ValueError('invalid cached candidate index')
                row = candidates[index]
                if any(str(row.get(key, '')) != stored.get(key, '') for key in INDEX_FIELDS):
                    raise ValueError('cached expression identity mismatch')
                used.add(index)
                cached_selection.append(row)
            if not cached_selection:
                raise ValueError('empty selection cache')
            cached_selection = cached_selection[:maximum]
        except (OSError, ValueError, KeyError, TypeError) as error:
            cache_note, cached_selection = str(error), None
    if cached_selection is not None:
        # Evaluation errors must not silently turn a cached selection into a different model.
        columns = [train.factor(row['expr']) for row in cached_selection]
        for row in cached_selection:
            row['status'] = 'selected'
        print(f'Reusing {len(cached_selection)} selected factors from {cache_dir}', flush=True)
        return cached_selection, columns, [], dict(reused=True, cache=str(cache_dir), contract_sha256=identity)

    selected, columns, errors, examined = [], [], [], 0
    if mode == 'reuse' and progress_path.is_file():
        try:
            progress = json.loads(progress_path.read_text(encoding='utf-8'))
            examined = int(progress['examined'])
            indices = [int(index) for index in progress['selected_indices']]
            if (progress['contract_sha256'] not in allowed_identities or
                    not 0 <= examined <= len(candidates) or
                    len(progress['statuses']) != examined or
                    len(indices) != len(set(indices)) or
                    any(not 0 <= index < examined for index in indices)):
                raise ValueError('invalid selection progress')
            for row, state in zip(candidates[:examined], progress['statuses']):
                row['status'], row['error'] = state
            if any(candidates[index]['status'] != 'selected' for index in indices):
                raise ValueError('selected identities differ')
            selected = [candidates[index] for index in indices]
            columns = [train.factor(row['expr']) for row in selected]
            errors = progress['errors']
            cache_note = f'resuming after {examined} candidates'
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            selected, columns, errors, examined = [], [], [], 0
            cache_note = 'invalid selection progress; starting over'
    print('Selecting factors: ' + cache_note, flush=True)
    for row in candidates[examined:]:
        if len(selected) >= maximum:
            break
        examined += 1
        try:
            value = train.factor(row['expr'])
            constant = not np.any(np.std(value, axis=1) > 1e-12) if np.ndim(value) == 2 else not np.any(value)
            if not np.isfinite(value).all() or constant:
                raise ValueError('Invalid or all-zero training factor')
            correlations = [train.corr(value, old) for old in columns]
            if not np.isfinite(correlations).all():
                raise ValueError('Nonfinite training correlation')
            if any(abs(correlation) > threshold for correlation in correlations):
                row['status'] = 'correlated'
            else:
                row['status'] = 'selected'
                selected.append(row)
                columns.append(value)
                print(f"Selected candidate {row['selected_idx']}, total: {len(selected)}", flush=True)
        except Exception as error:
            row.update(status='selection_error', error=str(error))
            errors.append(dict(phase='train_selection', expr_id=row['expr_id'], expr=row['expr'], error=str(error)))
        # Persist after every candidate, including rejected and failed candidates.
        write_json(progress_path, dict(contract_sha256=identity, maximum=maximum,
                   examined=examined,
                   selected_indices=[row['selected_idx'] for row in selected],
                   statuses=[(row.get('status', ''), row.get('error', ''))
                             for row in candidates[:examined]], errors=errors))
    write_json(progress_path, dict(contract_sha256=identity, maximum=maximum,
               examined=examined, selected_indices=[row['selected_idx'] for row in selected],
               statuses=[(row.get('status', ''), row.get('error', ''))
                         for row in candidates[:examined]], errors=errors))
    if selected:
        write_csv(indices_path, selected, INDEX_FIELDS)
        write_json(metadata_path, dict(contract=contract, contract_sha256=identity,
                   selected_idx_sha256=digest(indices_path), max_factors=maximum,
                   exhausted=examined == len(candidates), selected_count=len(selected)))
    return selected, columns, errors, dict(reused=False, reason=cache_note, cache=str(cache_dir), contract_sha256=identity)


def run_combination(rows, train, valid_factory, test_factory, group_output, args, provenance):
    """Resume a seed from verified selection, fit and split checkpoints."""
    output = Path(group_output) / 'combination'
    output.mkdir(parents=True, exist_ok=True)
    config = dict(algorithm=ALGORITHM, **provenance,
                  selection='abs(validation IC) descending; greedy training mean cross-sectional correlation',
                  test_policy='all K including K=1 selected using validation only; weights saved before test data loaded',
                  fitting='training SSE, zero initialization, L-BFGS-B, no intercept/regularization/bounds',
                  factor_counts=sorted(set(args.n_factors)), min_valid_ic=args.min_valid_ic,
                  corr_threshold=args.corr_threshold, device=args.device,
                  optimizer=dict(maxiter=args.optimizer_maxiter, maxfun=args.optimizer_maxfun,
                                 ftol=args.optimizer_ftol, gtol=args.optimizer_gtol))
    identity = checkpoint_identity(config)
    old_config_path = output / 'combination_config.json'
    identities = {identity}
    if old_config_path.is_file():
        try:
            old_config = json.loads(old_config_path.read_text(encoding='utf-8'))
            if checkpoint_identity(old_config) != identity:
                raise ValueError('input, settings or market data changed')
            identities.add(checkpoint_identity(old_config, legacy=True))
            identities.update(old_config.get('legacy_checkpoint_identities', []))
            config['factor_counts'] = sorted(set(old_config['factor_counts']) | set(args.n_factors))
            config['legacy_checkpoint_identities'] = sorted(identities - {identity})
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError(f'Unusable combination cache: {old_config_path}; use --overwrite') from error
        except ValueError as error:
            raise ValueError(f'{error}: {old_config_path}; use --overwrite') from error
    elif any(output.iterdir()):
        raise ValueError(f'Combination cache has no configuration: {output}; use --overwrite')
    else:
        write_json(old_config_path, config)
    candidates, rejected = ranked_candidates(rows, args.min_valid_ic)
    contract = dict(algorithm=ALGORITHM, context=provenance, min_valid_ic=args.min_valid_ic,
                    corr_threshold=args.corr_threshold,
                    ranked_candidates_sha256=hashlib.sha256(json.dumps(
                        [{key: row.get(key, '') for key in INDEX_FIELDS} for row in candidates],
                        sort_keys=True).encode()).hexdigest())
    selected = restore_selection(output, candidates) if old_config_path.is_file() else None
    if selected is not None and len(selected) < max(args.n_factors):
        selected = None
    if selected is not None:
        columns = [train.factor(row['expr']) for row in selected]
        errors = read_csv(output / 'evaluation_errors.csv') if (output / 'evaluation_errors.csv').is_file() else []
        selection_info = dict(reused=True, cache=str(output / 'selected_idx.csv'),
                              contract_sha256=identity)
        print(f'Resumed {len(selected)} selected factors from {output}', flush=True)
    else:
        selected, columns, errors, selection_info = choose_factors(
            candidates, train, max(args.n_factors), args.corr_threshold,
            Path(group_output) / 'selection_cache', contract, args.selection_mode)
    config['selection_cache'] = selection_info
    write_json(output / 'combination_config.json', config)
    write_csv(output / 'selected_idx.csv', selected, INDEX_FIELDS)
    write_csv(output / 'factor_metrics.csv', sorted(candidates + rejected, key=lambda row: row['source_index']), FACTOR_FIELDS)
    write_csv(output / 'evaluation_errors.csv', errors, ERROR_FIELDS)
    if not selected:
        raise ValueError('No factors passed selection; see ' + str(output / 'factor_metrics.csv'))
    fits, summaries, daily, completed = {}, {}, {}, set()
    for k in sorted(set(args.n_factors)):
        n = min(k, len(selected))
        cached_fit = load_fit(output, k, selected)
        if cached_fit is None and n not in fits:
            print(f'Fitting K={n} with L-BFGS-B on {args.device}', flush=True)
            fits[n] = fit_linear(*train.training_arrays(columns[:n]), args)
        weights, optimizer = cached_fit if cached_fit is not None else fits[n]
        fits[n] = weights, optimizer
        if not optimizer['success'] or not optimizer['finite_weights'] or optimizer['loss'] is None:
            raise RuntimeError('L-BFGS-B did not converge; see ' + str(output / f'optimizer_{k}.json'))
        if cached_fit is None:
            write_json(output / f'optimizer_{k}.json', optimizer)
            write_csv(output / f'selected_factors_{k}.csv',
                      [{**row, 'weight': float(weight)} for row, weight in zip(selected[:n], weights)],
                      ['candidate_id', 'expr_id', 'expr', 'weight', 'ic_train', 'ic_valid'])
            record_fit(output, k, identity)
        summaries[k] = dict(requested_k=k, actual_k=n, input_candidates=len(rows), eligible=len(candidates),
                            solver=optimizer['solver'], optimizer_success=optimizer['success'],
                            optimizer_iterations=optimizer['iterations'], train_loss=optimizer['loss'], output=str(output))
        daily[k] = []
        if cached_completed_k(output, k, selected, identities):
            summaries[k] = json.loads((output / f'metrics_{k}.json').read_text(encoding='utf-8'))
            completed.add(k)
            print(f'K={k}: cached IC and combination reused', flush=True)
    for name, factory in (('train', lambda: train), ('valid', valid_factory), ('test', test_factory)):
        try:
            pending = []
            for k, summary in summaries.items():
                if k in completed:
                    continue
                saved = load_phase(output, name, k, identities, digest(output / f'selected_factors_{k}.csv'))
                if saved is not None:
                    metrics, days = saved
                    summary.update({name + '_' + key: value for key, value in metrics.items()})
                    daily[k].extend({'split': name, **day} for day in days)
                    print(f'K={k} {name}: cached IC reused', flush=True)
                else:
                    pending.append(k)
            if not pending:
                continue
            split = factory()
            values = columns if name == 'train' else [split.factor(row['expr']) for row in selected]
            for k in pending:
                summary = summaries[k]
                n = summary['actual_k']
                weights, _ = fits[n]
                prediction = sum(value * weight for value, weight in zip(values[:n], weights))
                metrics, days = split.metrics(prediction)
                summary.update({name + '_' + key: value for key, value in metrics.items()})
                daily[k].extend({'split': name, **day} for day in days)
                write_json(phase_path(output, name, k), dict(
                    contract_sha256=identity,
                    selected_factors_sha256=digest(output / f'selected_factors_{k}.csv'),
                    metrics=metrics, daily=days))
        except Exception as error:
            errors.append(dict(phase='frozen_' + name, error=str(error)))
            write_csv(output / 'evaluation_errors.csv', errors, ERROR_FIELDS)
            raise RuntimeError('Frozen combination evaluation failed; factors and weights unchanged') from error
    for k, summary in summaries.items():
        if k in completed:
            continue
        write_json(output / f'metrics_{k}.json', summary)
        write_csv(output / f'daily_metrics_{k}.csv', daily[k], ['split', 'date', 'ic', 'ric'])
        write_csv(output / f'ic_{k}.csv', [dict(ic=summary['test_ic'], ric=summary['test_ric'], actual_k=summary['actual_k'])],
                  ['ic', 'ric', 'actual_k'])
        record_completed_k(output, k, identity)
    rows = list(summaries.values())
    for k in config['factor_counts']:
        if k not in summaries and (output / f'metrics_{k}.json').is_file():
            rows.append(json.loads((output / f'metrics_{k}.json').read_text(encoding='utf-8')))
    rows.sort(key=lambda row: row['requested_k'])
    write_csv(output / 'summary.csv', rows, list(rows[0]))
    return list(summaries.values())
