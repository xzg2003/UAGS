"""Create verified checkpoints for already completed combination IC artifacts.

Usage: python analyze_result/migrate_combination_cache.py --result-root result
       python analyze_result/migrate_combination_cache.py --combination-dir result/csi300/0/combination
"""
import argparse
import json
from pathlib import Path

try:
    from .combination_engine import (cached_completed_k, checkpoint_identity,
                                     completed_k, read_csv, record_completed_k)
except ImportError:
    from combination_engine import (cached_completed_k, checkpoint_identity,
                                    completed_k, read_csv, record_completed_k)


def migrate_existing_ic(output):
    output = Path(output)
    config = json.loads((output / 'combination_config.json').read_text(encoding='utf-8'))
    selected = read_csv(output / 'selected_idx.csv')
    identity = checkpoint_identity(config)
    identities = {identity, checkpoint_identity(config, legacy=True),
                  *config.get('legacy_checkpoint_identities', [])}
    converted, valid = 0, True
    for k in sorted(set(config['factor_counts'])):
        marker = output / f'checkpoint_{k}.json'
        if cached_completed_k(output, k, selected, identities):
            continue
        if not marker.is_file() and completed_k(output, k, selected):
            record_completed_k(output, k, identity)
            converted += 1
        else:
            valid = False
    return converted, valid


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-root', type=Path, default=Path(__file__).resolve().parents[1] / 'result')
    parser.add_argument('--combination-dir', type=Path, nargs='+')
    args = parser.parse_args(argv)
    directories = args.combination_dir or sorted(args.result_root.glob('*/*/combination'))
    if not directories:
        parser.error('No combination directories found')
    incomplete = []
    for directory in directories:
        converted, valid = migrate_existing_ic(directory)
        print(f'{directory}: converted={converted}, complete={valid}', flush=True)
        if not valid:
            incomplete.append(str(directory))
    if incomplete:
        parser.exit(1, 'Incomplete or invalid combinations: ' + ', '.join(incomplete) + '\n')


if __name__ == '__main__':
    main()
