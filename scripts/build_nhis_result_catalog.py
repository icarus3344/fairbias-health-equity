"""Build a fresh, metadata-only result catalog; never select or evaluate models."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from nhis_fairbias.benchmark.result_catalog import build_catalog, CatalogError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output must be a fresh directory')
    try:
        catalog = build_catalog(args.manifest)
    except CatalogError as exc:
        print(json.dumps({'status': 'MANIFEST_REJECTED', 'code': str(exc), 'evaluation_authorized': False}))
        return 2
    target = args.output.resolve()
    for row in catalog['jobs']:
        run = Path(row['directory']).parent.parent.resolve()
        if target == run or target.is_relative_to(run) or run.is_relative_to(target):
            parser.error('Catalog output must be independent of every declared input run')
    args.output.mkdir(parents=True, exist_ok=False)
    for name, value in [('catalog.json', catalog), ('summary.json', catalog['summary']), ('issues.json', catalog['issues'])]:
        with (args.output / name).open('x') as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write('\n')
    with (args.output / 'jobs.jsonl').open('x') as handle:
        for row in catalog['jobs']:
            handle.write(json.dumps(row, allow_nan=False) + '\n')
    print(json.dumps({k: catalog[k] for k in ('integrity_passed', 'evaluation_authorized', 'unique_files_hashed')}))
    return 0 if catalog['integrity_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
