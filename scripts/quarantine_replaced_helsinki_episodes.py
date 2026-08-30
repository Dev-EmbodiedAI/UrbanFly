#!/usr/bin/env python3
"""Move explicitly superseded Helsinki HDF5 pairs into a hashed quarantine."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pre-qa', type=Path, required=True)
    parser.add_argument('--quarantine-dir', type=Path, required=True)
    parser.add_argument('--allowed-source-root', type=Path, action='append', required=True)
    args = parser.parse_args()

    pre_qa = args.pre_qa.resolve(strict=True)
    report = json.loads(pre_qa.read_text(encoding='utf-8'))
    if report.get('status') != 'PASS' or not all(report.get('gate_checks', {}).values()):
        raise RuntimeError('Pre-quarantine replacement QA is not a full PASS')
    replacements = report.get('replacement_selection', {}).get('replacements', [])
    if not replacements:
        raise RuntimeError('Pre-quarantine report contains no explicit replacements')

    allowed_roots = [root.resolve(strict=True) for root in args.allowed_source_root]
    quarantine = args.quarantine_dir.resolve()
    if quarantine.exists():
        raise FileExistsError(quarantine)
    quarantine.parent.mkdir(parents=True, exist_ok=True)
    quarantine.mkdir()

    entries = []
    for replacement in replacements:
        h5_path = Path(replacement['old_path']).resolve(strict=True)
        source_root = next((root for root in allowed_roots if h5_path.parent == root), None)
        if source_root is None:
            raise RuntimeError(f'Old HDF5 is outside an exact allowed source directory: {h5_path}')
        replacement_path = Path(replacement['replacement_path']).resolve(strict=True)
        if replacement_path.suffix.lower() != '.h5':
            raise RuntimeError(f'Invalid replacement HDF5 path: {replacement_path}')
        for source in (h5_path, h5_path.with_suffix('.metadata.json')):
            source = source.resolve(strict=True)
            if source.parent != source_root:
                raise RuntimeError(f'Source escaped allowed directory: {source}')
            destination = quarantine / source_root.name / source.name
            entries.append({
                'episode_index': replacement['episode_index'],
                'source': str(source),
                'destination': str(destination),
                'size_bytes': source.stat().st_size,
                'sha256': sha256(source),
                'replacement_hdf5': str(replacement_path),
            })

    manifest_path = quarantine / 'quarantine_manifest.json'
    manifest = {
        'schema': 'urbanfly-helsinki-replaced-quarantine-v1',
        'status': 'PREPARED',
        'created_unix_s': time.time(),
        'pre_quarantine_qa': str(pre_qa),
        'episode_count': len(replacements),
        'file_count': len(entries),
        'entries': entries,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    try:
        for entry in entries:
            source = Path(entry['source'])
            destination = Path(entry['destination'])
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
            if source.exists() or not destination.is_file():
                raise RuntimeError(f'Move verification failed: {source}')
            if destination.stat().st_size != entry['size_bytes'] or sha256(destination) != entry['sha256']:
                raise RuntimeError(f'Hash verification failed: {destination}')
        manifest['status'] = 'COMPLETE'
        manifest['completed_unix_s'] = time.time()
    except Exception as exc:
        manifest['status'] = 'FAILED'
        manifest['failure'] = repr(exc)
        manifest['failed_unix_s'] = time.time()
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        raise
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({
        'status': manifest['status'],
        'episode_count': manifest['episode_count'],
        'file_count': manifest['file_count'],
        'quarantine_dir': str(quarantine),
        'manifest': str(manifest_path),
    }, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
