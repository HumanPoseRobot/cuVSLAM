#!/usr/bin/env python3
"""Run the exact qualified rc3 core through its immutable dataset entry point."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from sealing import verify

VERSION = 'runtime_reduced_v1.37-rc3'
CORE = '6c19a32afd35d42fc7bdfd4b9554fb10141afb5bf131e77fa3e4826c6b657970'
DEFAULT_PACKAGE = '/nas/shared_halo/baidx/cuVSLAM/cuvslam_runs/correctness_candidate_20261126'


def qualified_package(package):
    manifest = verify(package)
    if manifest['metadata']['core_sha256'] != CORE:
        raise ValueError('Package core differs from the six-room-qualified rc3 snapshot')
    expected = json.loads(Path(__file__).with_name('profiles.json').read_text())
    actual = json.loads((package / 'profiles.json').read_text())
    for name, spec in expected.items():
        def portable(flags):
            # The existing entry resolves this library from its sealed vendor directory.
            return {k: v for k, v in flags.items() if k != 'VIO_JOINT_BLAS_LIBRARY'}
        if portable(spec['flags']) != portable(actual[name]['flags']):
            raise ValueError('Package profile differs from the qualified rc3 profile: ' + name)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', action='version', version=VERSION)
    parser.add_argument('--package', type=Path, default=Path(DEFAULT_PACKAGE))
    parser.add_argument('--profile', choices=['correctness_base', 'observed_history'], default='correctness_base')
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('replay_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    manifest = qualified_package(args.package.resolve())
    if args.verify_only:
        print(json.dumps(dict(version=VERSION, core_sha256=CORE, verified=True,
                              profile=args.profile, channel='release-candidate'), indent=2))
        return 0
    rest = args.replay_args[1:] if args.replay_args[:1] == ['--'] else args.replay_args
    if not rest:
        parser.error('pass dataset replay arguments after --; use -- tumvi --help')
    return subprocess.call([sys.executable, str(args.package.resolve() / 'bin/run.py'),
                            '--profile', args.profile, '--', *rest])


if __name__ == '__main__':
    raise SystemExit(main())
