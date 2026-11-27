#!/usr/bin/env python3
"""Rebuild committed rc3 backend with hash-verified frozen static dependencies.

This is an overlay rebuild, not a claim of a from-zero upstream/CUDA build.
All output is written into a new directory outside both source and frozen package.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

from sealing import verify, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen-package', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    package, output = args.frozen_package.resolve(), args.output.resolve()
    manifest = verify(package)
    if output.is_relative_to(root) or output.is_relative_to(package):
        parser.error('build output must be outside source and immutable package')
    # Refuse overwrites before any compilation or link action.
    output.mkdir(parents=True, exist_ok=False)
    recipe = json.loads((package / 'provenance/rebuild_recipe.json').read_text())
    for path, expected in recipe['external_files'].items():
        if digest(path) != expected:
            raise ValueError('External build dependency changed: ' + path)
    source = root / 'libs/pipelines/gpu_window'
    overlay = json.loads((package / 'provenance/BACKEND_BUILD.json').read_text())['sources']
    for path, expected in overlay.items():
        if digest(source / Path(path).name) != expected:
            raise ValueError('rc3 backend differs from qualified source: ' + path)
    def expand(value):
        return value.replace('@PACKAGE@', str(package))
    objects = []
    for name in ['joint_posterior.cpp', 'history_archive.cpp', 'joint_online.cpp']:
        cmd = [expand(x) for x in recipe['compile'] if x not in ['-fPIE', '-fpie']]
        obj = output / (name + '.o')
        for i, item in enumerate(cmd):
            if item in ['-o', '-MF', '-MT']:
                cmd[i + 1] = str(obj) + ('.d' if item == '-MF' else '')
        cmd[-1] = str(source / name)
        cmd[1:1] = ['-I' + str(source), '-I' + str(root / 'libs')]
        with (output / (name + '.log')).open('x') as log:
            subprocess.run(cmd, cwd=recipe['cwd'], stdout=log, stderr=subprocess.STDOUT, check=True)
        objects.append(str(obj))
    library = output / 'libpipelines.a'
    shutil.copy2(package / 'build_inputs/libpipelines.a', library)
    subprocess.run(['ar', 'r', str(library), *objects], check=True)
    subprocess.run(['ranlib', str(library)], check=True)
    runtime = output / 'runtime'
    shutil.copytree(package / 'runtime', runtime, ignore=shutil.ignore_patterns('libcuvslam.so', '__pycache__'))
    target = runtime / 'cuvslam/libcuvslam.so'
    cmd = [expand(x) for x in recipe['link']]
    cmd = [str(library) if x == str(package / 'build_inputs/libpipelines.a') else x for x in cmd]
    cmd[cmd.index('-o') + 1] = str(target) + '.tmp'
    with (output / 'link.log').open('x') as log:
        subprocess.run(cmd, cwd=recipe['cwd'], stdout=log, stderr=subprocess.STDOUT, check=True)
    os.replace(str(target) + '.tmp', target)
    verify(package)
    commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    provenance = dict(commit=commit, original_core_sha256=manifest['metadata']['core_sha256'],
                      core_sha256=digest(target), sources={p.name: digest(p) for p in source.glob('*') if p.is_file()},
                      scope='committed rc3 backend; sealed static dependency reuse; separate regression and replay required')
    (output / 'BUILD.json').write_text(json.dumps(provenance, indent=2))
    print(json.dumps(dict(core=str(target), **provenance), indent=2))


if __name__ == '__main__':
    main()
