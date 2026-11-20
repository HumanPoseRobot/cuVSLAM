# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA software released under the NVIDIA Community License is intended to be used to enable
# the further development of AI and robotics technologies. Such software has been designed, tested,
# and optimized for use with NVIDIA hardware, and this License grants permission to use the software
# solely with such hardware.
# Subject to the terms of this License, NVIDIA confirms that you are free to commercially use,
# modify, and distribute the software with NVIDIA hardware. NVIDIA does not claim ownership of any
# outputs generated using the software or derivative works thereof. Any code contributions that you
# share with NVIDIA are licensed to NVIDIA as feedback under this License and may be incorporated
# in future releases without notice or attribution.
# By using, reproducing, modifying, distributing, performing, or displaying any portion or element
# of the software or derivative works thereof, you agree to be bound by this License.

"""HALO release entry: decide cam2/calibration orientation before importing CUDA."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

from camera_orientation import discover, inspect, OrientationError


def normalize(packs, target):
    target.mkdir()
    (target/'calibration').symlink_to(packs[0]['caldir'], target_is_directory=True)
    for spec in packs:
        if (spec['caldir']/'camera_imu.yaml').read_bytes() != (packs[0]['caldir']/'camera_imu.yaml').read_bytes():
            raise OrientationError('Calibration changes between packs; split the recording at that boundary')
        pid = spec['pid']; pack = target/pid; pack.mkdir()
        for c in range(4):
            (pack/f'cam{c}_{pid}.mp4').symlink_to(spec['videos'][c])
        (pack/f'imu_{pid}.bin').symlink_to(spec['imu'])
        session = dict(spec['session'])
        session.setdefault('camera_topology', [{'id': c, 'software_rotate_deg': 0} for c in range(4)])
        (pack/f'sess_{pid}.json').write_text(json.dumps(session, indent=2))
        with spec['csv'].open() as f:
            reader = csv.DictReader(x for x in f if not x.startswith('#')); fields = reader.fieldnames; rows = list(reader)
        if 'capture_unix_ns_raw' in fields:
            (pack/f'frames_{pid}.csv').symlink_to(spec['csv'])
        else:
            import cv2
            cap = cv2.VideoCapture(str(spec['videos'][2])); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
            bycam = {c: [x for x in rows if int(x['cam_id']) == c] for c in range(4)}
            if len(bycam[2]) < n:
                raise OrientationError('Legacy cam2 nominal time grid is incomplete')
            clock = [int(x['unix_ns']) for x in bycam[2][:n]]
            if any(b <= a for a, b in zip(clock, clock[1:])):
                raise OrientationError('Legacy cam2 grid is not increasing')
            with (pack/f'frames_{pid}.csv').open('w') as f:
                f.write('# Legacy shared cam2 nominal grid; no raw capture time was recorded.\n')
                w = csv.writer(f); w.writerow(['cam_id', 'sync_group_id', 'capture_unix_ns_raw', 'source_unix_ns'])
                for i, stamp in enumerate(clock):
                    for c in range(4):
                        w.writerow([c, i+1, stamp, bycam[c][i]['unix_ns'] if i < len(bycam[c]) else ''])
            (pack/'legacy_time_audit.json').write_text(json.dumps(dict(
                timestamp_source='legacy_cam2_nominal_grid_not_raw_capture', source_csv=str(spec['csv']),
                source_counts={str(c): len(x) for c, x in bycam.items()}, video_frames=n), indent=2))
    return target



def release_environment():
    env = {k: v for k, v in os.environ.items() if not k.startswith(('CUVSLAM_', 'PGO_', 'VIO_'))}
    env.update(BUILD_KIND='release', SLAM_ENABLED='0', VIO_COMPARE_SOLVER='schur_fp64',
               CUVSLAM_ONLINE_TEMPORAL='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               VIO_PNP_WEIGHTED='1', VIO_PIXEL_GEOMETRY='1', VIO_VISUAL_GYRO_SEED='1',
               VIO_BOOTSTRAP_STEREO_REFRESH='1', VIO_BOOTSTRAP_TRANSACTION='1',
               VIO_INIT_VISUAL_CHECK='enforce', VIO_BOOTSTRAP_RETAIN_MAP='1')
    return env

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--version',action='version',version='runtime_reduced_v1.36')
    parser.add_argument('--time-profile',help='Explicit validated timing profile name from time_profiles.json')
    parser.add_argument('--sequence', help='Recording prefix when data-root contains several recordings')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--audit', action='store_true', help='Record per-frame solver and state diagnostics')
    parser.add_argument('--camera-time-offsets', type=float, nargs=4, help='Explicit physical cam0..3 offsets; default: YAML')
    parser.add_argument('--imu-timestamp-shift-ms', type=float, default=0.)
    parser.add_argument('--max-frames', type=int, default=0, help='Diagnostic prefix only; default runs the complete recording')
    args = parser.parse_args()
    if args.time_profile:
        profiles=json.loads(Path(__file__).with_name('time_profiles.json').read_text())
        if args.time_profile not in profiles or args.camera_time_offsets is not None:
            parser.error('Unknown time profile or conflicting --camera-time-offsets')
        args.camera_time_offsets=profiles[args.time_profile]['camera_time_offsets_s']
    out = args.output_root.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error('Output directory is not empty; choose a new directory to preserve previous results')
    out.mkdir(parents=True, exist_ok=True)
    try:
        packs = discover(args.data_root, args.sequence)
        report = inspect(packs)
        (out/'orientation_audit.json').write_text(json.dumps(report, indent=2))
        if not report['accepted']:
            raise OrientationError('Cam2 orientation is undetermined; VIO was not started. See orientation_audit.json')
        rotation = report['cam2_calibration_rotation_deg']
        print(f'CAM2_CHECK rotation={rotation} verdict={"keep" if rotation == 0 else "compensate180"}', flush=True)
        if args.check_only:
            return 0
        prepared = normalize(packs, out/'input')
        env = release_environment()
        cmd = [sys.executable, str(Path(__file__).with_name('replay.py')), '--data-root', str(prepared),
               '--output-root', str(out/'vio'), '--orientation-audit',str(out/'orientation_audit.json'), '--foreground-mask', 'none', '--cam2-rotation-deg', str(rotation),
               '--imu-timestamp-shift-ms', str(args.imu_timestamp_shift_ms)]
        if args.audit:
            cmd += ['--audit']
        if args.camera_time_offsets is not None:
            cmd += ['--camera-time-offsets', *map(str, args.camera_time_offsets)]
        if args.max_frames:
            cmd += ['--max-frames', str(args.max_frames)]
        (out/'execution.json').write_text(json.dumps(dict(command=cmd, cam2_rotation_deg=rotation,
            camera_time_offset_policy=('profile:'+args.time_profile) if args.time_profile else ('explicit' if args.camera_time_offsets is not None else 'calibration_yaml'),
            imu_timestamp_shift_ms=args.imu_timestamp_shift_ms, slam_enabled=False,
            foreground_mask='none', extra_spatial_crop=False, active_options={k:v for k,v in env.items() if k.startswith(('CUVSLAM_', 'VIO_'))}), indent=2))
        with (out/'run.log').open('w') as log:
            return subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    except (OrientationError, ValueError, OSError) as exc:
        (out/'error.json').write_text(json.dumps(dict(error=str(exc), vio_started=(out/'execution.json').exists()), indent=2))
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
