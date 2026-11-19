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

"""Check cam2 against the supplied calibration, independently of firmware flags."""
import copy
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import yaml


class OrientationError(ValueError):
    pass


def rotate_calibration(calibration, degrees):
    """Return a copy in the recorded pixel axes; never modify the input YAML."""
    if degrees not in (0, 180):
        raise ValueError('Only 0 or 180 degree calibration compensation is supported')
    result = copy.deepcopy(calibration)
    if degrees == 180:
        cam = result['fisheye2']
        width, height = cam['resolution']
        cam['intrinsics'][4] = width - 1 - cam['intrinsics'][4]
        cam['intrinsics'][5] = height - 1 - cam['intrinsics'][5]
        cam['T_cam_imu'] = (np.diag([-1., -1., 1., 1.]) @ np.asarray(cam['T_cam_imu'])).tolist()
    return result


def rays(pixels, camera):
    xi, alpha, fx, fy, cx, cy = map(float, camera['intrinsics'])
    if fx <= 0 or fy <= 0 or not 0 < alpha < 1:
        raise OrientationError('Invalid double-sphere intrinsics')
    x, y = ((pixels - [cx, cy]) / [fx, fy]).T
    r2 = x*x + y*y
    discriminant = 1 - (2*alpha-1)*r2
    with np.errstate(invalid='ignore', divide='ignore'):
        mz = (1-alpha*alpha*r2)/(alpha*np.sqrt(discriminant)+1-alpha)
        k = (mz*xi+np.sqrt(mz*mz+(1-xi*xi)*r2))/(mz*mz+r2)
        answer = np.column_stack((k*x, k*y, k*mz-xi))
        answer /= np.linalg.norm(answer, axis=1, keepdims=True)
    return answer


def decide(records):
    """Require aggregate separation and evidence in at least two distinct frames."""
    totals = {r: sum(x['inliers'] for x in records if x['rotation'] == r) for r in (0, 180)}
    winner = max(totals, key=totals.get)
    other = 180-winner
    groups = sorted({x['sample'] for x in records})
    votes = []
    for group in groups:
        scores = {r: sum(x['inliers'] for x in records if x['rotation'] == r and x['sample'] == group)
                  for r in (0, 180)}
        if scores[winner] >= 10 and scores[winner] >= 2*max(1, scores[other]):
            votes.append(group)
    good = totals[winner] >= 30 and totals[winner] >= 2*max(1, totals[other]) and len(votes) >= 2
    return dict(rotation_deg=winner if good else None, totals=totals, independent_frame_votes=votes,
                verdict=('keep' if winner == 0 else 'rotate180') if good else 'undetermined')


def discover(root, sequence=None):
    root = Path(root).resolve()
    if (root/'fisheye2.mp4').exists():
        packs = [root]
    elif (root/'calibration').is_dir():
        packs = sorted(root.glob('pack_*'))
    elif sequence:
        packs = sorted(root.glob(sequence+'_pack_*'))
    else:
        raise OrientationError('Use a recording directory, or --sequence to select one recording in a parent directory')
    result = []
    for pack in packs:
        csvs = list(pack.glob('frames_pack_*.csv'))
        if len(csvs) != 1:
            raise OrientationError(f'{pack}: expected one frames_pack_*.csv')
        pid = csvs[0].stem.removeprefix('frames_')
        caldir = root/'calibration' if (root/'calibration').is_dir() else pack
        videos = {c: pack/(f'cam{c}_{pid}.mp4' if (pack/f'cam{c}_{pid}.mp4').exists() else f'fisheye{c}.mp4')
                  for c in range(4)}
        required = [*videos.values(), caldir/'camera_imu.yaml', caldir/'imu.yaml', caldir/'imu_bias.yaml',
                    pack/f'imu_{pid}.bin']
        if any(not x.is_file() for x in required):
            raise OrientationError(f'{pack}: missing video, IMU or calibration file')
        session_path = pack/f'sess_{pid}.json'
        session = json.loads(session_path.read_text()) if session_path.exists() else {}
        result.append(dict(pack=pack, pid=pid, caldir=caldir, csv=csvs[0], videos=videos,
                           session=session, imu=pack/f'imu_{pid}.bin'))
    if not result:
        raise OrientationError('No recording packs found')
    return result


def inspect(packs):
    cv2.setNumThreads(2)
    sift = cv2.SIFT_create(nfeatures=2500)
    matcher = cv2.BFMatcher()
    audits = []
    for pack in packs:
        calpath = pack['caldir']/'camera_imu.yaml'
        cal = yaml.safe_load(calpath.read_text())
        for c in range(4):
            T = np.asarray(cal[f'fisheye{c}']['T_cam_imu'], dtype=float)
            if (T.shape != (4, 4) or not np.isfinite(T).all() or
                    not np.allclose(T[3], [0, 0, 0, 1]) or
                    not np.allclose(T[:3, :3].T@T[:3, :3], np.eye(3), atol=1e-4) or
                    not np.isclose(np.linalg.det(T[:3, :3]), 1, atol=1e-4)):
                raise OrientationError(f'Camera {c}: invalid SE(3) extrinsic')
        counts = {}
        for c, video in pack['videos'].items():
            cap = cv2.VideoCapture(str(video)); counts[c] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
        if min(counts.values()) < 2 or len(set(counts.values())) != 1:
            raise OrientationError(f'Video counts differ: {counts}')
        with pack['csv'].open() as f:
            reader = csv.DictReader(line for line in f if not line.startswith('#'))
            rows = list(reader)
        bycam = {c: [x for x in rows if int(x['cam_id']) == c] for c in range(4)}
        # Select actual sync groups where available. Legacy hardware recordings use equal video indices;
        # their lack of raw capture time is explicitly recorded, not hidden by invented precision.
        raw = all('capture_unix_ns_raw' in x and 'sync_group_id' in x for x in rows)
        if raw:
            groups = {}
            for c, entries in bycam.items():
                times = [int(x['capture_unix_ns_raw']) for x in entries]
                if len(entries) != counts[c] or any(b <= a for a, b in zip(times, times[1:])):
                    raise OrientationError(f'Camera {c}: timestamp count/order mismatch')
                for i, x in enumerate(entries):
                    g = int(x['sync_group_id'])
                    if c in groups.setdefault(g, {}):
                        raise OrientationError('Duplicate camera in sync group')
                    groups[g][c] = i
            if any(set(g) != {0, 1, 2, 3} for g in groups.values()):
                raise OrientationError('Incomplete sync group')
            allgroups = [groups[g] for g in sorted(groups)]
        else:
            if any(abs(len(bycam[c])-counts[c]) > 1 for c in range(4)):
                raise OrientationError('Legacy timestamp/video count mismatch exceeds one frame')
            allgroups = [{c: i for c in range(4)} for i in range(min(counts.values()))]
        picks = [allgroups[i] for i in np.unique(np.linspace(0, len(allgroups)-1, min(9, len(allgroups)), dtype=int))]
        images = {}
        for c, video in pack['videos'].items():
            selected = {g[c] for g in picks}; cap = cv2.VideoCapture(str(video))
            for i in range(counts[c]):
                if not cap.grab():
                    raise OrientationError(f'{video}: decode failed at {i}')
                if i in selected:
                    ok, im = cap.retrieve()
                    if not ok or tuple(im.shape[:2][::-1]) != tuple(cal[f'fisheye{c}']['resolution']):
                        raise OrientationError(f'{video}: decode/resolution mismatch')
                    images[c, i] = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), None, fx=.5, fy=.5)
            cap.release()
        records = []
        for sample, group in enumerate(picks):
            features = {}
            for c in range(4):
                for rot in ((0, 180) if c == 2 else (0,)):
                    im = images[c, group[c]]
                    if rot:
                        im = cv2.rotate(im, cv2.ROTATE_180)
                    kp, desc = sift.detectAndCompute(im, None)
                    features[c, rot] = (np.array([x.pt for x in kp]).reshape(-1, 2)*2+.5, desc)
            for rot in (0, 180):
                u, du = features[2, rot]
                for c in (0, 1, 3):
                    v, dv = features[c, 0]
                    if du is None or dv is None or len(dv) < 2:
                        continue
                    pairs = [m for candidates in matcher.knnMatch(du, dv, k=2) if len(candidates) == 2
                             for m, n in [candidates] if m.distance < .74*n.distance]
                    if not pairs:
                        continue
                    T = np.asarray(cal[f'fisheye{c}']['T_cam_imu'])@np.linalg.inv(np.asarray(cal['fisheye2']['T_cam_imu']))
                    aa = rays(u[[m.queryIdx for m in pairs]], cal['fisheye2'])@T[:3, :3].T
                    bb = rays(v[[m.trainIdx for m in pairs]], cal[f'fisheye{c}'])
                    normal = np.cross(T[:3, 3], aa); norm = np.linalg.norm(normal, axis=1)
                    good = np.isfinite(aa).all(1)&np.isfinite(bb).all(1)&(norm > 1e-10)
                    error = np.degrees(np.arcsin(np.clip(abs(np.sum(bb[good]*normal[good], axis=1))/norm[good], 0, 1)))
                    records.append(dict(sample=sample, reference=c, rotation=rot, matches=len(error),
                                        inliers=int(np.sum(error < .75)), median_deg=float(np.median(error)) if len(error) else None))
        decision = decide(records)
        audits.append(dict(pack=pack['pid'], calibration_sha256=hashlib.sha256(calpath.read_bytes()).hexdigest(),
                           recorded_rotations={str(x['id']): x.get('software_rotate_deg', 0)
                                               for x in pack['session'].get('camera_topology', [])},
                           timestamp_source='raw_capture' if raw else 'legacy_nominal_grid',
                           frames=counts[2], evidence=records, **decision))
    rotations = {x['rotation_deg'] for x in audits}
    accepted = len(rotations) == 1 and None not in rotations
    return dict(accepted=accepted, cam2_calibration_rotation_deg=next(iter(rotations)) if accepted else None,
                packs=audits, reason='independent geometry consensus' if accepted else 'insufficient or inconsistent geometry')
