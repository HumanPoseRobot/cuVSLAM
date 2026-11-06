"""Replay the four-fisheye pack_001 format with the optimized rig configuration.

Run with the patched cuVSLAM Python module, numpy, scipy, pyyaml and OpenCV.
Usage: python run.py --data-root SEQUENCE --output-root NEW_DIRECTORY
Time offsets are calibrated for the reference recording, in physical camera order.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import cv2
import cuvslam as v
import numpy as np
import yaml
from scipy.spatial.transform import Rotation, Slerp

CAMERA_ORDER = (2, 3, 0, 1)
TIME_OFFSETS = (-0.0055, -0.004656218695898521, -0.0047981480290696125, -0.004649456023684627)
SIZE = 768
CENTER = (SIZE - 1) / 2
FOCAL = SIZE / (2 * np.tan(np.deg2rad(130) / 2))


def pinhole_maps(camera):
    xi, alpha, fx, fy, cx, cy = camera["intrinsics"]
    width, height = camera["resolution"]
    u, v_coord = np.meshgrid(np.arange(SIZE), np.arange(SIZE))
    x, y = (u - CENTER) / FOCAL, (v_coord - CENTER) / FOCAL
    zz = xi * np.sqrt(x*x + y*y + 1) + 1
    den = alpha * np.sqrt(x*x + y*y + zz*zz) + (1-alpha)*zz
    px, py = fx*x/den + cx, fy*y/den + cy
    valid = (px >= 2) & (px < width-3) & (py >= 2) & (py < height-3) & (den > 1e-9)
    return (np.where(valid, px, -1).astype("float32"), np.where(valid, py, -1).astype("float32"),
            (~valid).astype("uint8") * 255)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--camera-time-offsets", type=float, nargs=4, default=TIME_OFFSETS)
    args = parser.parse_args()
    data, out = args.data_root, args.output_root
    os.environ.update(CUVSLAM_JOINT_VI="1", CUVSLAM_CAM01_TEMPORAL="insert", CUVSLAM_METRIC_INIT="1")
    cv2.setNumThreads(4)
    v.set_verbosity(3 if args.audit else 0)
    cameras_yaml = yaml.safe_load((data / "camera_imu.yaml").read_text())
    imu_yaml = yaml.safe_load((data / "imu.yaml").read_text())
    bias = np.array(yaml.safe_load((data / "imu_bias.yaml").read_text())["gyroscope_bias"])
    with (data / "frames_pack_001.csv").open() as stream:
        rows = list(csv.DictReader(line for line in stream if not line.startswith("#")))
    timestamps = [
        np.array([int(r["capture_unix_ns_raw"]) for r in rows if int(r["cam_id"]) == cam], dtype=np.int64)
        + round(offset * 1e9) for cam, offset in enumerate(args.camera_time_offsets)
    ]
    targets = timestamps[2] + (timestamps[3] - timestamps[2]) // 2
    frames = np.flatnonzero((targets >= timestamps[0][0]) & (targets <= timestamps[0][-1]))
    if args.max_frames:
        frames = frames[:args.max_frames]
    if len(frames) == 0:
        raise ValueError("No frames in the common camera interval")
    dtype = np.dtype(dict(names=["t", "acc", "gyro"], formats=["<i8", ("<f8", 3), ("<f8", 3)],
                          offsets=[0, 48, 72], itemsize=152))
    imu = np.frombuffer((data / "imu_pack_001.bin").read_bytes(), dtype=dtype, offset=72)
    imu = imu[np.linalg.norm(imu["acc"], axis=1) > 1]
    epoch = int(imu["t"][0])
    times = (imu["t"] - epoch) / 1e9
    rotations = [Rotation.identity()]
    for j in range(1, len(imu)):
        omega = (imu["gyro"][j] + imu["gyro"][j-1]) * 0.5 - bias
        rotations.append(rotations[-1] * Rotation.from_rotvec(omega * (times[j] - times[j-1])))
    orientation = Slerp(times, Rotation.from_quat([r.as_quat() for r in rotations]))

    maps = [pinhole_maps(cameras_yaml[f"fisheye{i}"]) for i in range(4)]
    cameras = []
    for cam in CAMERA_ORDER:
        transform = np.linalg.inv(np.array(cameras_yaml[f"fisheye{cam}"]["T_cam_imu"]))
        if cam in (0, 1):
            roll = np.eye(4)
            roll[:3, :3] = Rotation.from_euler("z", 90 if cam == 0 else -90, degrees=True).as_matrix()
            transform = transform @ roll
        pose = v.Pose(rotation=Rotation.from_matrix(transform[:3, :3]).as_quat(), translation=transform[:3, 3])
        cameras.append(v.Camera(size=[SIZE, SIZE], focal=[FOCAL]*2, principal=[CENTER]*2, rig_from_camera=pose))
    noise_keys = ("gyroscope_noise_density", "gyroscope_random_walk",
                  "accelerometer_noise_density", "accelerometer_random_walk")
    imu_cal = v.ImuCalibration(rig_from_imu=v.Pose(), frequency=imu_yaml["update_rate"],
                               **{k: imu_yaml[k] for k in noise_keys})
    config = v.Tracker.OdometryConfig(odometry_mode=v.Tracker.OdometryMode.Multisensor,
                                      rectified_stereo_camera=False, async_sba=False,
                                      use_motion_model=True, enable_observations_export=True)
    tracker = v.Tracker(v.Rig(cameras, [imu_cal]), config)
    expert = json.loads(Path(__file__).with_name("anchored.json").read_text())
    tracker.odom.apply_expert_parameters(expert)
    out.mkdir(parents=True, exist_ok=False)
    captures = [cv2.VideoCapture(str(data / f"fisheye{i}.mp4")) for i in range(4)]
    last_index = [-1]*4
    last_image = [None]*4

    def read_frame(cam, index):
        if index < last_index[cam]:
            raise ValueError(f"Nonmonotonic frame index for camera {cam}")
        while last_index[cam] < index:
            ok, image = captures[cam].read()
            if not ok:
                raise ValueError(f"Missing camera {cam} frame {last_index[cam]+1}")
            last_index[cam] += 1
            last_image[cam] = cv2.remap(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), *maps[cam][:2], cv2.INTER_LINEAR)
        return last_image[cam]

    k = np.array([[FOCAL, 0, CENTER], [0, FOCAL, CENTER], [0, 0, 1.]])
    k_inv = np.linalg.inv(k)
    r_ci = np.array(cameras_yaml["fisheye0"]["T_cam_imu"])[:3, :3]
    y, x = np.ogrid[:SIZE, :SIZE]
    central = (((x-CENTER)**2 + (y-CENTER)**2) > 360**2).astype("uint8")*255
    digest = hashlib.sha256()
    trajectory, diagnostics = [], []
    imu_index = 0
    for frame in frames:
        stamp = int(targets[frame])
        images, masks = [], []
        for cam in CAMERA_ORDER:
            nearest = int(np.argmin(abs(timestamps[cam] - stamp)))
            image, mask = read_frame(cam, nearest), maps[cam][2]
            if cam == 0:
                source_rotation = orientation((int(timestamps[cam][nearest])-epoch)/1e9)
                target_rotation = orientation((stamp-epoch)/1e9)
                rotation = r_ci @ source_rotation.inv().as_matrix() @ target_rotation.as_matrix() @ r_ci.T
                h = k @ rotation @ k_inv
                image = cv2.warpPerspective(image, h, (SIZE, SIZE), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
                mask = 255-cv2.warpPerspective(255-mask, h, (SIZE, SIZE), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)
            if cam in (0, 1):
                roll = cv2.ROTATE_90_COUNTERCLOCKWISE if cam == 0 else cv2.ROTATE_90_CLOCKWISE
                image = cv2.rotate(image, roll)
                mask = cv2.rotate(cv2.bitwise_or(mask, central), roll)
            images.append(image)
            masks.append(mask)
        while imu_index < len(imu) and int(imu["t"][imu_index]) <= stamp:
            row = imu[imu_index]
            digest.update(row.tobytes())
            tracker.register_imu_measurement(0, v.ImuMeasurement(timestamp_ns=int(row["t"])-epoch,
                                             linear_accelerations=row["acc"], angular_velocities=row["gyro"]-bias))
            imu_index += 1
        digest.update(np.int64(stamp).tobytes())
        for array in images + masks:
            digest.update(array.tobytes())
        estimate, _ = tracker.track(stamp-epoch, images, masks=masks)
        pose = estimate.world_from_rig
        values = [*pose.pose.translation, *pose.pose.rotation] if pose is not None else [np.nan]*7
        trajectory.append([*values, *(tracker.get_last_gravity() or [np.nan]*3)])
        state = tracker.odom.get_state()
        diagnostics.append([frame, int(state.keyframe), int(state.warming_up),
                            *(len(tracker.get_last_observations(i)) for i in range(4))])
    for cap in captures:
        cap.release()
    values = np.asarray(trajectory, dtype=float)
    np.savez_compressed(out/"trajectory.npz", frame_index=frames, timestamp_unix_ns=targets[frames],
                        translation_m=values[:, :3], quaternion_xyzw=values[:, 3:7], gravity_rig_m_s2=values[:, 7:])
    with (out/"diagnostics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame", "keyframe", "warming_up", *(f"observations_cam{i}" for i in CAMERA_ORDER)])
        writer.writerows(diagnostics)
    (out/"configs.json").write_text(json.dumps(dict(camera_order=CAMERA_ORDER, camera_time_offsets_s=args.camera_time_offsets,
        time_origin_unix_ns=epoch, input_sha256=digest.hexdigest(), expert_parameters=expert), indent=2))


if __name__ == "__main__":
    main()
