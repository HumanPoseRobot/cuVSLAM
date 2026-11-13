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
import time
from pathlib import Path

import cv2
import cuvslam as v
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

CAMERA_ORDER = (2, 3, 0, 1)
TIME_OFFSETS = (-0.0055, -0.004656218695898521, -0.0047981480290696125, -0.004649456023684627)
SIZE = 768
CENTER = (SIZE - 1) / 2
FOCAL = SIZE / (2 * np.tan(np.deg2rad(130) / 2))


FISHEYE_FOCAL=(SIZE/2)/np.deg2rad(65)

def remap_mask(mask):
    u,v=np.meshgrid(np.arange(SIZE),np.arange(SIZE))
    dx,dy=u-CENTER,v-CENTER
    radius=np.hypot(dx,dy)
    angle=radius/FISHEYE_FOCAL
    scale=FOCAL*np.tan(np.minimum(angle,1.56))/np.maximum(radius,1e-12)
    return cv2.remap(mask,(CENTER+dx*scale).astype(np.float32),(CENTER+dy*scale).astype(np.float32),
                     cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=255)

def pinhole_maps(camera):
    xi, alpha, fx, fy, cx, cy = camera["intrinsics"]
    width, height = camera["resolution"]
    u, v_coord = np.meshgrid(np.arange(SIZE), np.arange(SIZE))
    dx,dy=(u-CENTER)/FISHEYE_FOCAL,(v_coord-CENTER)/FISHEYE_FOCAL
    theta=np.hypot(dx,dy)
    x,y=dx*np.sinc(theta/np.pi),dy*np.sinc(theta/np.pi)
    z=np.cos(theta)
    zz = xi + z
    den = alpha * np.sqrt(x*x + y*y + zz*zz) + (1-alpha)*zz
    px, py = fx*x/den + cx, fy*y/den + cy
    valid = (px >= 2) & (px < width-3) & (py >= 2) & (py < height-3) & (den > 1e-9)
    return (np.where(valid, px, -1).astype("float32"), np.where(valid, py, -1).astype("float32"),
            (~valid).astype("uint8") * 255)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--imu-timestamp-shift-ms",type=float,default=0.)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--camera-time-offsets", type=float, nargs=4, default=None)
    parser.add_argument("--apply-recorded-rotation",action="store_true")
    parser.add_argument("--foreground-mask", choices=["none","stereo","all"], default="none")
    args = parser.parse_args()
    data, out = args.data_root, args.output_root
    for key in ("CUVSLAM_JOINT_VI", "CUVSLAM_JOINT_BACKEND", "CUVSLAM_WINDOW_SOLVER", "CUVSLAM_METRIC_INIT", "CUVSLAM_CAM01_TEMPORAL"):
        os.environ.pop(key, None)
    os.environ.update(CUVSLAM_ONLINE_TEMPORAL="1", CUVSLAM_JOINT_VI="1",CUVSLAM_JOINT_BACKEND="window",CUVSLAM_WINDOW_SOLVER="schur_fp64",CUVSLAM_WINDOW_SIZE="4",CUVSLAM_METRIC_INIT="1")
    cv2.setNumThreads(4)
    v.set_verbosity(3 if args.audit else 0)
    packs=sorted(data.glob("pack_*"))
    if not packs:
        raise ValueError("No recording packs")
    calibration=data/"calibration"
    cameras_yaml = yaml.safe_load((calibration / "camera_imu.yaml").read_text())
    rotations_metadata={}
    for pack in packs:
        session=json.loads((pack/f"sess_{pack.name}.json").read_text())
        current={int(c["id"]):int(c.get("software_rotate_deg",0)) for c in session["camera_topology"]}
        if rotations_metadata and current!=rotations_metadata:
            raise ValueError("Camera rotations change across recording packs")
        rotations_metadata=current
    if args.apply_recorded_rotation:
        for i,degrees in rotations_metadata.items():
            if degrees not in (0,180):
                raise ValueError("Only the recorded 0/180 degree formats are supported")
            if degrees==180:
                cam=cameras_yaml[f"fisheye{i}"]
                width,height=cam["resolution"]
                cam["intrinsics"][4]=width-1-cam["intrinsics"][4]
                cam["intrinsics"][5]=height-1-cam["intrinsics"][5]
                rotation=np.diag([-1.,-1.,1.,1.])
                cam["T_cam_imu"]=(rotation@np.array(cam["T_cam_imu"])).tolist()
    args.camera_time_offsets=args.camera_time_offsets or [cameras_yaml[f"fisheye{i}"]["timeshift_cam_imu"] for i in range(4)]
    rows=[]
    pack_audit=[]
    for pack in packs:
        with (pack/f"frames_{pack.name}.csv").open() as stream:
            current=list(csv.DictReader(line for line in stream if not line.startswith("#")))
        counts=[]
        for cam in range(4):
            cap=cv2.VideoCapture(str(pack/f"cam{cam}_{pack.name}.mp4"))
            count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
            camera_rows=[r for r in current if int(r["cam_id"])==cam]
            assert count==len(camera_rows),(pack,cam,count,len(camera_rows))
            assert count>0
            counts.append(count)
        assert len(set(counts))==1
        pack_audit.append(dict(pack=pack.name,frames=counts[0],start_index=len(rows)//4))
        rows.extend(current)
    timestamps = [
        np.array([int(r["capture_unix_ns_raw"]) for r in rows if int(r["cam_id"]) == cam], dtype=np.int64)
        + round(offset * 1e9) for cam, offset in enumerate(args.camera_time_offsets)
    ]
    assert all(np.all(np.diff(t)>0) for t in timestamps)
    targets = timestamps[2] + (timestamps[3] - timestamps[2]) // 2
    frames = np.flatnonzero((targets >= max(t[0] for t in timestamps)) & (targets <= min(t[-1] for t in timestamps)))
    if args.max_frames:
        frames = frames[:args.max_frames]
    if len(frames) == 0:
        raise ValueError("No frames in the common camera interval")
    # Preserve the pure-visual frame selection and images exactly; add inertial constraints only.
    dtype=np.dtype(dict(names=["t","acc","gyro"],formats=["<i8",("<f8",3),("<f8",3)],offsets=[0,48,72],itemsize=152))
    chunks=[]
    for pack in packs:
        raw=(pack/f"imu_{pack.name}.bin").read_bytes()
        assert raw[:8]==b"RKIMU002" and int.from_bytes(raw[12:16],"little")==152
        chunks.append(np.frombuffer(raw,dtype=dtype,offset=72))
    imu=np.concatenate(chunks);imu["t"]+=round(args.imu_timestamp_shift_ms*1e6);imu=imu[np.linalg.norm(imu["acc"],axis=1)>1]
    assert np.all(np.diff(imu["t"])>0)
    assert targets[frames[0]]>=imu["t"][0] and targets[frames[-1]]<=imu["t"][-1]
    bias=np.asarray(yaml.safe_load((calibration/"imu_bias.yaml").read_text())["gyroscope_bias"])
    imu_yaml=yaml.safe_load((calibration/"imu.yaml").read_text())
    epoch=int(imu["t"][0]);imu_index=0

    maps = [pinhole_maps(cameras_yaml[f"fisheye{i}"]) for i in range(4)]
    cameras = []
    camera_only_calibration = {}
    # Re-express the static camera calibration in physical camera 2 coordinates.
    # T_cam_imu is only a stored extrinsic convention; the common intermediate
    # frame cancels. No IMU orientation/gravity/sample is used.
    reference_from_calibration = np.array(cameras_yaml["fisheye2"]["T_cam_imu"])
    for cam in CAMERA_ORDER:
        transform = reference_from_calibration @ np.linalg.inv(np.array(cameras_yaml[f"fisheye{cam}"]["T_cam_imu"]))
        if cam in (0, 1):
            roll = np.eye(4)
            roll[:3, :3] = Rotation.from_euler("z", 90 if cam == 0 else -90, degrees=True).as_matrix()
            transform = transform @ roll
        camera_only_calibration[str(cam)] = dict(reference_from_camera=transform.tolist(),size=[SIZE,SIZE],focal=[FISHEYE_FOCAL,FISHEYE_FOCAL],principal=[CENTER,CENTER])
        pose = v.Pose(rotation=Rotation.from_matrix(transform[:3, :3]).as_quat(), translation=transform[:3, 3])
        cameras.append(v.Camera(size=[SIZE, SIZE], focal=[FISHEYE_FOCAL]*2, principal=[CENTER]*2, distortion=v.Distortion(v.Distortion.Model.Fisheye,[0.]*4), rig_from_camera=pose))
    config = v.Tracker.OdometryConfig(odometry_mode=v.Tracker.OdometryMode.Multisensor,
                                      rectified_stereo_camera=False, async_sba=False,
                                      use_motion_model=True, enable_observations_export=True)
    imu_pose=v.Pose(rotation=Rotation.from_matrix(reference_from_calibration[:3,:3]).as_quat(),translation=reference_from_calibration[:3,3])
    keys=("gyroscope_noise_density","gyroscope_random_walk","accelerometer_noise_density","accelerometer_random_walk")
    imu_cal=v.ImuCalibration(rig_from_imu=imu_pose,frequency=imu_yaml["update_rate"],**{k:imu_yaml[k] for k in keys})
    slam = None  # Release output is VIO-only, with no SLAM backend.
    tracker = v.Tracker(v.Rig(cameras,[imu_cal]),config,slam)
    events, online_slam = {}, []
    graph_max = [0, 0]
    diagnostic_ms = 0.0
    expert={"sba.num_inertial_sba_frames":"10","sba.num_inertial_fixed_sba_frames":"5","sba.num_inertial_sba_iterations":"10","sba.inertial_robustifier_scale":"1.5"}
    tracker.odom.apply_expert_parameters(expert)
    print("MATCHED_INPUT_VIO imu_count=1 gyro_image_warp=0 temporal_insert=0 cameras=4 reference=camera2",flush=True)
    out.mkdir(parents=True, exist_ok=False)
    capture_pack=[0]*4
    (out/"effective_calibration.json").write_text(json.dumps(dict(reference="camera_2",cameras=camera_only_calibration),indent=2))
    captures=[cv2.VideoCapture(str(packs[0]/f"cam{i}_{packs[0].name}.mp4")) for i in range(4)]
    residual_ms=np.array([(timestamps[i]-targets)*1e-6 for i in range(4)])
    (out/"input_audit.json").write_text(json.dumps(dict(packs=pack_audit,total_groups=len(targets),used_frames=len(frames),
        camera_offset_s=args.camera_time_offsets,recorded_rotation_applied=args.apply_recorded_rotation,recorded_rotations=rotations_metadata, camera_residual_abs_ms_p50_p95_max=np.percentile(abs(residual_ms),[50,95,100],axis=1).tolist(),
        imu_timestamp_shift_ms=args.imu_timestamp_shift_ms,imu_samples=len(imu),imu_registered=True,gyro_image_warp=False,frame_selection="camera_common_interval_only",visual_up_axis=[0,-1,0],rig_reference="physical_camera_2"),indent=2))
    last_index = [-1]*4
    last_image = [None]*4

    def read_frame(cam, index):
        if index < last_index[cam]:
            raise ValueError(f"Nonmonotonic frame index for camera {cam}")
        while last_index[cam] < index:
            ok, image = captures[cam].read()
            if not ok and capture_pack[cam]+1<len(packs):
                captures[cam].release()
                capture_pack[cam]+=1
                pack=packs[capture_pack[cam]]
                captures[cam]=cv2.VideoCapture(str(pack/f"cam{cam}_{pack.name}.mp4"))
                ok,image=captures[cam].read()
            if not ok:
                raise ValueError(f"Missing camera {cam} frame {last_index[cam]+1}")
            last_index[cam] += 1
            last_image[cam] = cv2.remap(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), *maps[cam][:2], cv2.INTER_LINEAR)
        return last_image[cam]

    k = np.array([[FOCAL, 0, CENTER], [0, FOCAL, CENTER], [0, 0, 1.]])
    k_inv = np.linalg.inv(k)
    y, x = np.ogrid[:SIZE, :SIZE]
    central = (((x-CENTER)**2 + (y-CENTER)**2) > 360**2).astype("uint8")*255
    remap_mask_original = remap_mask
    mask_cache={}
    def cached_mask(cam, mask):
        if cam not in mask_cache: mask_cache[cam]=remap_mask_original(mask)
        return mask_cache[cam]
    central_fisheye=remap_mask(central)
    digest = hashlib.sha256()
    trajectory, diagnostics = [], []
    frame_hashes=[]
    started=time.monotonic()
    stream=(out/"online.csv").open("w",buffering=1)
    writer=csv.writer(stream)
    writer.writerow(["frame","timestamp_unix_ns","tx","ty","tz","qx","qy","qz","qw","gx","gy","gz"])

    for frame in frames:
        stamp = int(targets[frame])
        images, masks = [], []
        for cam in CAMERA_ORDER:
            nearest = int(np.argmin(abs(timestamps[cam] - stamp)))
            image, mask = read_frame(cam, nearest), maps[cam][2]
            if cam in (0, 1):
                roll = cv2.ROTATE_90_COUNTERCLOCKWISE if cam == 0 else cv2.ROTATE_90_CLOCKWISE
                image = cv2.rotate(image, roll)
                mask = cv2.rotate(cv2.bitwise_or(mask, central_fisheye), roll)
            if args.foreground_mask != "none" and (cam in (2,3) or args.foreground_mask == "all"):
                # Fixed image-space diagnostic ROI, identical for both recordings.
                # 255 means excluded. Apply AFTER the existing camera roll.
                polygons = {
                    2: [(0.32,0.48),(0.65,0.48),(0.85,1),(0.10,1)],
                    3: [(0.32,0.48),(0.65,0.48),(0.85,1),(0.10,1)],
                    0: [(0.33,0.23),(0.59,0.28),(0.72,0.61),(0.56,1),(0,1),(0,0.48)],
                    1: [(0.67,0.23),(0.41,0.28),(0.28,0.61),(0.44,1),(1,1),(1,0.48)],
                }
                mask = mask.copy()
                polygon = np.rint(np.array(polygons[cam])*(SIZE-1)).astype(np.int32)
                poly_mask=np.zeros_like(mask)
                cv2.fillPoly(poly_mask,[polygon],255)
                mask=cv2.bitwise_or(mask,cached_mask(cam,poly_mask))
            if frame == frames[0]:
                cv2.imwrite(str(out/f"mask_cam{cam}.png"),mask)
            images.append(image)
            masks.append(mask)
        while imu_index<len(imu) and int(imu["t"][imu_index])<=stamp:
            row=imu[imu_index]
            tracker.register_imu_measurement(0,v.ImuMeasurement(timestamp_ns=int(row["t"])-epoch,linear_accelerations=row["acc"],angular_velocities=row["gyro"]-bias))
            imu_index+=1
        digest.update(np.int64(stamp).tobytes())
        for array in images + masks:
            digest.update(array.tobytes())
        frame_hashes.append(digest.hexdigest())
        start_track = time.perf_counter()
        estimate, corrected = tracker.track(stamp-epoch, images, masks=masks)
        publish_ms = (time.perf_counter() - start_track) * 1000.
        probe=time.perf_counter()
        online_slam.append([np.nan]*7 if corrected is None else [*corrected.translation,*corrected.rotation])
        if slam is not None:
            metric=tracker.get_slam_metrics()
            if metric is not None and int(metric.timestamp_ns) not in events:
                events[int(metric.timestamp_ns)]=[int(metric.lc_status),int(metric.pgo_status),
                    metric.lc_selected_landmarks_count,metric.lc_tracked_landmarks_count,
                    metric.lc_pnp_landmarks_count,metric.lc_good_landmarks_count]
                if metric.pgo_status:
                    print("SLAM_EVENT", stamp, events[int(metric.timestamp_ns)], flush=True)
                    graph=tracker.get_pose_graph()
                    graph_max=[max(graph_max[0],len(graph.nodes)),max(graph_max[1],len(graph.edges))]
        diagnostic_ms+=(time.perf_counter()-probe)*1000
        pose = estimate.world_from_rig
        values = [*pose.pose.translation, *pose.pose.rotation] if pose is not None else [np.nan]*7
        trajectory.append([*values, *(tracker.get_last_gravity() or [np.nan]*3)])
        writer.writerow([int(frame),stamp,*trajectory[-1]])
        if len(trajectory)%100==0 or frame==frames[-1]:
            print(json.dumps(dict(frame=int(frame),count=len(trajectory),total=len(frames),elapsed=time.monotonic()-started)),flush=True)
        state = tracker.odom.get_state()
        diagnostics.append([frame, int(state.keyframe), int(state.warming_up), publish_ms,
                            *(len(tracker.get_last_observations(i)) for i in range(4))])
    probe=time.perf_counter()
    history=tracker.get_all_slam_poses() if slam is not None else []
    export_ms=(time.perf_counter()-probe)*1000
    prefix=out/"slam_diagnostics"
    np.savez_compressed(str(prefix)+".slam.npz", online=np.asarray(online_slam),
        history_t=np.array([p.timestamp_ns for p in history],dtype=np.int64)+epoch,
        history=np.array([[*p.pose.translation,*p.pose.rotation] for p in history]).reshape(-1,7))
    with open(str(prefix)+".events.csv","w",newline="") as stream_events:
        rows=csv.writer(stream_events)
        rows.writerow(["timestamp_ns","lc_status","pgo_status","selected","tracked","pnp","good"])
        rows.writerows([t+epoch,*flags] for t,flags in events.items())
    Path(str(prefix)+".backend.json").write_text(json.dumps(dict(version=v.get_version()[0], enabled=slam is not None,
        synchronous=slam is not None, history_frames=len(history), lc=sum(r[0] for r in events.values()),
        pgo=sum(r[1] for r in events.values()), graph_nodes_edges_max=graph_max,
        diagnostics_ms=diagnostic_ms, history_fetch_ms=export_ms)))
    stream.close()
    (out/"input_frame_hashes.json").write_text(json.dumps(frame_hashes))
    for cap in captures:
        cap.release()
    values = np.asarray(trajectory, dtype=float)
    np.savez_compressed(out/"trajectory.npz", frame_index=frames, timestamp_unix_ns=targets[frames],
                        translation_m=values[:, :3], quaternion_xyzw=values[:, 3:7], gravity_rig_m_s2=values[:, 7:])
    with (out/"diagnostics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame", "keyframe", "warming_up", "track_to_pose_ms", *(f"observations_cam{i}" for i in CAMERA_ORDER)])
        writer.writerows(diagnostics)
    (out/"configs.json").write_text(json.dumps(dict(camera_order=CAMERA_ORDER, camera_time_offsets_s=args.camera_time_offsets,
        foreground_mask=args.foreground_mask,time_origin_unix_ns=epoch,rig_reference="physical_camera_2",visual_up_axis=[0,-1,0], imu_count=1,imu_registered=imu_index,gyro_image_warp=False,hash_format="frame timestamp, images, masks ONLY; IMU is registered but excluded from image hash", input_sha256=digest.hexdigest(), expert_parameters=expert), indent=2))


if __name__ == "__main__":
    main()
