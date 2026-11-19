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

"""Replay the four-fisheye pack_001 format with the optimized rig configuration.

Run with the patched cuVSLAM Python module, numpy, scipy, pyyaml and OpenCV.
Usage: python run.py --data-root SEQUENCE --output-root NEW_DIRECTORY
Time offsets come from the supplied YAML or explicit CLI values in physical camera order.
"""

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

import cv2
from camera_orientation import rotate_calibration
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

CAMERA_ORDER = (2, 3, 0, 1)
SIZE = 768
CENTER = (SIZE - 1) / 2
FOCAL = SIZE / (2 * np.tan(np.deg2rad(130) / 2))


FISHEYE_FOCAL=(SIZE/2)/np.deg2rad(65)

def native_model(camera, cam, foreground):
    xi,alpha,fx,fy,cx,cy=map(float,camera["intrinsics"])
    width,height=map(int,camera["resolution"])
    theta=np.linspace(0,np.deg2rad(89),4097)
    z=np.cos(theta)+xi
    radial=np.sin(theta)/(alpha*np.hypot(np.sin(theta),z)+(1-alpha)*z)
    design=theta[:,None]**np.array([1,3,5,7,9])
    coeff=np.linalg.lstsq(design,radial,rcond=None)[0]
    fit_err=float(np.max(abs(design@coeff-radial))*max(fx,fy))
    if fit_err>.05 or not np.all(np.diff(radial)>0):
        raise ValueError(f"Invalid native fisheye calibration for camera {cam}: {fit_err}")
    yy,xx=np.ogrid[:height,:width]
    nx=(xx-cx)/fx
    ny=(yy-cy)/fy
    r=np.hypot(nx,ny)
    angle=np.interp(r,radial,theta)
    unit=FISHEYE_FOCAL*angle/np.maximum(r,1e-12)
    eqx,eqy=nx*unit,ny*unit
    valid=(r<=radial[-1]) & (abs(eqx)<=CENTER) & (abs(eqy)<=CENTER)
    if cam in (0,1):
        valid &= np.hypot(eqx,eqy)<=360
    ratio=FOCAL*np.tan(np.minimum(angle,1.56))/np.maximum(r,1e-12)
    px,py=nx*ratio,ny*ratio
    if cam==0:
        px,py=py,-px
    elif cam==1:
        px,py=-py,px
    mask=(~valid).astype(np.uint8)*255
    if foreground=="all" or (foreground=="stereo" and cam in (2,3)):
        polygons={
            2: [(0.32,0.48),(0.65,0.48),(0.85,1),(0.10,1)],
            3: [(0.32,0.48),(0.65,0.48),(0.85,1),(0.10,1)],
            0: [(0.33,0.23),(0.59,0.28),(0.72,0.61),(0.56,1),(0,1),(0,0.48)],
            1: [(0.67,0.23),(0.41,0.28),(0.28,0.61),(0.44,1),(1,1),(1,0.48)],
        }
        polygon=np.rint(np.asarray(polygons[cam])*(SIZE-1)).astype(np.int32)
        old=np.zeros((SIZE,SIZE),np.uint8)
        cv2.fillPoly(old,[polygon],255)
        invalid=cv2.remap(old,(CENTER+px).astype(np.float32),(CENTER+py).astype(np.float32),
                          cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=255)
        mask=cv2.bitwise_or(mask,invalid)
    return dict(size=[width,height],focal=[fx*coeff[0],fy*coeff[0]],principal=[cx,cy],
                coefficients=(coeff[1:]/coeff[0]).tolist(),model_fit_max_pixel=fit_err),mask


def main():
    cv2.setNumThreads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--imu-timestamp-shift-ms",type=float,default=0.)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--camera-time-offsets", type=float, nargs=4, default=None)
    parser.add_argument("--orientation-audit",type=Path,required=True)
    parser.add_argument("--cam2-rotation-deg",type=int,choices=(0,180),required=True)
    parser.add_argument("--foreground-mask", choices=["none","stereo","all"], default="none")
    args = parser.parse_args()
    orientation=json.loads(args.orientation_audit.read_text())
    if not orientation.get('accepted') or orientation.get('cam2_calibration_rotation_deg')!=args.cam2_rotation_deg:
        raise ValueError('A matching accepted orientation audit is required before VIO')
    calhash=hashlib.sha256((args.data_root/'calibration/camera_imu.yaml').read_bytes()).hexdigest()
    if any(p['calibration_sha256']!=calhash for p in orientation['packs']):
        raise ValueError('Calibration changed after the orientation check')
    import cuvslam as v
    v.set_verbosity(3 if args.audit else 0)
    data, out = args.data_root, args.output_root
    for key in ("CUVSLAM_JOINT_VI", "CUVSLAM_JOINT_BACKEND", "CUVSLAM_WINDOW_SOLVER", "CUVSLAM_METRIC_INIT", "CUVSLAM_CAM01_TEMPORAL"):
        os.environ.pop(key, None)
    if os.environ['BUILD_KIND'] != 'original': os.environ.update(CUVSLAM_JOINT_VI="1",CUVSLAM_JOINT_BACKEND="window",CUVSLAM_WINDOW_SOLVER=os.environ["VIO_COMPARE_SOLVER"],CUVSLAM_WINDOW_SIZE="4",CUVSLAM_METRIC_INIT="1")
    cv2.setNumThreads(4)
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
    cameras_yaml=rotate_calibration(cameras_yaml,args.cam2_rotation_deg)
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
    frames=frames[(targets[frames]>=imu["t"][0])&(targets[frames]<=imu["t"][-1])]
    if len(frames)==0:
        raise ValueError("No common image/IMU time coverage")
    bias=np.asarray(yaml.safe_load((calibration/"imu_bias.yaml").read_text())["gyroscope_bias"])
    imu_yaml=yaml.safe_load((calibration/"imu.yaml").read_text())
    epoch=int(imu["t"][0]);imu_index=0

    native=[native_model(cameras_yaml[f"fisheye{i}"],i,args.foreground_mask) for i in range(4)]
    source_shapes=[tuple(cameras_yaml[f"fisheye{i}"]["resolution"][::-1]) for i in range(4)]
    cameras = []
    camera_only_calibration = {}
    # Re-express the static camera calibration in physical camera 2 coordinates.
    # T_cam_imu is only a stored extrinsic convention; the common intermediate
    # frame cancels. No IMU orientation/gravity/sample is used.
    reference_from_calibration = np.array(cameras_yaml["fisheye2"]["T_cam_imu"])
    for cam in CAMERA_ORDER:
        transform = reference_from_calibration @ np.linalg.inv(np.array(cameras_yaml[f"fisheye{cam}"]["T_cam_imu"]))
        model,mask=native[cam]
        camera_only_calibration[str(cam)]=dict(reference_from_camera=transform.tolist(),**model)
        pose=v.Pose(rotation=Rotation.from_matrix(transform[:3,:3]).as_quat(),translation=transform[:3,3])
        cameras.append(v.Camera(size=model["size"],focal=model["focal"],principal=model["principal"],
                                distortion=v.Distortion(v.Distortion.Model.Fisheye,model["coefficients"]),
                                rig_from_camera=pose))
    config = v.Tracker.OdometryConfig(odometry_mode=v.Tracker.OdometryMode.Multisensor,
                                      rectified_stereo_camera=False, async_sba=False,
                                      use_motion_model=True, enable_observations_export=True)
    imu_pose=v.Pose(rotation=Rotation.from_matrix(reference_from_calibration[:3,:3]).as_quat(),translation=reference_from_calibration[:3,3])
    keys=("gyroscope_noise_density","gyroscope_random_walk","accelerometer_noise_density","accelerometer_random_walk")
    imu_cal=v.ImuCalibration(rig_from_imu=imu_pose,frequency=imu_yaml["update_rate"],**{k:imu_yaml[k] for k in keys})
    slam = None
    if os.environ["SLAM_ENABLED"] == "1":
        slam = v.Tracker.SlamConfig()
        slam.sync_mode = True
        slam.enable_reading_internals = True
    tracker = v.Tracker(v.Rig(cameras,[imu_cal]),config,slam)
    events, online_slam = {}, []
    graph_max = [0, 0]
    diagnostic_ms = 0.0
    expert={"sba.num_inertial_sba_frames":"10","sba.num_inertial_fixed_sba_frames":"5","sba.num_inertial_sba_iterations":"10","sba.inertial_robustifier_scale":"1.5"}
    if os.environ['BUILD_KIND'] != 'original': tracker.odom.apply_expert_parameters(expert)
    print("MATCHED_INPUT_VIO imu_count=1 gyro_image_warp=0 temporal_insert=0 cameras=4 reference=camera2",flush=True)
    out.mkdir(parents=True, exist_ok=True)
    capture_pack=[0]*4
    (out/"effective_calibration.json").write_text(json.dumps(dict(reference="camera_2",cameras=camera_only_calibration),indent=2))
    captures=[cv2.VideoCapture(str(packs[0]/f"cam{i}_{packs[0].name}.mp4")) for i in range(4)]
    residual_ms=np.array([(timestamps[i]-targets)*1e-6 for i in range(4)])
    (out/"input_audit.json").write_text(json.dumps(dict(packs=pack_audit,total_groups=len(targets),used_frames=len(frames),
        camera_offset_s=args.camera_time_offsets,recorded_rotation_applied=args.cam2_rotation_deg==180,cam2_calibration_rotation_deg=args.cam2_rotation_deg,recorded_rotations=rotations_metadata, camera_residual_abs_ms_p50_p95_max=np.percentile(abs(residual_ms),[50,95,100],axis=1).tolist(),
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
            if image.shape[:2]!=source_shapes[cam]:
                raise ValueError(f"Physical camera {cam} has unexpected shape {image.shape[:2]}")
            last_image[cam]=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        return last_image[cam]

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
            image,mask=read_frame(cam,nearest),native[cam][1]
            if image.shape!=mask.shape:
                raise ValueError(f"Camera {cam} has inconsistent native data and mask")
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
        if frame==frames[0]:
            (out/"native_input_audit.json").write_text(json.dumps(dict(
              decoded_shapes=[list(v) for v in source_shapes],track_order=CAMERA_ORDER,
              track_shapes=[list(v.shape) for v in images],resampling=False,media_reencoding=False,
              fisheye_approximation_max_px=[m[0]["model_fit_max_pixel"] for m in native]),indent=2))
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
    np.savez(str(prefix)+".slam.npz", online=np.asarray(online_slam),
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
    np.savez(out/"trajectory.npz", frame_index=frames, timestamp_unix_ns=targets[frames],
                        translation_m=values[:, :3], quaternion_xyzw=values[:, 3:7], gravity_rig_m_s2=values[:, 7:])
    with (out/"diagnostics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame", "keyframe", "warming_up", "track_to_pose_ms", *(f"observations_cam{i}" for i in CAMERA_ORDER)])
        writer.writerows(diagnostics)
    (out/"configs.json").write_text(json.dumps(dict(camera_order=CAMERA_ORDER, camera_time_offsets_s=args.camera_time_offsets,
        foreground_mask=args.foreground_mask,time_origin_unix_ns=epoch,rig_reference="physical_camera_2",visual_up_axis=[0,-1,0], imu_count=1,imu_registered=imu_index,gyro_image_warp=False,hash_format="frame timestamp, images, masks ONLY; IMU is registered but excluded from image hash", input_sha256=digest.hexdigest(), expert_parameters=expert), indent=2))


if __name__ == "__main__":
    main()
