# HALO 四目 GPU VIO · runtime_reduced_v1.36

发布入口是 `run_vio.sh`。它先在 CPU 上核验 cam2 与**当前提供的标定**之间的几何关系，写出 `orientation_audit.json`，通过后才加载 cuVSLAM 并计算完整 VIO。默认四目、原始分辨率、无人体掩膜、无额外空间裁剪、GPU Schur FP64、四状态窗口，关闭回环和 PGO。

## 使用

运行包要求 Linux x86_64、Python 3.10 和兼容 CUDA 12 的 NVIDIA 驱动；已在 A100 上验证。包内 Python 依赖和 CUDA 数学库由启动器相对路径加载。源码用法是将相同 Python 环境中的 `run.py` 作为入口。

```bash
./run_vio.sh --data-root /path/to/recordvideo_20261104_195610 --output-root /path/to/new_result
./run_vio.sh --data-root /path/to/HALO_851 --sequence recordvideo_20261028_170302 --output-root /path/to/new_result
./run_vio.sh --data-root /path/to/data_test2 --time-profile data_test2_validated --output-root /path/to/new_result
```

只检查方向：在相同命令追加 `--check-only`。输出目录必须为空，以保护已有结果。默认运行全部共同图像/IMU覆盖范围；`--max-frames` 仅供明确的前缀诊断。

支持 `calibration/` 加 `pack_*` 的目录、单个 `fisheye*.mp4` 录像目录，以及用 `--sequence` 指定的多包父目录。合并包必须使用相同标定与相同旋转补偿。

## 方向如何确定

对每包最多九个不同同步组，比较 cam2 的 0°/180°假设与 cam0/1/3 的标定极线一致性，使用 double-sphere 反投影。至少两个不同时间点需有独立支持；弱纹理、两假设无法区分、包间结论不同、标定无效或解码失败时停止，保留诊断文件，**不会开始 VIO**。

录像的 `software_rotate_deg` 描述固件曾经做过的操作；当前 YAML 可能已经包含这个变换。因此决定的是“当前标定还需补偿多少”，而非盲从元数据。

若需 180°补偿，入口同时变换 cam2 的主点和 `T_cam_imu`，使标定进入录像像素坐标。图像保持原始分辨率，不重编码。该变换有投影几何等价性单元测试。内部 `replay.py` 要求匹配的预检报告，并在加载 SDK 前重新核验标定哈希。

这是文件回放入口的前置检查。直接调用 C++/Python SDK 时，调用者仍负责提供与图像相匹配的相机标定；可以复用 `camera_orientation.py` 的检查函数。

## 时间和旧格式

默认使用每目 YAML 的 `timeshift_cam_imu`，额外 IMU 时移默认为 0。支持显式 `--camera-time-offsets` 或命名 `--time-profile`，不会按目录名称暗中选择实验参数。

`time_profiles.json` 保存两项明确适用范围的已验证例外：`data_test2_validated` 与 `halo856_shared_cam23`。HALO856 原 cam3 时移接近一帧，但原始视频同帧配对明显优于错一帧，使用该 profile 时 cam3 共享 cam2 的标定时移。原标定文件不改动。

旧固件仅有 `unix_ns` 名义网格时，在几何同帧检查通过后按 cam2 名义网格建立适配，并在 `input/*/legacy_time_audit.json` 和 CSV 注释中明确标记；不能从中恢复未保存的真实曝光时刻。源 CSV 的时间值保留在 `source_unix_ns`。

## 输出

- `orientation_audit.json`：逐包元数据、校验哈希、两假设支持量与结论。
- `execution.json`：实际旋转补偿与时序策略。
- `run.log`：完整 VIO 日志。
- `vio/trajectory.npz`、`vio/online.csv`：完整有效覆盖范围，参考系为物理 cam2。
- `vio/configs.json`、`vio/native_input_audit.json`：实际参数、输入尺寸、是否重采样。
- `vio/slam_diagnostics.backend.json`：可核验回环和 PGO 均关闭。

## 验证

```bash
python3 -m unittest discover -s examples/halo_vio -p test_orientation.py -v
```

`doc/v1.36/results/REPORT.html` 包含 19 组最优稳定版本结果（room1–room6 有真值）；`doc/v1.36/audit/REPORT.html` 包含时间、窗口和发布一致性核查。发布基于稳定核心 0c68c5c 的源码快照，重新提交和构建后核心哈希会改变，逐帧验证记录保存在运行包 manifest 中。

已知问题：201148 仍有 3.648 m/s 原始发布位姿速度峰值；room 高度误差未完全解决。Cauchy 导数实验、PnP 多假设实验、实际鱼眼地图验收统一与全 FP64 后验不作为本版本已验证修复。当前默认配置关闭这些实验。四帧是窗口上限，拒绝候选后保留前台结果，不保证每段都能达到满窗。
