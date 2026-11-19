# runtime_reduced_v1.35

本发布基于 `07c3132` 已验证的投影、新地标重捕获及恢复后发布修复，增加双目验收覆盖、自动 cam2 标定方向预检和可复现 cuNLS 依赖补丁。HALO 四目入口默认开启四路前景掩膜和 GPU Schur FP64，关闭 SLAM 回环与 PGO。

## 修改及原因

| 文件 | 修改及原因 |
|---|---|
| `libs/pipelines/imu_fusion_context.cpp` | 将前台发布验收覆盖扩展到双目，避免双目跳过速度、偏置及观测检查。 |
| `cmake/ext/cunls_runtime_compat.patch` | 保存已验证依赖中的 `<cstdlib>`、CUDA 头文件兼容声明，以及 PCG 两次共享存储归约之间的同步屏障，消除源码重建漏带修复的风险。 |
| `cmake/ext/cunls.cmake` | 对下载与本地覆盖的 cuNLS 都按顺序、幂等应用两份补丁，失败时停止配置。 |
| `examples/halo_vio/camera_orientation.py` | 按提供的 YAML 核验 0°/180°几何假设、独立帧支持和旋转矩阵有效性，避免重复应用固件旋转。 |
| `examples/halo_vio/run.py` | 把预检放在 SDK 加载之前；支持原始与规范化录像目录、明确的旧名义时钟适配和审计输出。 |
| `examples/halo_vio/replay.py` | 使用显式且经审计的 cam2 旋转补偿；加载 SDK 前检查校验结果和标定哈希；去除场景专用 PGO 探针，保留原分辨率回放。 |
| `examples/halo_vio/python.sh` | 使用包内相对依赖路径，提供 Python 解释器选择，移除个人实验目录依赖。 |
| `examples/halo_vio/run_vio.sh` | 将公开运行入口接到预检编排器。 |
| `examples/halo_vio/time_profiles.json` | 把已有特殊时序配置作为显式命名 profile 保存，保留原因，默认仍用 YAML。 |
| `examples/halo_vio/test_orientation.py` | 覆盖已旋转标定、缺少补偿、证据不足、投影等价和无效反投影域。 |
| `examples/halo_vio/README.md` | 说明入口、参数、坐标、时序限制与输出。 |
| `doc/v1.35/summary.csv`、`verification.json` | 保存本轮 14 组聚合验证及实际实验运行库哈希；该哈希与发布重建文件哈希分别记录。 |

## 已验证结果

14 组共 22422 帧均有有效位姿，无失跟；13 组重力初始化成功且其后无中断。152938 因短片未完成重力初始化，其竖直坐标只作视觉轴展示。所有实跑关闭 PGO，使用原始 1920×1200 图像；速度按实际帧间时间计算。14 组数据没有真值，不报告 ATE。

已补强的双目验收：room2 ATE 从 10.84cm 降至 8.53cm，起终高度误差从 -19.34cm 改善到 +2.07cm；room6 轨迹与原修复构建一致。room5 的剩余高度误差仍为 -19.50cm，未将跨数据退化的共识重优化候选纳入发布。

## 运行包

共享盘运行包：`/nas/shared/baidx/cuvslam_runs/runtime_reduced_v1.35/`。

从该目录执行 `./run_vio.sh --data-root DATA --output-root NEW_OUTPUT`，先自动核验方向，再完整运行。详见 `examples/halo_vio/README.md`。直接使用 SDK 的应用仍需自行保证图像与标定一致。

本 GitHub Release 提供源码与聚合验证附件；大体积 GPU 服务器运行包位于上述共享盘。SDK 核心 API 版本与本运行包发布编号分开记录，以提交哈希追踪构建来源。
