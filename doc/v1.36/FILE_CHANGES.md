# v1.36 逐文件修改与原因

相对 v1.35-r1（9572835）。算法来自已验证稳定快照；实验开关仍默认关闭，发布入口清除继承的实验配置。

| 文件 | 修改内容 | 原因 |
|---|---|---|
| `doc/v1.36/FILE_CHANGES.md` | 保存发布说明和逐文件变更 | 使源码、验证输入与已知限制可追溯 |
| `doc/v1.36/README.md` | 保存发布说明和逐文件变更 | 使源码、验证输入与已知限制可追溯 |
| `doc/v1.36/audit/AUDIT.json` | 保存时间、窗口和发布一致性核查 | 使源码、验证输入与已知限制可追溯 |
| `doc/v1.36/audit/CODE_FINDINGS.json` | 保存时间、窗口和发布一致性核查 | 使源码、验证输入与已知限制可追溯 |
| `doc/v1.36/audit/PIXEL_POLICY_AUDIT.json` | 保存时间、窗口和发布一致性核查 | 使源码、验证输入与已知限制可追溯 |
| `doc/v1.36/audit/REPORT.html` | 保存时间、窗口和发布一致性核查 | 使源码、验证输入与已知限制可追溯 |
| `doc/v1.36/audit/STAGE_AUDIT.json` | 保存时间、窗口和发布一致性核查 | 使源码、验证输入与已知限制可追溯 |
| `doc/v1.36/audit/WINDOW_SIZE_AUDIT.json` | 保存时间、窗口和发布一致性核查 | 使源码、验证输入与已知限制可追溯 |
| `doc/v1.36/audit/vio_time_state_commit_tests.log` | 保存时间、窗口和发布一致性核查 | 使源码、验证输入与已知限制可追溯 |
| `doc/v1.36/audit/vio_time_state_gap_tests.log` | 保存时间、窗口和发布一致性核查 | 使源码、验证输入与已知限制可追溯 |
| `doc/v1.36/audit/vio_time_state_window_tests.log` | 保存时间、窗口和发布一致性核查 | 使源码、验证输入与已知限制可追溯 |
| `doc/v1.36/baseline_ctest.log` | 保存发布说明和逐文件变更 | 使源码、验证输入与已知限制可追溯 |
| `doc/v1.36/results/MANIFEST.json` | 保存已验证最优版本结果 | 使源码、验证输入与已知限制可追溯 |
| `doc/v1.36/results/REPORT.html` | 保存已验证最优版本结果 | 使源码、验证输入与已知限制可追溯 |
| `doc/v1.36/results/VERSION.json` | 保存已验证最优版本结果 | 使源码、验证输入与已知限制可追溯 |
| `doc/v1.36/results/best_metrics.csv` | 保存已验证最优版本结果 | 使源码、验证输入与已知限制可追溯 |
| `doc/v1.36/results/best_metrics.json` | 保存已验证最优版本结果 | 使源码、验证输入与已知限制可追溯 |
| `doc/v1.36/results/halo851_152938.png` | 保存已验证最优版本结果 | 使源码、验证输入与已知限制可追溯 |
| `doc/v1.36/results/halo851_154711.png` | 保存已验证最优版本结果 | 使源码、验证输入与已知限制可追溯 |
| `doc/v1.36/results/halo851_155031.png` | 保存已验证最优版本结果 | 使源码、验证输入与已知限制可追溯 |
| `doc/v1.36/results/halo851_155625.png` | 保存已验证最优版本结果 | 使源码、验证输入与已知限制可追溯 |
| `doc/v1.36/results/halo851_155811.png` | 保存已验证最优版本结果 | 使源码、验证输入与已知限制可追溯 |
| `doc/v1.36/results/halo851_170302.png` | 保存已验证最优版本结果 | 使源码、验证输入与已知限制可追溯 |
| `doc/v1.36/results/literature_results.json` | 保存已验证最优版本结果 | 使源码、验证输入与已知限制可追溯 |
| `doc/v1.36/results/opensource_comparison.csv` | 保存已验证最优版本结果 | 使源码、验证输入与已知限制可追溯 |
| `doc/v1.36/results/opensource_comparison.json` | 保存已验证最优版本结果 | 使源码、验证输入与已知限制可追溯 |
| `doc/v1.36/results/room1.png` | 保存已验证最优版本结果 | 使源码、验证输入与已知限制可追溯 |
| `doc/v1.36/results/room2.png` | 保存已验证最优版本结果 | 使源码、验证输入与已知限制可追溯 |
| `doc/v1.36/results/room3.png` | 保存已验证最优版本结果 | 使源码、验证输入与已知限制可追溯 |
| `doc/v1.36/results/room4.png` | 保存已验证最优版本结果 | 使源码、验证输入与已知限制可追溯 |
| `doc/v1.36/results/room5.png` | 保存已验证最优版本结果 | 使源码、验证输入与已知限制可追溯 |
| `doc/v1.36/results/room6.png` | 保存已验证最优版本结果 | 使源码、验证输入与已知限制可追溯 |
| `doc/v1.36/results/run_manifest.json` | 保存已验证最优版本结果 | 使源码、验证输入与已知限制可追溯 |
| `doc/v1.36/results/test_162436.png` | 保存已验证最优版本结果 | 使源码、验证输入与已知限制可追溯 |
| `doc/v1.36/results/test_180123.png` | 保存已验证最优版本结果 | 使源码、验证输入与已知限制可追溯 |
| `doc/v1.36/results/trajectories.html` | 保存已验证最优版本结果 | 使源码、验证输入与已知限制可追溯 |
| `doc/v1.36/results/validation_195610.png` | 保存已验证最优版本结果 | 使源码、验证输入与已知限制可追溯 |
| `doc/v1.36/results/validation_200730.png` | 保存已验证最优版本结果 | 使源码、验证输入与已知限制可追溯 |
| `doc/v1.36/results/validation_201148.png` | 保存已验证最优版本结果 | 使源码、验证输入与已知限制可追溯 |
| `doc/v1.36/results/validation_202341.png` | 保存已验证最优版本结果 | 使源码、验证输入与已知限制可追溯 |
| `doc/v1.36/results/validation_203017.png` | 保存已验证最优版本结果 | 使源码、验证输入与已知限制可追溯 |
| `examples/halo_vio/README.md` | 更新无掩模使用方法、结果位置和已知问题 | 不把实验修复或未解决的高度/速度问题写成已解决 |
| `examples/halo_vio/replay.py` | 使用已验证原始鱼眼无掩模、有限 IMU 数据回放 | 保持原始视野、标定 YAML 时移和因果 IMU 输入 |
| `examples/halo_vio/run.py` | 设置 v1.36 稳定参数，清除继承实验环境，默认无 mask，加入审计选项 | 使直接运行与 19 组验证配置一致，并保留旋转预检 |
| `examples/halo_vio/test_release.py` | 测试发布默认环境不会继承 PGO 或实验选项 | 保证他人通过启动器运行时配置确定 |
| `libs/cuda_modules/cuda_kernels/CMakeLists.txt` | 注册标定图块复核 CUDA 核 | 使可选几何复核使用同一构建依赖 |
| `libs/cuda_modules/cuda_kernels/calibrated_patch_check.cu` | 加入有边界与低纹理保护的标定图块 NCC 对照核 | 供匹配几何诊断；未作为发布默认实验开关 |
| `libs/cuda_modules/cuda_kernels/cuda_kernels.h` | 声明标定图块复核接口 | 保持核调用的数据及参数一致 |
| `libs/cuda_modules/test/st_tracker_test.cpp` | 增加标定图块复核测试 | 检查越界和低纹理输入不会成为有效匹配 |
| `libs/epipolar/test/camera_selection_test.cpp` | 补充重复可见轨迹与相机选择回归 | 防止三角化缓存错误导致重复创建点 |
| `libs/imu/inertial_optimization.cpp` | 记录惯性初始化快照和 LM 诊断；增加显式对照开关 | 支持固定输入定位初始化问题；实验开关默认不启用 |
| `libs/imu/lm_model_reduction.h` | 定义对称二次模型预测下降辅助函数 | 供 LM 模型数值一致性测试与显式对照 |
| `libs/imu/test/imu_boundary_test.cpp` | 新增约 166.7 ms 丢帧区间与转头速度测试 | 验证完整时间间隔、重新积分及世界系速度不被固定周期或旋转放大 |
| `libs/map/map.cpp` | 增加按观测 ID 取地图点接口与清图日志 | 诊断旧地图观测可用性和非空地图重置 |
| `libs/map/map.h` | 声明按观测取点接口 | 避免为诊断重复构造不同的地图查找逻辑 |
| `libs/map/test/map_test.cpp` | 增加地图观测查询回归 | 检查重复观测与不存在的 ID |
| `libs/odometry/multisensor_odometry.cpp` | 传递初始化期跨目匹配刷新请求并记录早期关键帧 | 低支持时增加视觉证据，同时保留明确的关键帧语义 |
| `libs/odometry/svo_config.h` | 增加本帧 stereo_refresh_only 参数 | 匹配刷新与新增关键帧分开控制 |
| `libs/pipelines/gpu_window/frontend_guard_adapter.h` | 通过实际相机模型计算像素残差及覆盖 | 原生鱼眼不能用虚拟针孔像素作为前台验收依据 |
| `libs/pipelines/gpu_window/window_backend.cpp` | 兼容双精度后验对照并参数化窗口快照区间 | 诊断精度链条而不绑定特定录像时间；发布默认保持已验证路径 |
| `libs/pipelines/gpu_window/window_linear.h` | 线性装配兼容 float/double 评价行并检查形状 | 防止后验与边缘化数据类型、维度隐式不一致 |
| `libs/pipelines/gpu_window/window_test.cpp` | 新增 FP64 后验与 3/4/6 状态对照 | 验证线性装配和先验变换的数值一致性 |
| `libs/pipelines/gpu_window/window_validation.h` | 数值有效性检查支持不同评价标量类型 | 避免更高精度路径绕过有限值检查 |
| `libs/pipelines/imu_fusion_context.cpp` | 加入合法陀螺视觉初值、同输入重试、初始化视觉验收和失败保图 | 减少初始化前拒绝即清图；通过原验收后才采纳候选，并保留时间锚点 |
| `libs/pipelines/imu_fusion_context.h` | 保存初始化视觉证据和保图控制状态 | 使候选、时间与接受状态生命周期明确 |
| `libs/pipelines/imu_initialization_validation.h` | 检查陀螺初值旋转与积分区间 | 不以异常旋转或失去覆盖的 IMU 区间作为视觉初值 |
| `libs/pipelines/init_visual_admission.h` | 验证初始化后的预测仍支持独立当前视觉观测 | 避免有限但几何不一致的惯性初始化状态直接发布 |
| `libs/pipelines/test/tracker_state_machine_test.cpp` | 增加初始化、保图、鱼眼像素和 LM 数值回归 | 防止恢复/初始化边界再次绕过验证 |
| `libs/pipelines/track_online_multisensor.cpp` | 执行低支持刷新和私有点缓存；传递初始化视觉要求并记录帧快照 | 增加下一帧可用证据，保留提交、恢复与前台的一致状态 |
| `libs/pipelines/track_online_multisensor.h` | 声明匹配刷新与私有缓存状态 | 使刷新请求不隐式改变跨帧关键帧选择 |
| `libs/pipelines/triangulation_validation.h` | 提供多观测三角化几何验证接口 | 供独立几何对照；发布未开启未通过全程验证的实验配置 |
| `libs/pipelines/triangulator.cpp` | 修复 unordered_set 被误当有序集合求交；过滤非法输入 | 防止可见缓存轨迹被误删除并重复三角化 |
| `libs/pipelines/triangulator.h` | 删除不再使用的集合交集临时容器 | 与基于成员查询的正确缓存更新一致 |
| `libs/pipelines/visual_snapshot.h` | 增加受环境开关控制的视觉冻结输入导出 | 复现候选、初值、标定与观测，默认不产生快照 I/O |
| `libs/pnp/cauchy_parameters.h` | 定义旧代价和显式重试模式的参数适配 | 保持稳定路径及恢复事务可重现；不包含本轮未通过高度验证的原始导数修复 |
| `libs/pnp/multisensor_pose_estimator.cpp` | 保存最后候选和实际过滤观测；使用已验证视觉支持检查与损失参数适配 | 失败候选可复现，拒绝不污染调用方，并保留同输入恢复验证 |
| `libs/pnp/multisensor_pose_estimator.h` | 暴露诊断候选及观测的只读访问 | 初始化验收使用与求解一致的有效观测 |
| `libs/pnp/test/multisensor_pose_estimator_test.cpp` | 增加冻结输入、拒绝回写、支持与参数适配测试 | 验证位姿、相机中心和接受状态边界 |
| `libs/pnp/visual_support.h` | 集中视觉有效观测与几何支持检查 | 避免求解候选仅按参与数量判断有效 |
| `libs/sof/image_context.cpp` | 保存完整图像灰度统计并支持标定图块诊断 | 跨相机比较使用一致且可审计的图像数据 |
| `libs/sof/image_context.h` | 保存图像统计与缓存成员 | 保证逐帧图像状态生命周期明确 |
| `libs/sof/internal/calibrated_patch_warp.h` | 提供标定投影的局部图块变形辅助 | 供鱼眼跨相机匹配验证；实验不开启为默认 |
| `libs/sof/internal/calibrated_stereo_geometry.h` | 提供保留相机中心的标定极线搜索几何 | 供多相机几何对照，避免把不同视角当同心相机 |
| `libs/sof/internal/sof_multicamera_gpu.h` | 声明标定几何匹配对照与工作缓存 | 隔离可选对照的跨帧数据 |
| `libs/sof/sof_multicamera_base.cpp` | 跨目匹配可刷新而不强制新增关键帧 | 保持关键帧时间和初始化事件一致 |
| `libs/sof/sof_multicamera_gpu.cpp` | 接入可选标定匹配、图块复核与受控缓存 | 支持逐项固定输入诊断，同时保持发布默认配置可复现 |
| `libs/sof/test/image_test.cpp` | 补充图像统计/缓存生命周期回归 | 检查逐帧数据不会读取错误缓存 |
| `libs/sof/test/kf_selector_test.cpp` | 检查刷新匹配不篡改关键帧选择 | 避免恢复请求改变关键帧事件时序 |
| `libs/sof/test/st_tracker_test.cpp` | 补充标定匹配与图块几何测试 | 验证不同相机中心和图像边界 |
